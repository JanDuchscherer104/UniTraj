# myproject/parsing/parser.py

import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
from metadrive.scenario import utils as sd_utils
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from ..configs.path_config import PathConfig
from ..datasets import common_utils
from ..datasets.common_utils import (
    generate_mask,
    get_polyline_dir,
    interpolate_polyline,
)
from ..utils.base_config import CONSOLE
from .base_dataparser import BaseDataParser
from .common_utils import is_ddp
from .types import (
    BatchDict,
    DatasetItem,
    DynamicMapInfosDict,
    InternalFormatDict,
    MapInfosDict,
    ProcessedDataDict,
    RawScenarioDict,
    Stage,
    TrackInfosDict,
    TracksToPredictDict,
    object_type,
    polyline_type,
)

object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)


class DataParserConfig(BaseModel["DataParser"]):
    """
    Configuration for splitting and preprocessing raw ScenarioNet into cache shards.
    """

    stage: Stage = Field(Stage.TRAIN)
    """
    STAGE.TRAIN, STAGE.VAL, STAGE.TEST
    """

    # Where to find raw ScenarioNet folders (can be list for multiple splits)
    raw_data_dirs: List[Path] = Field(..., description="List of ScenarioNet root dirs")

    # Where to write HDF5/LMDB/Parquet shards
    cache_root: Path = Field(..., description="Root dir for all cache shards")

    # Number of parallel workers (-1 = all CPUs)
    num_workers: int = Field(-1, description="How many processes to spawn")

    # When True, ignore any existing cache and rebuild
    overwrite_cache: bool = Field(True, description="Rebuild cache even if found")

    # If True, only process `debug_samples` per split, single‐threaded
    is_debug: bool = Field(False, description="Debug: only few samples, no MP")

    # Data selection
    num_debug_samples: int = Field(
        10, description="Max scenarios per split when debug=True"
    )
    starting_frame: List[int] = Field(
        [0],
        description="History trajectory starts at this frame for each training dataset",
    )
    max_data_num: List[Optional[int]] = Field(
        [None],
        description="Maximum number of data for each training dataset, null means all data",
    )

    has_tqdm: bool = True
    """
    Whether to use tqdm progress bars for parallel processing.
    """

    # Trajectory configuration
    past_len: int = Field(21, description="History trajectory length, 2.1s")
    future_len: int = Field(60, description="Future trajectory length, 6s")
    trajectory_sample_interval: int = Field(
        1, description="Sample interval for the trajectory"
    )

    # Object and map configuration
    masked_attributes: List[str] = Field(
        ["z_axis", "size"], description="Attributes to be masked in the input"
    )

        center_offset_of_map: List[float] = Field(
        [30.0, 0.0], description="Center offset of the map"
    )

    # Processing configuration
    only_train_on_ego: bool = Field(False, description="Only train on AV")

    # Map processing configuration
    max_num_agents: int = Field(128, description="Maximum number of agents")
    max_num_roads: int = Field(256, description="Maximum number of road segments")
    manually_split_lane: bool = Field(
        False, description="Whether to manually split lane polylines"
    )
    map_range: float = Field(60.0, description="Range of the map in meters")
    max_points_per_lane: int = Field(
        20, description="Maximum number of points per lane segment"
    )
    point_sampled_interval: int = Field(
        1, description="Sampling interval for points in polylines"
    )
    vector_break_dist_thresh: float = Field(
        1.0, description="Distance threshold for breaking vectors"
    )
    num_points_each_polyline: int = Field(
        20, description="Number of points in each polyline"
    )

    paths: PathConfig = Field(default_factory=PathConfig)
    """
    Attributes (all of type Path):
        root: Path to the root of the project
        data: path to the data root
        checkpoints
        cache
        temp_dir
        split_data_output_dir
    """

    target: Type["DataParser"] = Field(default_factory=lambda: DataParser)

    def get_num_workers(self) -> int:
        total = cpu_count()
        return total if self.num_workers < 0 else min(max(1, self.num_workers), total)

    def setup_target(self, **kwargs) -> "DataParser":
        CONSOLE.set_prefix(self.__class__.__name__, "setup_target", self.stage)

        if self.is_debug:
            CONSOLE.log(
                "Debug mode → single worker, no cache (temp dir instead), limited to "
                f"{self.num_debug_samples} samples"
            )
            self.load_num_workers = 1
            self.use_cache = False
        else:
            # Use all available CPU cores if load_num_workers is < 0
            total_cpus = cpu_count()
            self.load_num_workers = (
                total_cpus
                if self.load_num_workers < 0
                else min(max(self.load_num_workers, 1), total_cpus)
            )
            CONSOLE.log(
                f"Using {self.load_num_workers}/{total_cpus} workers; "
                f"{'will' if self.use_cache else 'will [bold]not[/bold]'} use cache"
            )

        CONSOLE.unset_prefix()

        return self.target(self, **kwargs)


class DataParser(BaseDataParser):
    """
    Splits raw ScenarioNet summaries into shards and runs
    BaseDataset.process_data_chunk in parallel to build cache.
    """

    def preprocess(self, scenario: RawScenarioDict) -> Optional[InternalFormatDict]:
        """
        Preprocess a raw scenario dictionary into an internal format.

        Args:
            scenario (RawScenarioDict): Raw scenario data.

        Returns:
            Optional[InternalFormatDict]: Processed internal format dictionary, or None if invalid.
        """
        # Ensure metadata exists
        if "metadata" not in scenario:
            scenario["metadata"] = {}
        # Add dataset name if missing in metadata (should be added in process_data_chunk now)
        if "dataset" not in scenario["metadata"]:
            scenario["metadata"]["dataset"] = "unknown"  # Or handle differently

        # Ensure required top-level keys exist
        required_top_keys = ["dynamic_map_states", "tracks", "map_features", "metadata"]
        if any(key not in scenario for key in required_top_keys):
            CONSOLE.error(
                f"Scenario {scenario.get('metadata', {}).get('scenario_id', 'Unknown')} missing required top-level keys."
            )
            return None

        traffic_lights = scenario["dynamic_map_states"]
        tracks = scenario["tracks"]
        map_feat = scenario["map_features"]
        metadata = scenario["metadata"]

        # Ensure metadata has required fields
        required_meta_keys = ["scenario_id", "sdc_id"]
        if any(key not in metadata for key in required_meta_keys):
            CONSOLE.error(
                f"Scenario {metadata.get('scenario_id', 'Unknown')} missing required metadata fields (scenario_id, sdc_id)."
            )
            return None

        past_length = self.config.past_len
        future_length = self.config.future_len
        total_steps = past_length + future_length
        starting_fame = self.starting_frame  # Use self.starting_frame set in load_data
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config.trajectory_sample_interval
        frequency_mask = generate_mask(
            past_length - 1, total_steps, trajectory_sample_interval
        )

        track_infos = TrackInfosDict(
            object_id=[],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            object_type=[],
            trajs=[],
        )

        # Check if tracks is empty or None
        if not tracks:
            CONSOLE.warn(f"Scenario {metadata['scenario_id']} has no tracks.")
            # Decide if this is an error or just an empty scenario
            # return None # If empty tracks are invalid

        for k, v in tracks.items():
            # Ensure track value 'v' is a dictionary and has 'state' and 'type'
            if not isinstance(v, dict) or "state" not in v or "type" not in v:
                CONSOLE.warn(
                    f"Invalid track format for ID {k} in scenario {metadata['scenario_id']}. Skipping track."
                )
                continue
            state = v["state"]
            # Ensure state is a dictionary
            if not isinstance(state, dict):
                CONSOLE.warn(
                    f"Invalid state format for track ID {k} in scenario {metadata['scenario_id']}. Skipping track."
                )
                continue

            # Ensure all state arrays have at least 2 dimensions
            for key, value in state.items():
                if (
                    value is not None
                    and isinstance(value, np.ndarray)
                    and len(value.shape) == 1
                ):
                    state[key] = np.expand_dims(value, axis=-1)
                elif value is None:
                    CONSOLE.warn(
                        f"Track {k} has None state for key '{key}' in scenario {metadata['scenario_id']}"
                    )
                    # Handle None state, maybe skip track or fill with defaults?
                    # For now, let downstream code handle potential errors

            # Check if required keys exist and have data
            required_keys = [
                "position",
                "length",
                "width",
                "height",
                "heading",
                "velocity",
                "valid",
            ]
            if any(
                key not in state
                or state[key] is None
                or not isinstance(state[key], np.ndarray)
                or len(state[key]) == 0
                for key in required_keys
            ):
                CONSOLE.warn(
                    f"Track {k} missing required state data or has invalid format in scenario {metadata['scenario_id']}. Skipping track."
                )
                continue  # Skip this track

            # Ensure consistent lengths before concatenation (handle potential ragged arrays)
            try:
                # Find the minimum length among required state arrays
                min_len = min(len(state[key]) for key in required_keys)
                if min_len == 0:
                    CONSOLE.warn(
                        f"Track {k} has zero length for required state data in scenario {metadata['scenario_id']}. Skipping track."
                    )
                    continue
                all_state_list = [state[key][:min_len] for key in required_keys]
                all_state = np.concatenate(all_state_list, axis=-1)
            except (ValueError, TypeError) as e:
                CONSOLE.warn(
                    f"Error concatenating states for track {k} in scenario {metadata['scenario_id']}: {e}. Skipping track."
                )
                continue  # Skip this track

            # Pad if necessary BEFORE slicing
            current_len = all_state.shape[0]
            if current_len < ending_fame:
                pad_width = ((0, ending_fame - current_len), (0, 0))  # Pad at the end
                # Ensure padding uses valid values (e.g., 0 for state, but maintain validity=0)
                # A simple pad might be okay if downstream handles validity mask correctly
                all_state = np.pad(
                    all_state, pad_width, mode="constant", constant_values=0
                )
                # Explicitly set validity of padded steps to 0
                all_state[current_len:ending_fame, -1] = 0

            # Slice the required window
            all_state = all_state[starting_fame:ending_fame]

            if all_state.shape[0] != total_steps:
                CONSOLE.error(
                    f"Track {k} in scenario {metadata['scenario_id']} has incorrect shape after processing: "
                    f"{all_state.shape[0]} != {total_steps}. Skipping track."
                )
                continue  # Skip this track

            track_infos["object_id"].append(k)
            track_infos["object_type"].append(object_type[v["type"]])
            track_infos["trajs"].append(all_state)

        # Check if any valid tracks were processed
        if not track_infos["object_id"]:
            CONSOLE.warn(
                f"Scenario {metadata['scenario_id']} has no valid tracks after preprocessing."
            )
            return None  # No valid tracks to process

        try:
            track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)
        except ValueError as e:
            CONSOLE.error(
                f"Could not stack trajectories for scenario {metadata['scenario_id']}: {e}. Check for inconsistent shapes."
            )
            return None

        # Apply frequency mask
        track_infos["trajs"][..., -1] *= frequency_mask[np.newaxis]

        # Adjust timestamps if they exist
        if (
            "ts" in metadata
            and metadata["ts"] is not None
            and isinstance(metadata["ts"], (np.ndarray, list))
            and len(metadata["ts"]) > 0
        ):
            ts = np.array(metadata["ts"])  # Ensure numpy array
            # Pad timestamps similar to trajectory if needed
            current_ts_len = len(ts)
            if current_ts_len < ending_fame:
                ts = np.pad(
                    ts,
                    (0, ending_fame - current_ts_len),
                    mode="constant",
                    constant_values=np.nan,
                )  # Pad with NaN or last value
            metadata["ts"] = ts[starting_fame:ending_fame]
            # Ensure timestamps have the correct length
            if len(metadata["ts"]) != total_steps:
                CONSOLE.warn(
                    f"Timestamp length mismatch in {metadata['scenario_id']}: {len(metadata['ts'])} != {total_steps}. Using truncated/padded timestamps."
                )
                # Ensure it has total_steps length, padding if necessary
                ts_padded = np.full(total_steps, np.nan)
                copy_len = min(len(metadata["ts"]), total_steps)
                ts_padded[:copy_len] = metadata["ts"][:copy_len]
                metadata["ts"] = ts_padded

        else:
            CONSOLE.warn(
                f"Timestamps ('ts') not found, invalid, or empty in metadata for scenario {metadata['scenario_id']}. Cannot add 'timestamps_seconds'."
            )
            # Handle missing timestamps - maybe generate dummy ones or raise error?
            metadata["ts"] = np.full(total_steps, np.nan)  # Fill with NaN

        # x,y,z,type
        map_infos = MapInfosDict(
            lane=[],
            road_line=[],
            road_edge=[],
            stop_sign=[],
            crosswalk=[],
            speed_bump=[],
            lane_id=[],  # This seems misplaced, lane_id is usually per-lane, not global
            all_polylines=np.zeros((0, 7), dtype=np.float32),  # Initialize empty
        )
        polylines = []
        point_cnt = 0
        # Ensure map_feat is a dictionary
        if not isinstance(map_feat, dict):
            CONSOLE.warn(
                f"map_features is not a dictionary in scenario {metadata['scenario_id']}. Skipping map processing."
            )
            map_feat = {}  # Process with empty map

        for k, v in map_feat.items():
            # Ensure map feature value 'v' is a dictionary and has 'type'
            if not isinstance(v, dict) or "type" not in v:
                CONSOLE.warn(
                    f"Invalid map feature format for ID {k} in scenario {metadata['scenario_id']}. Skipping feature."
                )
                continue

            polyline_type_ = polyline_type[v["type"]]
            if polyline_type_ == 0:
                continue

            cur_info = {"id": k, "type": v["type"]}  # Store original type string
            polyline = None  # Initialize polyline

            try:  # Wrap polyline processing in try-except
                if polyline_type_ in [1, 2, 3]:  # Lane types
                    cur_info["speed_limit_mph"] = v.get(
                        "speed_limit_mph"
                    )  # Use .get for safety
                    cur_info["interpolating"] = v.get("interpolating")
                    cur_info["entry_lanes"] = v.get("entry_lanes")
                    # Simplified boundary handling (assuming structure is consistent or optional)
                    cur_info["left_boundary"] = v.get("left_neighbor", [])
                    cur_info["right_boundary"] = v.get("right_neighbor", [])
                    polyline = v.get("polyline")
                    if polyline is not None:
                        polyline = interpolate_polyline(polyline)
                    map_infos["lane"].append(cur_info)
                elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:  # Road line types
                    polyline = v.get(
                        "polyline", v.get("polygon")
                    )  # Try polyline then polygon
                    if polyline is not None:
                        polyline = interpolate_polyline(polyline)
                    map_infos["road_line"].append(cur_info)
                elif polyline_type_ in [15, 16]:  # Road edge types
                    polyline = v.get("polyline")
                    if polyline is not None:
                        polyline = interpolate_polyline(polyline)
                    cur_info["type_enum"] = (
                        7  # Override type enum? Check if this is correct.
                    )
                    map_infos["road_edge"].append(cur_info)  # Use road_edge key
                elif polyline_type_ == 17:  # Stop sign
                    cur_info["lane_ids"] = v.get("lane")
                    cur_info["position"] = v.get("position")
                    map_infos["stop_sign"].append(cur_info)
                    if cur_info["position"] is not None:
                        polyline = np.array(cur_info["position"])[
                            np.newaxis
                        ]  # Ensure it's a 2D array
                elif polyline_type_ == 18:  # Crosswalk
                    cur_info["polygon"] = v.get("polygon")
                    map_infos["crosswalk"].append(cur_info)
                    polyline = v.get("polygon")
                elif polyline_type_ == 19:  # Speed bump
                    cur_info["polygon"] = v.get("polygon")
                    map_infos["speed_bump"].append(cur_info)
                    polyline = v.get("polygon")
                else:
                    CONSOLE.warn(
                        f"Unhandled polyline type enum {polyline_type_} for ID {k} in scenario {metadata['scenario_id']}"
                    )
                    continue  # Skip unhandled types

                # Process polyline if found
                if (
                    polyline is None
                    or not isinstance(polyline, (np.ndarray, list))
                    or len(polyline) == 0
                ):
                    # CONSOLE.warn(f"Polyline data missing or empty for ID {k}, type {v['type']} in scenario {metadata['scenario_id']}")
                    cur_polyline = np.zeros(
                        (0, 7), dtype=np.float32
                    )  # Use empty array for consistency
                else:
                    # Ensure polyline is numpy array
                    polyline = np.array(polyline, dtype=np.float32)
                    # Check shape
                    if len(polyline.shape) != 2 or polyline.shape[1] < 2:
                        CONSOLE.warn(
                            f"Invalid polyline shape {polyline.shape} for ID {k}, type {v['type']} in scenario {metadata['scenario_id']}. Skipping."
                        )
                        cur_polyline = np.zeros((0, 7), dtype=np.float32)
                    else:
                        # Add Z-axis if missing
                        if polyline.shape[-1] == 2:
                            polyline = np.pad(
                                polyline, ((0, 0), (0, 1)), constant_values=0
                            )  # Pad Z with 0
                        elif polyline.shape[-1] > 3:
                            polyline = polyline[
                                :, :3
                            ]  # Keep only first 3 columns (x, y, z)

                        # Calculate direction
                        cur_polyline_dir = get_polyline_dir(polyline)
                        # Create type array
                        type_array = np.full(
                            (polyline.shape[0], 1), polyline_type_, dtype=np.float32
                        )
                        # Concatenate: [x, y, z, dir_x, dir_y, dir_z, type_enum]
                        cur_polyline = np.concatenate(
                            (polyline, cur_polyline_dir, type_array),
                            axis=-1,
                            dtype=np.float32,
                        )

                polylines.append(cur_polyline)
                cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
                point_cnt += len(cur_polyline)

            except Exception as e:
                CONSOLE.error(
                    f"Error processing map feature ID {k}, type {v['type']} in scenario {metadata['scenario_id']}: {e}"
                )
                # Append an empty polyline to maintain consistency if needed, or skip
                polylines.append(np.zeros((0, 7), dtype=np.float32))
                cur_info["polyline_index"] = (point_cnt, point_cnt)  # No points added

        # Concatenate all processed polylines
        if polylines:
            try:
                map_infos["all_polylines"] = np.concatenate(polylines, axis=0).astype(
                    np.float32
                )
            except ValueError:
                CONSOLE.warn(
                    f"Could not concatenate map polylines for scenario {metadata['scenario_id']}. Using empty map."
                )
                map_infos["all_polylines"] = np.zeros((0, 7), dtype=np.float32)
        else:
            # CONSOLE.warn(f"No valid map polylines found for scenario {metadata['scenario_id']}")
            map_infos["all_polylines"] = np.zeros((0, 7), dtype=np.float32)

        # Process dynamic map states (traffic lights)
        dynamic_map_infos = DynamicMapInfosDict(
            lane_id=[], state=[], stop_point=[]
        )  # Initialize with lists
        # Ensure traffic_lights is a dictionary
        if not isinstance(traffic_lights, dict):
            CONSOLE.warn(
                f"dynamic_map_states is not a dictionary in scenario {metadata['scenario_id']}. Skipping traffic light processing."
            )
            traffic_lights = {}

        for k, v in traffic_lights.items():
            try:
                # Ensure traffic light value 'v' is a dictionary
                if not isinstance(v, dict):
                    CONSOLE.warn(
                        f"Invalid traffic light format for ID {k} in scenario {metadata['scenario_id']}. Skipping."
                    )
                    continue

                # Ensure 'state' and 'object_state' exist
                object_states = v.get("state", {}).get("object_state", [])
                num_states = len(object_states)
                if num_states == 0:
                    continue  # Skip if no states observed

                lane_ids = [
                    str(v.get("lane", "UNKNOWN"))
                ] * num_states  # Use get with default
                states = object_states
                stop_points = v.get(
                    "stop_point", [0.0, 0.0, 0.0]
                )  # Use get with default

                # Ensure stop_points is a list of coordinates or a list of lists
                if not isinstance(stop_points, list) or (
                    stop_points and not isinstance(stop_points[0], (list, np.ndarray))
                ):
                    stop_points = [
                        list(stop_points)
                    ] * num_states  # Repeat the single stop point
                elif stop_points and isinstance(stop_points[0], np.ndarray):
                    stop_points = [
                        sp.tolist() for sp in stop_points
                    ]  # Convert numpy arrays to lists

                # Pad/truncate to total_steps
                current_len = len(states)
                if current_len < total_steps:
                    pad_len = total_steps - current_len
                    # Choose appropriate padding values (e.g., last known state, UNKNOWN state)
                    lane_ids.extend(
                        [lane_ids[-1]] * pad_len if lane_ids else ["UNKNOWN"] * pad_len
                    )
                    states.extend(
                        [states[-1]] * pad_len if states else ["UNKNOWN"] * pad_len
                    )  # Pad with last state or UNKNOWN
                    stop_points.extend(
                        [stop_points[-1]] * pad_len
                        if stop_points
                        else [[0.0, 0.0, 0.0]] * pad_len
                    )
                elif current_len > total_steps:
                    lane_ids = lane_ids[:total_steps]
                    states = states[:total_steps]
                    stop_points = stop_points[:total_steps]

                dynamic_map_infos["lane_id"].append(
                    np.array(lane_ids)
                )  # Store as 1D array per light
                dynamic_map_infos["state"].append(np.array(states))
                dynamic_map_infos["stop_point"].append(
                    np.array(stop_points)
                )  # Should be (total_steps, 3) or similar

            except Exception as e:
                CONSOLE.error(
                    f"Error processing dynamic map state for ID {k} in scenario {metadata['scenario_id']}: {e}"
                )
                continue  # Skip this traffic light

        # Construct the final InternalFormatDict
        ret = InternalFormatDict(
            track_infos=track_infos,
            dynamic_map_infos=dynamic_map_infos,
            map_infos=map_infos,
            # Add other metadata fields explicitly if they are part of InternalFormatDict
            scenario_id=metadata["scenario_id"],
            timestamps_seconds=metadata.get(
                "ts", np.full(total_steps, np.nan)
            ),  # Use get with default
            sdc_id=metadata["sdc_id"],
            # Add other optional fields from metadata using .get
            source_id=metadata.get("source_id"),
            version=metadata.get("version"),
            objects_of_interest=metadata.get("objects_of_interest"),
            tracks_to_predict=metadata.get(
                "tracks_to_predict"
            ),  # Keep original if present
            dataset=metadata.get("dataset"),
            map_center=metadata.get("map_center", np.zeros(3))[
                np.newaxis
            ],  # Add default
            # Calculated fields
            current_time_index=self.config.past_len - 1,
            track_length=total_steps,
            sdc_track_index=-1,  # Initialize SDC index
        )

        # Find SDC track index safely
        try:
            ret["sdc_track_index"] = track_infos["object_id"].index(ret["sdc_id"])
        except (ValueError, IndexError):
            CONSOLE.error(
                f"SDC ID {ret['sdc_id']} not found in processed tracks for scenario {ret['scenario_id']}. Setting sdc_track_index to -1."
            )
            # Decide how to handle this: return None, or proceed with index -1?
            # return None # If SDC must be present

        # Determine tracks_to_predict
        tracks_to_predict_output = TracksToPredictDict(
            track_index=[], object_type=[], difficulty=[]
        )  # Initialize empty

        if self.config.only_train_on_ego:
            if ret["sdc_track_index"] != -1:
                tracks_to_predict_output["track_index"].append(ret["sdc_track_index"])
                # Assuming SDC is always VEHICLE, get type dynamically if possible
                sdc_obj_type_enum = track_infos["object_type"][ret["sdc_track_index"]]
                tracks_to_predict_output["object_type"].append(sdc_obj_type_enum)
                tracks_to_predict_output["difficulty"].append(
                    0
                )  # Default difficulty for SDC
            else:
                CONSOLE.warn(
                    f"only_train_on_ego is True, but SDC track index is invalid for scenario {ret['scenario_id']}"
                )
        else:
            candidate_ids = []
            source_of_candidates = "metadata"  # Track where the candidates came from

            # Prefer tracks_to_predict from metadata if available and valid format
            metadata_ttp = ret.get("tracks_to_predict")
            if (
                isinstance(metadata_ttp, dict) and "track_index" in metadata_ttp
            ):  # Check if it's the processed dict format
                # If it's already processed, use it directly (shouldn't happen here, but defensive)
                tracks_to_predict_output = metadata_ttp
                candidate_ids = None  # Mark that we don't need to process candidates
            elif isinstance(
                metadata_ttp, dict
            ):  # Original format {obj_id: {difficulty: x}}
                candidate_ids = list(metadata_ttp.keys())
            # Fallback to objects_of_interest if metadata tracks_to_predict is missing/invalid
            elif ret.get("objects_of_interest") is not None and isinstance(
                ret["objects_of_interest"], list
            ):
                candidate_ids = ret["objects_of_interest"]
                source_of_candidates = "objects_of_interest"
            # Fallback to filtering all tracks if neither is available
            else:
                # Consider all tracks initially
                candidate_ids = track_infos["object_id"]
                source_of_candidates = "all_tracks"

            # Filter candidates if needed
            if candidate_ids is not None:
                valid_indices = []
                valid_types = []
                valid_difficulties = []
                selected_type_enums = [object_type[x] for x in self.config.object_type]

                for obj_id in candidate_ids:
                    try:
                        idx = track_infos["object_id"].index(obj_id)
                        obj_type_enum = track_infos["object_type"][idx]
                        # Check if object type is in the configured list
                        if obj_type_enum in selected_type_enums:
                            # Check validity at current time index
                            if (
                                track_infos["trajs"][idx, ret["current_time_index"], -1]
                                > 0
                            ):
                                valid_indices.append(idx)
                                valid_types.append(obj_type_enum)
                                # Get difficulty if available from original metadata_ttp
                                difficulty = 0  # Default
                                if (
                                    isinstance(metadata_ttp, dict)
                                    and obj_id in metadata_ttp
                                    and isinstance(metadata_ttp[obj_id], dict)
                                ):
                                    difficulty = metadata_ttp[obj_id].get(
                                        "difficulty", 0
                                    )
                                valid_difficulties.append(difficulty)
                            # else: CONSOLE.warn(f"Candidate {obj_id} from {source_of_candidates} is not valid at current time.")
                        # else: CONSOLE.warn(f"Candidate {obj_id} from {source_of_candidates} has type {obj_type_enum}, which is not in selected types.")
                    except ValueError:
                        # CONSOLE.warn(f"Candidate object ID {obj_id} from {source_of_candidates} not found in processed tracks.")
                        pass  # Ignore IDs not found

                tracks_to_predict_output["track_index"] = valid_indices
                tracks_to_predict_output["object_type"] = valid_types
                tracks_to_predict_output["difficulty"] = valid_difficulties

        # Final check if any tracks are selected for prediction
        if not tracks_to_predict_output["track_index"]:
            CONSOLE.warn(
                f"No valid tracks selected for prediction in scenario {ret['scenario_id']}"
            )
            # Decide if this is critical
            # return None # If prediction targets are required

        ret["tracks_to_predict"] = tracks_to_predict_output  # Assign the processed dict

        return ret

    def process(
        self, internal_format: InternalFormatDict
    ) -> Optional[List[DatasetItem]]:
        """
        Process internal format data to generate model inputs for each interested agent.

        Args:
            internal_format (InternalFormatDict): Data in internal format.

        Returns:
            Optional[List[DatasetItem]]: List of processed data items, one per interested agent.
        """
        # Debug: overview of incoming data
        CONSOLE.dbg(f"process() for scenario {internal_format['scenario_id']} got")
        CONSOLE.plog(
            {
                "len(num_tracks)": len(internal_format["track_infos"]["object_id"]),
                "traj_shape": (
                    internal_format["track_infos"]["trajs"].shape
                    if "trajs" in internal_format["track_infos"]
                    else None
                ),
                "map_polylines_shape": internal_format["map_infos"][
                    "all_polylines"
                ].shape,
                "dynamic_map_count": len(
                    internal_format["dynamic_map_infos"]["lane_id"]
                ),
            }
        )
        info = internal_format
        scene_id = info["scenario_id"]

        sdc_track_index = info["sdc_track_index"]
        current_time_index = info["current_time_index"]
        timestamps = np.array(
            info["timestamps_seconds"][: current_time_index + 1], dtype=np.float32
        )

        track_infos = info["track_infos"]
        track_index_to_predict = np.array(info["tracks_to_predict"]["track_index"])

        # Check if there are any agents to predict
        if len(track_index_to_predict) == 0:
            CONSOLE.warn(f"No agents to predict in process() for scenario {scene_id}")
            return None  # Return None if no agents are selected

        obj_types = np.array(track_infos["object_type"])
        obj_trajs_full = track_infos["trajs"]  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, : current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1 :]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types,
            scene_id=scene_id,
        )
        if center_objects is None:
            CONSOLE.warn(
                f"get_interested_agents() returned type(center_objects)={center_objects} (expected np.ndarray) for scenario {scene_id}"
            )
            return None

        sample_num = center_objects.shape[0]

        # Get agent data relative to the valid center objects
        (
            obj_trajs_data,
            obj_trajs_mask,
            obj_trajs_pos,
            obj_trajs_last_pos,
            obj_trajs_future_state,
            obj_trajs_future_mask,
            center_gt_trajs,
            center_gt_trajs_mask,
            center_gt_final_valid_idx,
            track_index_to_predict_new,
        ) = self.get_agent_data(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict,  # used to select center-features
            sdc_track_index=sdc_track_index,
            timestamps=timestamps,
            obj_types=obj_types,
        )

        # Debug: shapes of agent data arrays
        if self.config.is_debug:
            CONSOLE.dbg("get_agent_data() outputs arrays of shapes:")
            CONSOLE.plog(
                {
                    "obj_trajs_data": obj_trajs_data.shape,
                    "obj_trajs_mask": obj_trajs_mask.shape,
                    "obj_trajs_pos": obj_trajs_pos.shape,
                    "obj_trajs_last_pos": obj_trajs_last_pos.shape,
                    "obj_trajs_future_state": obj_trajs_future_state.shape,
                    "obj_trajs_future_mask": obj_trajs_future_mask.shape,
                    "center_gt_trajs": center_gt_trajs.shape,
                    "center_gt_trajs_mask": center_gt_trajs_mask.shape,
                    "center_gt_final_valid_idx": center_gt_final_valid_idx.shape,
                    "track_index_to_predict_new": track_index_to_predict_new.shape,
                }
            )

        # Prepare the dictionary for processed data
        ret_dict: ProcessedDataDict = {
            "scenario_id": np.array([scene_id] * len(track_index_to_predict)),
            "obj_trajs": obj_trajs_data,
            "obj_trajs_mask": obj_trajs_mask,
            "track_index_to_predict": track_index_to_predict_new,  # used to select center-features
            "obj_trajs_pos": obj_trajs_pos,
            "obj_trajs_last_pos": obj_trajs_last_pos,
            "center_objects_world": center_objects,
            "center_objects_id": np.array(track_infos["object_id"])[
                track_index_to_predict
            ],
            "center_objects_type": np.array(track_infos["object_type"])[
                track_index_to_predict
            ],
            "map_center": info["map_center"],
            "obj_trajs_future_state": obj_trajs_future_state,
            "obj_trajs_future_mask": obj_trajs_future_mask,
            "center_gt_trajs": center_gt_trajs,
            "center_gt_trajs_mask": center_gt_trajs_mask,
            "center_gt_final_valid_idx": center_gt_final_valid_idx,
            "center_gt_trajs_src": obj_trajs_full[track_index_to_predict],
        }

        if len(info["map_infos"]["all_polylines"]) == 0:
            info["map_infos"]["all_polylines"] = np.zeros((2, 7), dtype=np.float32)
            CONSOLE.log(f"Empty HDMap (zero polylines) for {scene_id}")

        if self.config.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = (
                self.get_manually_split_map_data(
                    center_objects=center_objects, map_infos=info["map_infos"]
                )
            )
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center = (
                self.get_map_data(
                    center_objects=center_objects, map_infos=info["map_infos"]
                )
            )

        ret_dict["map_polylines"] = map_polylines_data
        ret_dict["map_polylines_mask"] = map_polylines_mask.astype(bool)
        ret_dict["map_polylines_center"] = map_polylines_center

        # masking out unused attributes to Zero
        masked_attributes = self.config.masked_attributes
        if "z_axis" in masked_attributes:
            ret_dict["obj_trajs"][..., 2] = 0
            ret_dict["map_polylines"][..., 2] = 0
        if "size" in masked_attributes:
            ret_dict["obj_trajs"][..., 3:6] = 0
        if "velocity" in masked_attributes:
            ret_dict["obj_trajs"][..., 25:27] = 0
        if "acceleration" in masked_attributes:
            ret_dict["obj_trajs"][..., 27:29] = 0
        if "heading" in masked_attributes:
            ret_dict["obj_trajs"][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict["map_center"] = ret_dict["map_center"].repeat(sample_num, axis=0)
        ret_dict["dataset_name"] = [info["dataset"]] * sample_num

        # Convert each sample to a DatasetItem instance
        ret_list = []
        for i in range(sample_num):
            # Extract data for this sample
            sample_data = {}
            for k, v in ret_dict.items():
                if isinstance(v, list):
                    sample_data[k] = (
                        np.array(v[i], dtype="S") if isinstance(v[i], str) else v[i]
                    )
                else:
                    sample_data[k] = v[i]

            # Convert string fields to bytes for HDF5 compatibility
            if isinstance(sample_data["scenario_id"], str):
                sample_data["scenario_id"] = np.array(
                    sample_data["scenario_id"], dtype="S"
                )
            if isinstance(sample_data["dataset_name"], str):
                sample_data["dataset_name"] = np.array(
                    sample_data["dataset_name"], dtype="S"
                )
            if isinstance(sample_data["center_objects_id"], str):
                sample_data["center_objects_id"] = np.array(
                    sample_data["center_objects_id"], dtype="S"
                )

            # Create DatasetItem instance
            try:
                dataset_item = DatasetItem(**sample_data)
                ret_list.append(dataset_item)
            except Exception as e:
                CONSOLE.error(f"Error creating DatasetItem for sample {i}: {e}")
                # Log the keys and shapes to help debug
                shapes = {
                    k: (v.shape if isinstance(v, np.ndarray) else type(v))
                    for k, v in sample_data.items()
                }
                CONSOLE.error(f"Sample data keys and shapes: {shapes}")
                # Continue to next sample instead of failing completely
                continue

        return ret_list

    def postprocess(self, output: List[DatasetItem]) -> Optional[List[DatasetItem]]:
        """
        Perform post-processing steps like calculating difficulty and trajectory type.

        Args:
            output (List[DatasetItem]): List of processed data items from the process() step.

        Returns:
            Optional[List[DatasetItem]]: List of data items with added post-processing info, or None if input is invalid.
        """
        if not output:  # Handle empty list case
            CONSOLE.warn("postprocess received an empty list.")
            return None

        # Process each DatasetItem
        for item in output:
            # Calculate Kalman difficulty
            try:
                # Check if we have the necessary data for Kalman difficulty calculation
                future_traj = item.center_gt_trajs
                future_mask = item.center_gt_trajs_mask

                # Calculate difficulty if we have valid data
                if (
                    future_traj is not None
                    and future_mask is not None
                    and np.any(future_mask)
                ):
                    # Get only the valid future trajectory points
                    valid_indices = np.where(future_mask)[0]
                    if len(valid_indices) >= 2:  # Need at least 2 points
                        valid_traj = future_traj[valid_indices]
                        difficulty = common_utils.get_kalman_difficulty_single(
                            valid_traj
                        )
                        item.kalman_difficulty = np.array(
                            [difficulty], dtype=np.float32
                        )
                    else:
                        item.kalman_difficulty = np.array([0.0], dtype=np.float32)
                else:
                    item.kalman_difficulty = np.array([0.0], dtype=np.float32)
            except Exception as e:
                CONSOLE.warn(f"Error calculating Kalman difficulty: {e}")
                item.kalman_difficulty = np.array([0.0], dtype=np.float32)

            # Determine trajectory type
            try:
                # Check if we have the necessary data for trajectory type calculation
                future_traj = item.center_gt_trajs
                future_mask = item.center_gt_trajs_mask

                if (
                    future_traj is not None
                    and future_mask is not None
                    and np.any(future_mask)
                ):
                    # Find segments of valid points
                    segments = common_utils.find_true_segments(future_mask)
                    if segments:
                        longest_segment = max(segments, key=lambda s: s[1] - s[0])
                        if (
                            longest_segment[1] - longest_segment[0] >= 3
                        ):  # Need at least 3 consecutive points
                            start_idx, end_idx = longest_segment
                            traj_segment = future_traj[start_idx:end_idx]
                            traj_type = common_utils.get_trajectory_type_single(
                                traj_segment
                            )
                            item.trajectory_type = traj_type
                        else:
                            item.trajectory_type = 0  # Stationary type as default
                    else:
                        item.trajectory_type = 0  # Stationary type as default
                else:
                    item.trajectory_type = 0  # Stationary type as default
            except Exception as e:
                CONSOLE.warn(f"Error determining trajectory type: {e}")
                item.trajectory_type = 0  # Stationary type as default

        return output

    def get_agent_data(
        self,
        center_objects: np.ndarray,
        obj_trajs_past: np.ndarray,
        obj_trajs_future: np.ndarray,
        track_index_to_predict: np.ndarray,  # Indices relative to obj_trajs_full
        sdc_track_index: int,
        timestamps: np.ndarray,
        obj_types: np.ndarray,
    ) -> Tuple[
        np.ndarray,  # obj_trajs_data
        np.ndarray,  # obj_trajs_mask
        np.ndarray,  # obj_trajs_pos
        np.ndarray,  # obj_trajs_last_pos
        np.ndarray,  # obj_trajs_future_state
        np.ndarray,  # obj_trajs_future_mask
        np.ndarray,  # center_gt_trajs
        np.ndarray,  # center_gt_trajs_mask
        np.ndarray,  # center_gt_final_valid_idx
        np.ndarray,  # track_index_to_predict_new (indices relative to filtered agents)
    ]:
        """
        Prepare agent-centric trajectory and mask data for model input. Used in `process()`.

        Args:
            center_objects (np.ndarray['num_center_objects, 10', float32]): Center agent states at current time.
            obj_trajs_past (np.ndarray['num_objects, num_past_timestamps, box_dim', float32]): Past trajectories of all objects.
            obj_trajs_future (np.ndarray['num_objects, num_future_timestamps, box_dim', float32]): Future trajectories of all objects.
            track_index_to_predict (np.ndarray['num_center_objects', int]): Original indices of center agents relative to obj_trajs_full.
            sdc_track_index (int): Original index of the self-driving car.
            timestamps (np.ndarray['num_past_timestamps', float32]): Timestamps for each past step.
            obj_types (np.ndarray['num_objects', int]): Object types for all objects.

        Returns:
            Tuple of arrays containing processed agent data, filtered and padded.
                - obj_trajs_data (np.ndarray)
                - obj_trajs_mask (np.ndarray)
                - obj_trajs_pos (np.ndarray)
                - obj_trajs_last_pos (np.ndarray)
                - obj_trajs_future_state (np.ndarray)
                - obj_trajs_future_mask (np.ndarray)
                - center_gt_trajs (np.ndarray)
                - center_gt_trajs_mask (np.ndarray)
                - center_gt_final_valid_idx (np.ndarray)
                - track_index_to_predict_new (np.ndarray)
        """

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6,
            rot_vel_index=[7, 8],
        )

        object_onehot_mask = np.zeros(
            (num_center_objects, num_objects, num_timestamps, 5)
        )
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[
            np.arange(num_center_objects), track_index_to_predict, :, 3
        ] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros(
            (num_center_objects, num_objects, num_timestamps, num_timestamps + 1)
        )
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros(
            (num_center_objects, num_objects, num_timestamps, 2)
        )
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate(
            [
                obj_trajs[:, :, :, 0:6],
                object_onehot_mask,
                object_time_embedding,
                object_heading_embedding,
                obj_trajs[:, :, :, 7:9],
                acce,
            ],
            axis=-1,
        )

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6,
            rot_vel_index=[7, 8],
        )
        obj_trajs_future_state = obj_trajs_future[
            :, :, :, [0, 1, 7, 8]
        ]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[
            center_obj_idxs, track_index_to_predict
        ]
        center_gt_trajs_mask = obj_trajs_future_mask[
            center_obj_idxs, track_index_to_predict
        ]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros(
            (num_center_objects, num_objects, 3), dtype=np.float32
        )
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][
                cur_valid_mask
            ]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config.max_num_agents
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(
            obj_trajs_last_pos, topk_idxs[..., 0], axis=1
        )
        obj_trajs_future_state = np.take_along_axis(
            obj_trajs_future_state, topk_idxs, axis=1
        )
        obj_trajs_future_mask = np.take_along_axis(
            obj_trajs_future_mask, topk_idxs[..., 0], axis=1
        )
        track_index_to_predict_new = np.zeros(
            len(track_index_to_predict), dtype=np.int64
        )

        obj_trajs_data = np.pad(
            obj_trajs_data,
            ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)),
        )
        obj_trajs_mask = np.pad(
            obj_trajs_mask,
            ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)),
        )
        obj_trajs_pos = np.pad(
            obj_trajs_pos,
            ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)),
        )
        obj_trajs_last_pos = np.pad(
            obj_trajs_last_pos,
            ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)),
        )
        obj_trajs_future_state = np.pad(
            obj_trajs_future_state,
            (
                (0, 0),
                (0, max_num_agents - obj_trajs_future_state.shape[1]),
                (0, 0),
                (0, 0),
            ),
        )
        obj_trajs_future_mask = np.pad(
            obj_trajs_future_mask,
            ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)),
        )

        return (
            obj_trajs_data,
            obj_trajs_mask.astype(bool),
            obj_trajs_pos,
            obj_trajs_last_pos,
            obj_trajs_future_state,
            obj_trajs_future_mask,
            center_gt_trajs,
            center_gt_trajs_mask,
            center_gt_final_valid_idx,
            track_index_to_predict_new,
        )

    def get_manually_split_map_data(
        self, center_objects: np.ndarray, map_infos: MapInfosDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare map polyline features relative to each center agent, manually splitting long lanes. Used in `process()`.

        Args:
            center_objects (np.ndarray['num_center_objects, 10', float32]): Center agent states.
            map_infos (MapInfosDict): Map information containing 'all_polylines'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, num_feat): Padded map features.
                - map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline): Boolean mask.
                - map_polylines_center (num_center_objects, num_topk_polylines, 3): Center of each polyline segment.
        """
        num_center_objects = center_objects.shape[0]
        all_polylines_orig = map_infos[
            "all_polylines"
        ]  # Shape (total_points, 7) [x, y, z, dx, dy, dz, type]

        # Default feature dimension
        feat_dim = 29  # pos(3)+dir(3)+pre_pos(3)+type_onehot(20)

        if all_polylines_orig is None or all_polylines_orig.shape[0] == 0:
            CONSOLE.warn(
                "get_manually_split_map_data received empty 'all_polylines'. Returning dummy data."
            )
            num_poly = self.config.max_num_roads
            num_pts = self.config.num_points_each_polyline
            map_polylines_data = np.zeros(
                (num_center_objects, num_poly, num_pts, feat_dim), dtype=np.float32
            )
            map_polylines_mask = np.zeros(
                (num_center_objects, num_poly, num_pts), dtype=bool
            )
            map_polylines_center = np.zeros(
                (num_center_objects, num_poly, 3), dtype=np.float32
            )
            return map_polylines_data, map_polylines_mask, map_polylines_center

        point_dim = all_polylines_orig.shape[-1]  # Should be 7
        point_sampled_interval = self.config.point_sampled_interval
        vector_break_dist_thresh = self.config.vector_break_dist_thresh
        num_points_each_polyline = self.config.num_points_each_polyline

        # --- 1. Sample points and identify breaks ---
        sampled_points = all_polylines_orig[::point_sampled_interval]
        if len(sampled_points) <= 1:  # Handle case with 0 or 1 point
            polyline_list = [sampled_points] if len(sampled_points) > 0 else []
        else:
            sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
            # Calculate distance between consecutive sampled points
            dist_diff = np.linalg.norm(
                sampled_points[:, 0:2] - sampled_points_shift[:, 0:2], axis=-1
            )
            dist_diff[0] = 0  # First point has no preceding point

            break_idxs = np.where(dist_diff > vector_break_dist_thresh)[0]
            polyline_list = np.array_split(sampled_points, break_idxs, axis=0)

        # --- 2. Split segments into fixed size chunks ---
        ret_polylines = []
        ret_polylines_mask = []

        for segment in polyline_list:
            if len(segment) == 0:
                continue
            # Split each segment into chunks of size num_points_each_polyline
            for i in range(0, len(segment), num_points_each_polyline):
                chunk = segment[i : i + num_points_each_polyline]
                num_actual_points = len(chunk)

                padded_chunk = np.zeros(
                    (num_points_each_polyline, point_dim), dtype=np.float32
                )
                mask = np.zeros(num_points_each_polyline, dtype=bool)

                padded_chunk[:num_actual_points] = chunk
                mask[:num_actual_points] = True

                ret_polylines.append(padded_chunk)
                ret_polylines_mask.append(mask)

        if not ret_polylines:
            CONSOLE.warn(
                "No valid polyline segments after splitting. Returning dummy data."
            )
            # Return dummy data
            num_poly = self.config.max_num_roads
            num_pts = self.config.num_points_each_polyline
            map_polylines_data = np.zeros(
                (num_center_objects, num_poly, num_pts, feat_dim), dtype=np.float32
            )
            map_polylines_mask = np.zeros(
                (num_center_objects, num_poly, num_pts), dtype=bool
            )
            map_polylines_center = np.zeros(
                (num_center_objects, num_poly, 3), dtype=np.float32
            )
            return map_polylines_data, map_polylines_mask, map_polylines_center

        batch_polylines = np.stack(
            ret_polylines, axis=0
        )  # (num_total_segments, num_points_each, 7)
        batch_polylines_mask = np.stack(
            ret_polylines_mask, axis=0
        )  # (num_total_segments, num_points_each)

        # --- 3. Select Top-K Segments based on distance ---
        num_total_segments = batch_polylines.shape[0]
        num_segments_to_keep = min(num_total_segments, self.config.max_num_roads)

        # Calculate center of each segment
        masked_sum = np.sum(
            batch_polylines[:, :, 0:3] * batch_polylines_mask[:, :, np.newaxis], axis=1
        )
        point_counts = np.sum(batch_polylines_mask, axis=1, keepdims=True)
        segment_centers = masked_sum / np.clip(
            point_counts, a_min=1, a_max=None
        )  # (num_total_segments, 3)

        # Calculate distance from agent centers to segment centers
        center_offset = np.array(self.config.center_offset_of_map, dtype=np.float32)
        center_offset_rot = common_utils.rotate_points_along_z(
            np.tile(center_offset[np.newaxis, :], (num_center_objects, 1)),
            center_objects[:, 6],
        )
        agent_map_centers = center_objects[:, 0:2] + center_offset_rot
        dist = np.linalg.norm(
            agent_map_centers[:, np.newaxis, :] - segment_centers[np.newaxis, :, 0:2],
            axis=-1,
        )  # (num_center, num_total_segments)

        # Get top-k indices
        topk_idxs = np.argsort(dist, axis=1)[
            :, :num_segments_to_keep
        ]  # (num_center, num_segments_to_keep)

        # Gather top-k segments
        # Need to handle potential empty topk_idxs if num_segments_to_keep is 0
        if num_segments_to_keep > 0:
            map_polylines_topk = np.take_along_axis(
                batch_polylines[np.newaxis, :, :, :],
                topk_idxs[:, :, np.newaxis, np.newaxis],
                axis=1,
            )
            map_polylines_mask_topk = np.take_along_axis(
                batch_polylines_mask[np.newaxis, :, :],
                topk_idxs[:, :, np.newaxis],
                axis=1,
            )
        else:
            map_polylines_topk = np.zeros(
                (num_center_objects, 0, num_points_each_polyline, point_dim),
                dtype=np.float32,
            )
            map_polylines_mask_topk = np.zeros(
                (num_center_objects, 0, num_points_each_polyline), dtype=bool
            )

        # --- 4. Transform Top-K Segments to Agent Coordinates ---
        map_polylines_rel = map_polylines_topk.copy()
        map_polylines_rel[:, :, :, 0:3] -= center_objects[
            :, np.newaxis, np.newaxis, 0:3
        ]
        if num_segments_to_keep > 0:  # Only rotate if there are segments
            pos_to_rotate = map_polylines_rel[:, :, :, 0:2].reshape(
                num_center_objects, -1, 2
            )
            dir_to_rotate = map_polylines_rel[:, :, :, 3:5].reshape(
                num_center_objects, -1, 2
            )
            rotated_pos = common_utils.rotate_points_along_z(
                pos_to_rotate, -center_objects[:, 6]
            )
            rotated_dir = common_utils.rotate_points_along_z(
                dir_to_rotate, -center_objects[:, 6]
            )
            map_polylines_rel[:, :, :, 0:2] = rotated_pos.reshape(
                num_center_objects, num_segments_to_keep, num_points_each_polyline, 2
            )
            map_polylines_rel[:, :, :, 3:5] = rotated_dir.reshape(
                num_center_objects, num_segments_to_keep, num_points_each_polyline, 2
            )

        # --- 5. Create Final Features ---
        xy_pos_pre = np.roll(map_polylines_rel[:, :, :, 0:3], shift=1, axis=-2)
        if num_segments_to_keep > 0:  # Avoid index error on empty array
            xy_pos_pre[:, :, 0, :] = map_polylines_rel[:, :, 0, 0:3]
        map_types_enum = np.clip(map_polylines_rel[:, :, :, 6].astype(int), 0, 19)
        map_types_onehot = np.eye(20, dtype=np.float32)[map_types_enum]
        map_polylines_final = np.concatenate(
            (
                map_polylines_rel[:, :, :, 0:3],  # Rel pos
                map_polylines_rel[:, :, :, 3:6],  # Rel dir
                xy_pos_pre,  # Prev rel pos
                map_types_onehot,
            ),
            axis=-1,
            dtype=np.float32,
        )
        map_polylines_final[map_polylines_mask_topk == 0] = 0

        # --- 6. Pad to max_num_roads ---
        pad_width_segments = self.config.max_num_roads - num_segments_to_keep
        if pad_width_segments > 0:
            pad_arg_4d = ((0, 0), (0, pad_width_segments), (0, 0), (0, 0))
            pad_arg_3d = ((0, 0), (0, pad_width_segments), (0, 0))
            map_polylines_final = np.pad(
                map_polylines_final, pad_arg_4d, mode="constant", constant_values=0
            )
            map_polylines_mask_final = np.pad(
                map_polylines_mask_topk, pad_arg_3d, mode="constant", constant_values=0
            )
        else:
            map_polylines_mask_final = map_polylines_mask_topk

        # --- 7. Calculate Segment Centers (relative coordinates) ---
        temp_sum = (
            map_polylines_final[:, :, :, 0:3]
            * map_polylines_mask_final[:, :, :, np.newaxis].astype(float)
        ).sum(axis=-2)
        map_polylines_center = temp_sum / np.clip(
            map_polylines_mask_final.sum(axis=-1).astype(float)[:, :, np.newaxis],
            a_min=1.0,
            a_max=None,
        )

        return (
            map_polylines_final,
            map_polylines_mask_final.astype(bool),
            map_polylines_center,
        )

    def get_map_data(
        self, center_objects: np.ndarray, map_infos: MapInfosDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare map polyline features relative to each center agent, using fixed-size points per polyline. Used in `process()`.

        Args:
            center_objects (np.ndarray['num_center_objects, 10', float32]): Center agent states.
            map_infos (MapInfosDict): Map information containing 'all_polylines'.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - map_polylines (num_center_objects, num_topk_polylines, max_points_per_lane, num_feat): Padded map features.
                - map_polylines_mask (num_center_objects, num_topk_polylines, max_points_per_lane): Boolean mask.
                - map_polylines_center (num_center_objects, num_topk_polylines, 3): Center of each polyline.
        """
        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2], angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5], angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        polylines = np.expand_dims(map_infos["all_polylines"].copy(), axis=0).repeat(
            num_center_objects, axis=0
        )

        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config.max_num_roads
        map_infos["polyline_transformed"] = map_polylines

        all_polylines = map_infos["polyline_transformed"]
        max_points_per_lane = self.config.max_points_per_lane
        line_type = self.config.line_type
        map_range = self.config.map_range
        center_offset = self.config.center_offset_of_map
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == "all_polylines" or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get("polyline_index", None)
                polyline_segment = all_polylines[
                    :, polyline_index[0] : polyline_index[1]
                ]
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                in_range_mask = (abs(polyline_segment_x) < map_range) * (
                    abs(polyline_segment_y) < map_range
                )

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros(
                    [num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32
                )
                segment_mask_list = np.zeros(
                    [num_agents, max_segments, max_points_per_lane], dtype=np.int32
                )

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(
                                    0,
                                    segment.shape[0] - 1,
                                    max_points_per_lane,
                                    dtype=int,
                                )
                            ]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, : segment.shape[0]] = segment
                            segment_mask_list[i, num, : segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if len(polyline_list) == 0:
            return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
                (num_agents, 0, max_points_per_lane)
            )
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(
            center_offset, (1, 1, 1, 2)
        )
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(
            -1
        ) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None
        )
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Ensure topk_idxs has the correct shape for indexing
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(
            batch_polylines_mask, topk_idxs[..., 0], axis=1
        )

        # pad map_polylines and map_polylines_mask to num_of_src_polylines
        map_polylines = np.pad(
            map_polylines,
            (
                (0, 0),
                (0, num_of_src_polylines - map_polylines.shape[1]),
                (0, 0),
                (0, 0),
            ),
        )
        map_polylines_mask = np.pad(
            map_polylines_mask,
            ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)),
        )

        temp_sum = (
            map_polylines[:, :, :, 0:3]
            * map_polylines_mask[:, :, :, None].astype(float)
        ).sum(
            axis=-2
        )  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(
            map_polylines_mask.sum(axis=-1).astype(float)[:, :, None],
            a_min=1.0,
            a_max=None,
        )  # (num_center_objects, num_polylines, 3)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center

    def get_interested_agents(
        self,
        track_index_to_predict: np.ndarray,
        obj_trajs_full: np.ndarray,
        current_time_index: int,
        obj_types: np.ndarray,
        scene_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Select valid center agents for prediction based on validity and type. Used in `process()`.

        Args:
            track_index_to_predict (np.ndarray): Indices of candidate agents.
            obj_trajs_full (np.ndarray): Full trajectories for all objects.
            current_time_index (int): Current time index.
            obj_types (np.ndarray): Object types.
            scene_id (str): Scenario ID.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (center_objects, selected_indices) or (None, None)
        """
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config.object_type
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                CONSOLE.warn(
                    f"obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}"
                )
                continue
            if obj_types[obj_idx] not in selected_type:
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            CONSOLE.warn(
                f"No center objects at time step {current_time_index}, scene_id={scene_id}"
            )
            return None, []
        center_objects = np.stack(
            center_objects_list, axis=0
        )  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(
        self,
        obj_trajs: np.ndarray,
        center_xyz: np.ndarray,
        center_heading: np.ndarray,
        heading_index: int,
        rot_vel_index: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Transforms trajectories to be relative to center agents' coordinates. Used in `get_agent_data()` within `process()`.

        Args:
            obj_trajs (np.ndarray['num_objects, num_timestamps, num_attrs']): Trajectories to transform.
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (np.ndarray['num_center_objects,  3 or 2']): Center agent positions [x, y, z] or [x, y].
            center_heading (np.ndarray['num_center_objects']): Center agent headings.
            heading_index (int): Index of heading in the num_attrs dimension of obj_trajs.
            rot_vel_index (Optional[List[int]]): Indices of velocity [vx, vy] to rotate.

        Returns:
            np.ndarray: Transformed trajectories relative to each center agent
                        Shape: (num_center_objects, num_objects, num_timestamps, num_attrs).
        """
        num_objects, num_timestamps, _num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0 : center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading,
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(
                    num_center_objects, -1, 2
                ),
                angle=-center_heading,
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs
