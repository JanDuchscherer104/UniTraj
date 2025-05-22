from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from pydantic import Field

from ..configs.path_config import PathConfig
from ..datasets.common_utils import (
    find_true_segments,
    generate_mask,
    get_polyline_dir,
    interpolate_polyline,
)
from ..utils.base_config import BaseConfig
from ..utils.console import Console
from . import common_utils
from .base_dataparser import BaseDataParser
from .types import (
    DatasetItem,
    DynamicMapInfosDict,
    InternalFormatDict,
    MapInfosDict,
    MetaDriveType,
    ObjectType,
    PolylineType,
    ProcessedDataDict,
    RawScenarioDict,
    Stage,
    TrackInfosDict,
    TracksToPredictDict,
)


class DataParserConfig(BaseConfig["DataParser"]):
    """
    Configuration object for `DataParser`, defining all preprocessing and cache-splitting behavior
    for raw ScenarioNet-format datasets. Controls how trajectory and map data are parsed, sampled,
    and filtered into model-ready format.
    """

    stage: Stage = Field(Stage.TRAIN)
    """Dataset split context ("train", "val", or "test")"""

    # Number of parallel workers (-1 = all CPUs)
    num_workers: int = Field(-1)
    """Number of parallel processes for multiprocessing. Use -1 to utilize all available cores."""

    # When True, ignore any existing cache and rebuild
    rebuild_dataset: bool = Field(False)
    """Whether to force rebuilding of preprocessed cache, even if it already exists."""

    # If True, only process `debug_samples` per split, single‐threaded, will rebuild in temp dir
    is_debug: bool = Field(False)
    """Enables a debug mode with single-threaded processing, limited data, and temporary cache use."""

    # Data selection
    num_debug_samples: Optional[int] = Field(None)
    """Maximum number of scenarios to parse in debug mode per split."""
    take_debug_samples_front: bool = Field(False)
    """If True, take the first num_debug_samples samples from the dataset, else sample randomly."""

    starting_frame: List[int] = Field(
        [0],
    )
    """For each training dataset, indicates the starting index of the trajectory history (frame offset)."""

    max_data_num: List[Optional[int]] = Field(
        [None],
    )
    """Optional cap for number of samples loaded per dataset. `None` uses the full set."""

    # Trajectory configuration
    past_len: int = Field(21)
    """Number of past timesteps to consider (in frames). Defines the length of agent history used for prediction."""

    future_len: int = Field(60)
    """Number of future timesteps to predict (in frames). Ground truth will cover this interval."""

    trajectory_sample_interval: int = Field(
        1, description="Sample interval for the trajectory"
    )
    """Subsampling rate along the temporal axis of trajectories. 1 = no subsampling."""

    # Object and map configuration
    masked_attributes: List[str] = Field(
        ["z_axis", "size"], description="Attributes to be masked in the input"
    )
    """List of agent attributes to be zeroed out. Options include "z_axis", "size", "heading", etc."""

    allowed_line_types: List[str] = Field(
        default_factory=lambda: [
            "lane",
            "stop_sign",
            "road_edge",
            "road_line",
            "crosswalk",
            "speed_bump",
        ],
    )
    """Polyline categories to include from the map (e.g., "lane", "stop_sign")."""

    # Processing configuration
    only_train_on_ego: bool = Field(False, description="Only train on AV")
    """If True, only predict the AV's trajectory"""

    # Which object types to include in parsing/filtering
    agent_types_to_predict: List[ObjectType] = Field(
        default_factory=lambda: [
            ObjectType.VEHICLE,
            ObjectType.PEDESTRIAN,
            ObjectType.CYCLIST,
        ],
        description="Only include these ObjectType enums in filtering tracks",
    )
    """Defines which object categories (VEHICLE, PEDESTRIAN, etc.) are retained for parsing."""

    predict_specified_agents: bool = Field(
        False,
    )
    """If True, only predict agents as specified in the dataset. If False, predict agents based on trajectory_filter."""

    # Map processing configuration
    center_offset_of_map: List[float] = Field([30.0, 0.0])
    """XY offset from the AV used to center map crops. Applied before range filtering."""

    crop_agents: bool = Field(False)
    """If True, prunes agents outside the map crop region. Currently unimplemented."""

    max_num_agents: int = Field(64)
    """Maximum number of agents retained per sample after filtering and distance-based selection."""

    max_num_roads: int = Field(256)
    """Maximum number of road polylines retained per sample."""

    manually_split_lane: bool = Field(False)
    """Whether to manually divide long polylines into segments using distance thresholds."""

    map_range: float = Field(120.0)
    """Radius in meters used for filtering map elements (square crop centered on AV)."""

    max_points_per_lane: int = Field(20)
    """Maximum number of discrete points per lane segment (used for fixed-size input)."""

    point_sampled_interval: int = Field(1)
    """Sampling interval / step size when sampling points along raw polylines."""

    vector_break_dist_thresh: float = Field(
        1.0,
    )
    """Threshold for splitting polylines into new segments based on discontinuity in distance."""

    num_points_each_polyline: int = Field(20)
    """Target number of points in each polyline segment after resampling."""

    paths: PathConfig = Field(default_factory=PathConfig, exclude=True)
    """
    Stores all filesystem paths required for loading/writing datasets and cache.
    """

    target: Type["DataParser"] = Field(default_factory=lambda: DataParser, exclude=True)

    def get_num_workers(self) -> int:
        total = cpu_count()
        return total if self.num_workers < 0 else min(max(1, self.num_workers), total)

    def setup_target(self, **kwargs) -> "DataParser":
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "setup_target", self.stage.name
        )

        if self.is_debug:
            CONSOLE.log(
                "Debug mode → single worker, limited to "
                f"{self.num_debug_samples} samples"
            )
            self.num_workers = 1
        else:
            # Use all available CPU cores if load_num_workers is < 0
            self.num_workers = self.get_num_workers()
            CONSOLE.log(
                f"Using {self.num_workers}/{cpu_count()} workers\n"
                f"Rebuild dataset: {self.rebuild_dataset}\n"
            )

        # prefix removed automatically on new CONSOLE instances

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
        traffic_lights = scenario["dynamic_map_states"]
        tracks = scenario["tracks"]
        map_feat = scenario["map_features"]

        past_length = self.config.past_len
        future_length = self.config.future_len
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config.trajectory_sample_interval
        frequency_mask = generate_mask(
            past_length - 1, total_steps, trajectory_sample_interval
        )

        track_infos = {
            "object_id": [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            "object_type": [],
            "trajs": [],
        }  # type: TrackInfosDict

        for k, v in tracks.items():

            state = v["state"]
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [
                state["position"],
                state["length"],
                state["width"],
                state["height"],
                state["heading"],
                state["velocity"],
                state["valid"],
            ]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            assert isinstance(all_state, np.ndarray)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                all_state = np.pad(
                    all_state, ((ending_fame - all_state.shape[0], 0), (0, 0))
                )
            all_state = all_state[starting_fame:ending_fame]

            assert (
                all_state.shape[0] == total_steps
            ), f"Error: {all_state.shape[0]} != {total_steps}"

            track_infos["object_id"].append(k)
            track_infos["object_type"].append(ObjectType.from_metadrive_type(v["type"]))
            track_infos["trajs"].append(all_state)

        track_infos["trajs"] = np.stack(track_infos["trajs"], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos["trajs"][..., -1] *= frequency_mask[np.newaxis]
        scenario["metadata"]["ts"] = scenario["metadata"]["ts"][:total_steps]

        # x,y,z,type
        map_infos = {
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "stop_sign": [],
            "crosswalk": [],
            "speed_bump": [],
        }  # type: MapInfosDict
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = PolylineType.from_metadrive_type(v["type"])
            if polyline_type_ == PolylineType.UNSET:
                continue

            cur_info = {"id": k}
            cur_info["type"] = v["type"]
            if polyline_type_ in [
                PolylineType.LANE_FREEWAY,
                PolylineType.LANE_SURFACE_STREET,
                PolylineType.LANE_BIKE_LANE,
            ]:
                cur_info["speed_limit_mph"] = v.get("speed_limit_mph", None)
                cur_info["interpolating"] = v.get("interpolating", None)
                cur_info["entry_lanes"] = v.get("entry_lanes", None)
                try:
                    cur_info["left_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["left_neighbor"]
                    ]
                    cur_info["right_boundary"] = [
                        {
                            "start_index": x["self_start_index"],
                            "end_index": x["self_end_index"],
                            "feature_id": x["feature_id"],
                            "boundary_type": "UNKNOWN",  # roadline type
                        }
                        for x in v["right_neighbor"]
                    ]
                except:
                    cur_info["left_boundary"] = []
                    cur_info["right_boundary"] = []
                polyline = v["polyline"]
                polyline = interpolate_polyline(polyline)
                map_infos["lane"].append(cur_info)
            elif polyline_type_ in [
                PolylineType.LINE_BROKEN_SINGLE_WHITE,  # 6
                PolylineType.LINE_SOLID_SINGLE_WHITE,  # 7
                PolylineType.LINE_SOLID_DOUBLE_WHITE,  # 8
                PolylineType.LINE_BROKEN_SINGLE_YELLOW,  # 9
                PolylineType.LINE_BROKEN_DOUBLE_YELLOW,  # 10
                PolylineType.LINE_SOLID_SINGLE_YELLOW,  # 11
                PolylineType.LINE_SOLID_DOUBLE_YELLOW,  # 12
                PolylineType.LINE_PASSING_DOUBLE_YELLOW,  # 13
            ]:
                try:
                    polyline = v["polyline"]
                except:
                    polyline = v["polygon"]
                polyline = interpolate_polyline(polyline)
                map_infos["road_line"].append(cur_info)
            elif polyline_type_ in [
                PolylineType.BOUNDARY_LINE,
                PolylineType.BOUNDARY_MEDIAN,
            ]:
                polyline = v["polyline"]
                polyline = interpolate_polyline(polyline)
                cur_info["type"] = 7
                map_infos["road_line"].append(cur_info)
            elif polyline_type_ == PolylineType.STOP_SIGN:
                cur_info["lane_ids"] = v["lane"]
                cur_info["position"] = v["position"]
                map_infos["stop_sign"].append(cur_info)
                polyline = v["position"][np.newaxis]
            elif polyline_type_ == PolylineType.CROSSWALK:
                map_infos["crosswalk"].append(cur_info)
                polyline = v["polygon"]
            elif polyline_type_ == PolylineType.SPEED_BUMP:
                map_infos["crosswalk"].append(cur_info)
                polyline = v["polygon"]
            if polyline.shape[-1] == 2:
                polyline = np.concatenate(
                    (polyline, np.zeros((polyline.shape[0], 1))), axis=-1
                )
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_.value
                cur_polyline = np.concatenate(
                    (polyline, cur_polyline_dir, type_array), axis=-1
                )
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos["all_polylines"] = polylines

        dynamic_map_infos = {
            "lane_id": [],
            "state": [],
            "stop_point": [],
        }  # type: DynamicMapInfosDict
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v["state"]["object_state"]:  # (num_observed_signals)
                lane_id.append(str(v["lane"]))
                state.append(cur_signal)
                if type(v["stop_point"]) == list:
                    stop_point.append(v["stop_point"])
                else:
                    stop_point.append(v["stop_point"].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos["lane_id"].append(np.array([lane_id]))
            dynamic_map_infos["state"].append(np.array([state]))
            dynamic_map_infos["stop_point"].append(np.array([stop_point]))

        ret = {
            "track_infos": track_infos,
            "dynamic_map_infos": dynamic_map_infos,
            "map_infos": map_infos,
        }
        ret.update(scenario["metadata"])
        ret["timestamps_seconds"] = ret.pop("ts")
        ret["current_time_index"] = self.config.past_len - 1
        ret["sdc_track_index"] = track_infos["object_id"].index(ret["sdc_id"])

        # Determine prediction agents based on config priorities
        tracks_to_predict: TracksToPredictDict = {}
        # 1. Ego only
        if self.config.only_train_on_ego:
            tracks_to_predict = {
                "track_index": [ret["sdc_track_index"]],
                "difficulty": [0],
                "object_type": [MetaDriveType.VEHICLE],
            }
        # 2. Predict filtered agents
        elif (
            ret.get("tracks_to_predict") is not None
            and self.config.predict_specified_agents
        ):
            sample_list = list(
                ret["tracks_to_predict"].keys()
            )  # + ret.get('objects_of_interest', [])
            sample_list = list(set(sample_list))
            tracks_to_predict = {
                "track_index": [
                    track_infos["object_id"].index(id)
                    for id in sample_list
                    if id in track_infos["object_id"]
                ],
                "object_type": [
                    track_infos["object_type"][track_infos["object_id"].index(id)]
                    for id in sample_list
                    if id in track_infos["object_id"]
                ],
            }
        else:
            filtered_tracks = self.trajectory_filter(ret)
            sample_list = list(
                set(filtered_tracks.keys()).union(
                    (ret.get("tracks_to_predict") or dict()).keys()
                )
            )
            tracks_to_predict = {
                "track_index": [
                    track_infos["object_id"].index(id)
                    for id in sample_list
                    if id in track_infos["object_id"]
                ],
                "object_type": [
                    track_infos["object_type"][track_infos["object_id"].index(id)]
                    for id in sample_list
                    if id in track_infos["object_id"]
                ],
            }
        # 3. Fallback
        ret["tracks_to_predict"] = tracks_to_predict

        ret["map_center"] = scenario["metadata"].get("map_center", np.zeros(3))[
            np.newaxis
        ]

        ret["track_length"] = total_steps
        return ret

    def process(
        self, internal_format: InternalFormatDict
    ) -> Optional[List[DatasetItem]]:
        """
        Process internal format data to generate model inputs for each interested agent.

        Args:
            internal_format (InternalFormatDict): Data in internal format.

        Returns:
            Optional[List[DatasetItem]]: List of processed data items, one per agent of interest.
        """
        # Debug: overview of incoming data
        if self.config.is_debug:
            CONSOLE = Console.with_prefix(
                self.__class__.__name__, "process", internal_format["scenario_id"]
            )
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
            "center_objects_type": np.array(
                list(map(lambda x: x.value, track_infos["object_type"])), dtype=np.int64
            )[track_index_to_predict],
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

        # Get map polylines: support functions that return 2 or 3 elements
        if self.config.manually_split_lane:
            result = self.get_manually_split_map_data(
                center_objects=center_objects, map_infos=info["map_infos"]
            )
        else:
            result = self.get_map_data(
                center_objects=center_objects, map_infos=info["map_infos"]
            )
        # Unpack result, allowing 2- or 3-tuple returns
        map_polylines_data, map_polylines_mask, map_polylines_center = result

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

        # change everything to float32
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

    def postprocess(self, output: List[DatasetItem]) -> List[DatasetItem]:
        """
        Perform post-processing steps like calculating difficulty and trajectory type.

        Args:
            output (List[DatasetItem]): List of processed data items from the process() step.

        Returns:
            Optional[List[DatasetItem]]: List of data items with added post-processing info, or None if input is invalid.
        """
        common_utils.get_kalman_difficulty(output)
        common_utils.get_trajectory_type(output)

        return output

    def trajectory_filter(self, data: InternalFormatDict) -> Dict[str, Any]:
        """
        Apply a sequence of validity and motion-based criteria to identify
        which object tracks are suitable prediction targets.

        The filtering logic enforces:
          1. Type constraint: only VEHICLE, PEDESTRIAN, or CYCLIST.
          2. Validity ratio: at least 50% of the track frames must be valid.
          3. Movement threshold (vehicles only): total traveled distance ≥ 2 meters.
          4. Present at prediction time: the object must be visible at current index.
          5. Future continuity: ensures there is at least one valid future step
             (identifies the first invalid future frame without discarding the track).

        Args:
            data (InternalFormatDict): Preprocessed scenario data containing
                'track_infos.trajs' array of shape [N, S, D] and 'current_time_index'.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from object_id to a dict with keys:
              - 'track_index' (int): index of the track in the trajs array.
              - 'track_id' (str): original object identifier.
              - 'difficulty' (int): placeholder difficulty score (0 = unknown).
              - 'object_type' (str): raw type string of the object.
        """
        trajs = data["track_infos"]["trajs"]
        current_idx = data["current_time_index"]
        obj_summary = data["object_summary"]

        tracks_to_predict = {}
        for idx, (k, v) in enumerate(obj_summary.items()):
            type = v["type"]

            positions = trajs[idx, :, 0:2]
            validity = trajs[idx, :, -1]
            if type not in ["VEHICLE", "PEDESTRIAN", "CYCLIST"]:
                continue
            valid_ratio = v["valid_length"] / v["track_length"]
            if valid_ratio < 0.5:
                continue
            moving_distance = v["moving_distance"]
            if moving_distance < 2.0 and type == "VEHICLE":
                continue
            is_valid_at_m = validity[current_idx] > 0
            if not is_valid_at_m:
                continue

            # past_traj = positions[:current_idx+1, :]  # Time X (x,y)
            # gt_future = positions[current_idx+1:, :]
            # valid_past = count_valid_steps_past(validity[:current_idx+1])

            future_mask = validity[current_idx + 1 :]
            future_mask[-1] = 0
            idx_of_first_zero = np.where(future_mask == 0)[0]
            idx_of_first_zero = (
                len(future_mask)
                if len(idx_of_first_zero) == 0
                else idx_of_first_zero[0]
            )

            # past_trajectory_valid = past_traj[-valid_past:, :]  # Time(valid) X (x,y)

            # try:
            #     kalman_traj = estimate_kalman_filter(past_trajectory_valid, idx_of_first_zero)  # (x,y)
            #     kalman_diff = calculate_epe(kalman_traj, gt_future[idx_of_first_zero-1])
            # except:
            #     continue
            # if kalman_diff < 20: continue

            tracks_to_predict[k] = {
                "track_index": idx,
                "track_id": k,
                "difficulty": 0,
                "object_type": type,
            }

        return tracks_to_predict

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

        # ===== FIX: Correctly map each center agent to its position in the topk list =====
        track_index_to_predict_new = np.zeros(
            len(track_index_to_predict), dtype=np.int64
        )
        for i, center_idx in enumerate(np.arange(num_center_objects)):
            # Map from original index to index in the valid_past_mask filtered array
            orig_idx_in_filtered = (
                np.where(valid_past_mask)[0].tolist().index(track_index_to_predict[i])
            )
            # Find where this filtered index appears in the topk array for this center object
            where_in_topk = np.where(topk_idxs[i] == orig_idx_in_filtered)[0]
            if len(where_in_topk) > 0:
                track_index_to_predict_new[i] = where_in_topk[0]
            # If not found in topk (shouldn't happen since center agent should always be close to itself),
            # keep it as 0 but log a warning
            else:
                CONSOLE = Console.with_prefix(self.__class__.__name__, "get_agent_data")
                CONSOLE.warn(
                    f"Center agent {i} (original idx: {track_index_to_predict[i]}) "
                    f"not found in its own topk agents list! This may cause incorrect predictions."
                )
        # ===== End of FIX =====

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
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "get_manually_split_map_data"
        )

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
        CONSOLE = Console.with_prefix(self.__class__.__name__, "get_map_data")

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

        map_range = self.config.map_range
        center_offset = self.config.center_offset_of_map
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == "all_polylines" or k not in self.config.allowed_line_types:
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
            CONSOLE.warn("No valid polylines found. Returning empty data.")
            empty_polys = np.zeros(
                (num_agents, 0, max_points_per_lane, 7), dtype=np.float32
            )
            empty_mask = np.zeros((num_agents, 0, max_points_per_lane), dtype=np.int32)
            empty_center = np.zeros((num_agents, 0, 3), dtype=np.float32)
            return empty_polys, empty_mask, empty_center

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
        CONSOLE = Console.with_prefix(self.__class__.__name__, "get_interested_agents")

        center_objects_list = []
        track_index_to_predict_selected = []
        # filter by configured object_types
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                CONSOLE.warn(
                    f"obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}"
                )
                continue
            if obj_types[obj_idx] not in self.config.agent_types_to_predict:
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            CONSOLE.warn(
                f"No center objects at time step {current_time_index}, scene_id={scene_id}"
            )
            return None, None
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
