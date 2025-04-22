import pickle
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path
from re import S
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import h5py
import hydra
import numpy as np
import torch
from metadrive.scenario import utils as sd_utils
from metadrive.scenario.scenario_description import MetaDriveType
from omegaconf import OmegaConf
from PIL import Image
from pydantic import Field
from scenarionet.common_utils import read_scenario
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ..configs.path_config import PathConfig
from ..datasets import common_utils
from ..datasets.common_utils import (
    find_true_segments,
    generate_mask,
    get_kalman_difficulty,
    get_polyline_dir,
    get_trajectory_type,
    interpolate_polyline,
    is_ddp,
)
from ..datasets.types import (
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
from ..utils.base_config import CONSOLE, BaseConfig
from ..utils.visualization import check_loaded_data

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)

# TODO: use polars internally??
# TODO: use pydantic objects instead of typed dicts
# TODO: create DataModel(BaseModel) for the DataSet Item, they should have a useful __repr__ function
# TODO: improve the splitting into train:val:test (using the original split as in argoverse2, and scenarioNet)
# TODO: add plotting functions
# TODO: split into Dataset and DataParser
# TODO: improve naming of the variables
# TODO: use unified logging style and system (LOGGER.set_prefix(self.__class__.__name__))


class DatasetConfig(BaseConfig["BaseDataset"]):
    """Configuration for the BaseDataset."""

    target: Type["BaseDataset"] = Field(default_factory=lambda: BaseDataset)

    stage: Stage = Field(Stage.TRAIN)
    """
    STAGE.TRAIN, STAGE.VAL, STAGE.TEST
    """
    # Data loading configuration
    # TODO: move descrtiptions to doc-strings of each field
    load_num_workers: int = Field(
        -1,
        description="Number of workers for loading data, -1 to use all available CPU cores",
    )
    is_debug: bool = Field(
        False, description="Debug mode (no MP, limited samples, CPU only)"
    )
    # Data selection
    num_debug_samples: Optional[int] = 2
    max_data_num: List[Optional[int]] = Field(
        [None],
        description="Maximum number of data for each training dataset, null means all data",
    )
    starting_frame: List[int] = Field(
        [0],
        description="History trajectory starts at this frame for each training dataset",
    )

    # Trajectory configuration
    past_len: int = Field(21, description="History trajectory length, 2.1s")
    future_len: int = Field(60, description="Future trajectory length, 6s")
    trajectory_sample_interval: int = Field(
        1, description="Sample interval for the trajectory"
    )

    # Object and map configuration
    object_type: List[str] = Field(
        ["VEHICLE"], description="Object types included in the training set"
    )
    line_type: List[str] = Field(
        ["lane", "stop_sign", "road_edge", "road_line", "crosswalk", "speed_bump"],
        description="Line type to be considered in the input",
    )
    masked_attributes: List[str] = Field(
        ["z_axis", "size"], description="Attributes to be masked in the input"
    )

    # Processing configuration
    only_train_on_ego: bool = Field(False, description="Only train on AV")
    center_offset_of_map: List[float] = Field(
        [30.0, 0.0], description="Center offset of the map"
    )

    # Caching configuration
    use_cache: bool = Field(False, description="Use cache for data loading")
    overwrite_cache: bool = Field(True, description="Overwrite existing cache")
    store_data_in_memory: bool = Field(
        False, description="Store data in memory"
    )  # TODO: not used

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

    # Paths - Ensure PathConfig is used
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

    def setup_target(self, **kwargs) -> "BaseDataset":

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

        return self.target(self, **kwargs)  # type: ignore


class BaseDataset(Dataset):

    def __init__(self, config: DatasetConfig):
        self.config = config

        self.data_loaded_memory: List[Dict[str, Any]] = []
        self.file_cache: Dict[str, Optional[h5py.File]] = {}
        # Use PathConfig for path management
        self.paths: PathConfig = config.paths
        self.load_data()

    def load_data(self):
        """
        Load or rebuild the cached file_list for each dataset split.
        In debug mode → single-threaded rebuild + truncate overall to num_debug_samples.
        """
        self.data_loaded = {}
        cfg = self.config
        CONSOLE.log(
            f"{self.config.stage} data loading started "
            f"(use_cache={cfg.use_cache}, overwrite_cache={cfg.overwrite_cache}, "
            f"debug={cfg.is_debug}, workers={cfg.load_num_workers})"
        )

        # Normalize and sanity‐check data paths
        data_paths = (
            cfg.paths.data if isinstance(cfg.paths.data, list) else [cfg.paths.data]
        )
        if not data_paths:
            raise RuntimeError("No `paths.data` configured → cannot load any datasets")
        for p in data_paths:
            if not isinstance(p, Path) or not p.exists():
                raise RuntimeError(f"Data path invalid or missing: {p}")

        # Decide once whether to rebuild at all or attempt to load cache
        do_rebuild = cfg.is_debug or not cfg.use_cache

        total_loaded = 0
        for idx, dp in enumerate(data_paths):
            phase, name = dp.parent.name, dp.name
            cache_dir = self.paths.get_cache_path(name, phase)
            cache_file = cache_dir / f"{name}_{phase.lower()}.pkl"
            if not cache_file.exists():
                CONSOLE.log(
                    f"({name}::{self.config.stage}) cache file not found, will rebuild: {cache_file}"
                )
                do_rebuild = True

            data_usage = cfg.max_data_num[idx] if idx < len(cfg.max_data_num) else None
            self.starting_frame = (
                cfg.starting_frame[idx] if idx < len(cfg.starting_frame) else 0
            )

            CONSOLE.log(f"({name}::{self.config}) Path to ScenarioNet dir = {dp}")

            CONSOLE.log(
                f"({name}::{self.config}) {'will attempt to load' if not do_rebuild else 'will rebuild'} cache from {cache_dir}"
            )

            file_list: Dict[str, Any] = {}
            if not do_rebuild and not cfg.overwrite_cache or is_ddp():
                # 1) Try loading from cache
                CONSOLE.log(f"({name}::{self.config}) → loading from cache")
                try:
                    file_list = self.get_data_list(name, phase, data_usage)
                except Exception as e:
                    CONSOLE.error(f"({name}::{self.config}) cache load failed: {e}")
                    CONSOLE.log(f"({name}::{self.config}) falling back to rebuild")
                else:
                    CONSOLE.log(
                        f"({name}::{self.config}) cache hit → {len(file_list)} samples"
                    )

            if not file_list:
                # 2) Rebuild cache
                CONSOLE.log(f"({name}::{self.config}) → rebuilding cache")
                # read summary
                try:
                    map_id_to_meta, scenario_ids_list, map_id_to_file = (
                        sd_utils.read_dataset_summary(dp)
                    )
                    num_scenarios_total = len(scenario_ids_list)
                    if not scenario_ids_list:
                        raise ValueError("no scenarios in summary")
                except Exception as e:
                    raise RuntimeError(
                        f"({name}::{self.config}) summary read error: {e}"
                    )
                CONSOLE.log(
                    f"({name}::{self.config}) found {len(scenario_ids_list)} scenarios"
                )

                # select only num_debug_samples randomly selcted samples if in debug mode
                if cfg.is_debug:
                    max_dbg = cfg.num_debug_samples or num_scenarios_total
                    n_dbg = min(max_dbg, num_scenarios_total)
                    selected_ids = list(
                        np.random.choice(scenario_ids_list, size=n_dbg, replace=False)
                    )
                    scenario_ids_list = selected_ids
                    map_id_to_meta = {
                        mid: meta
                        for mid, meta in map_id_to_meta.items()
                        if mid in selected_ids
                    }
                    map_id_to_file = {
                        mid: fpath
                        for mid, fpath in map_id_to_file.items()
                        if mid in selected_ids
                    }
                    CONSOLE.log(
                        f"({name}::{self.config}) debug mode: sampling {n_dbg} / {num_scenarios_total} scenarios"
                    )

                # prepare cache dir
                self.paths.remove_dir(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                # split and process
                splits = np.array_split(scenario_ids_list, cfg.load_num_workers)

                # 2) build list of direct args (passing worker_index in place of filename stem)
                args: List[Tuple[str, Dict, List[str], str, str, int]] = [
                    (str(dp), map_id_to_file, list(chunk), name, phase, i)
                    for i, chunk in enumerate(splits)
                ]

                # 3) dispatch
                if cfg.is_debug:
                    results = [
                        self.process_data_chunk(dp, mapping, chunk, name, phase, wid)
                        for dp, mapping, chunk, name, phase, wid in args
                    ]
                else:
                    CONSOLE.log(
                        f"({name}::{self.config}) spawning {cfg.load_num_workers} worker(s)"
                    )
                    try:
                        with Pool(cfg.load_num_workers) as pool:
                            results = pool.starmap(self.process_data_chunk, args)
                    except Exception as e:
                        raise RuntimeError(
                            f"({name}::{self.config}) multiprocessing failed: {e}"
                        )

                # collect
                for r in results:
                    if r:
                        file_list.update(r)
                if not file_list:
                    raise RuntimeError(
                        f"({name}::{self.config}) rebuild produced 0 samples"
                    )

                # write cache
                with cache_file.open("wb") as f:
                    pickle.dump(file_list, f)
                CONSOLE.log(
                    f"({name}::{self.config}) rebuild complete → {len(file_list)} samples"
                )

                # apply per-split data_usage
                if not self.self.config and data_usage:
                    items = list(file_list.items())
                    np.random.shuffle(items)
                    file_list = dict(items[:data_usage])
                    CONSOLE.log(
                        f"({name}::{self.config}) applied data_usage={data_usage} → {len(file_list)} samples"
                    )

            # accumulate
            self.data_loaded.update(file_list)
            total_loaded += len(file_list)
            CONSOLE.log(
                f"({name}::{self.config}) accumulated → {total_loaded} total samples so far"
            )

        # TODO: improve naming of var "all_keys" and "self.data_loaded_keys"

        # wrap up
        all_keys = list(self.data_loaded)
        CONSOLE.log(
            f"\n{self.config} data loading finished → total {total_loaded} samples"
        )

        self.data_loaded_keys = all_keys

        if total_loaded == 0:
            raise RuntimeError(
                f"{self.config} loaded 0 samples → please verify `paths.data` and cache settings"
            )

    def process_data_chunk(
        self,
        data_path_str: str,
        mapping: Dict[Any, Any],
        data_list: List[str],
        dataset_name: str,
        phase: str,
        worker_index: int,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Process a chunk of scenarios in parallel, writing them to an HDF5 shard
        and returning a map of group_name → file_info (including kalman_difficulty).

        Args:
            data_path_str (str): Path to the scenario data.
            mapping (Dict[Any, Any]): Mapping of scenario IDs to file names.
            data_list (List[str]): List of scenario file names to process.
            dataset_name (str): Name of the dataset.
            phase (str): Phase of the dataset (e.g., "train", "val").
            worker_index (int): Index of the worker processing this chunk.

        Returns:
            A dict mapping each HDF5 group name to its file info, or None on error.
        """
        progress_bar = getattr(self, "_shared_progress_bars", {}).get(worker_index)

        # Prepare output container
        file_list: Dict[str, Dict[str, Any]] = {}

        # Determine HDF5 shard path via PathConfig
        hdf5_path: Path = self.paths.get_cache_chunk_path(
            dataset_name, phase, worker_index
        )
        h5f = h5py.File(hdf5_path, "w")

        # --- Create / open HDF5 file and write each scenario’s outputs ---
        try:
            n = len(data_list)
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt and cnt % max(n // 10, 1) == 0:
                    CONSOLE.log(f"[{dataset_name}/{phase}] Worker0 processed {cnt}/{n}")

                # Update progress bar if available
                if progress_bar is not None:
                    progress_bar.set_description(f"Worker {worker_index}")
                    progress_bar.update(1)
                # Log progress every 10%
                if worker_index == 0 and cnt and cnt % max(n // 10, 1) == 0:
                    CONSOLE.log(
                        f"[{dataset_name}/{phase}] Worker 0: processed {cnt}/{n}"
                    )

                # Read the raw scenario
                try:
                    scenario = read_scenario(data_path_str, mapping, file_name)
                    scenario.setdefault("metadata", {})[
                        "dataset"
                    ] = f"{dataset_name}/{phase}"
                except Exception as e:
                    CONSOLE.warn(
                        f"[read_scenario] {dataset_name}/{phase}/{file_name}: {e}"
                    )
                    continue

                # Pipeline: preprocess → process → postprocess
                try:
                    out = self.preprocess(scenario)
                    assert out is not None, "preprocess returned None"
                    out = self.process(out)
                    assert out is not None, "process returned None"
                    out = self.postprocess(out)
                    assert out is not None, "postprocess returned None"

                except Exception as e:
                    CONSOLE.warn(f"[pipeline] {dataset_name}/{phase}/{file_name}: {e}")
                    continue

                # Write each record in this scenario to HDF5
                for i, record in enumerate(out):
                    grp_name = f"{dataset_name}-{worker_index}-{cnt}-{i}"
                    grp = h5f.create_group(grp_name)

                    for key, value in record.items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)

                    file_info = {
                        "kalman_difficulty": np.stack(
                            [x["kalman_difficulty"] for x in out]
                        ),
                        "h5_path": hdf5_path,
                    }
                    file_list[grp_name] = file_info

                del scenario, out

        except Exception as e:
            CONSOLE.error(
                f"[worker {worker_index}] HDF5 write error at {hdf5_path}: {e}"
            )
            return None
        finally:
            h5f.close()
            if progress_bar is not None:
                progress_bar.close()

        return file_list

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
                f"get_interested_agents() returned type(center_objects)={center_objects} (expected np.ndarray) for scenario {scene_id} in process()"
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

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

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
        # Add the trajectory difficulty
        get_kalman_difficulty(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type(output)

        return output

    def collate_fn(self, data_list: List[Optional[DatasetItem]]) -> Optional[BatchDict]:
        """
        Collates a list of DatasetItems into a BatchDict for model input.

        Args:
            batch_list (List[Optional[DatasetItem]]): List of data items, potentially containing None values.

        Returns:
            Optional[BatchDict]: A dictionary containing batched tensors, or None if the batch is empty.
        """
        batch = list(filter(lambda x: x is not None, data_list))
        if not batch:
            return None

        # 2) stack arrays → Tensors, leave other objects alone
        input_dict = default_collate(batch)

        cot = input_dict.get("center_objects_type")
        if isinstance(cot, torch.Tensor):
            input_dict["center_objects_type"] = cot.numpy()
        # batch_list = []
        # for batch in data_list:
        #     batch_list.append(batch)

        # batch_size = len(batch_list)
        # key_to_list = {}
        # for key in batch_list[0].keys():
        #     key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        # input_dict = {}
        # for key, val_list in key_to_list.items():
        #     # if val_list is str:
        #     try:
        #         input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
        #     except:
        #         input_dict[key] = val_list

        # input_dict["center_objects_type"] = input_dict["center_objects_type"].numpy()
        batch_dict: BatchDict = {
            "batch_size": len(batch),
            "input_dict": input_dict,
            "batch_sample_count": len(batch),
        }
        return batch_dict

    def __len__(self):
        return len(self.data_loaded_keys)

    @lru_cache(maxsize=None)
    def _get_file(self, file_path_str: str) -> Optional[h5py.File]:
        """Safely opens and caches HDF5 file handles."""
        file_path = Path(file_path_str)
        if not self.paths.check_exists(file_path):
            CONSOLE.error(f"HDF5 file does not exist: {file_path}")
            return None
        try:
            h5_file = h5py.File(file_path, "r")  # , swmr=True)
            return h5_file
        except Exception as e:
            CONSOLE.error(f"Failed to open HDF5 file {file_path}: {e}")
            return None  # Return None on failure

    def __getitem__(self, idx: int) -> DatasetItem:
        """
        Get a DatasetItem by index.

        Args:
            idx (int): Index of the data item.

        Returns:
            DatasetItem: Data item at the given index.
        """
        file_key = self.data_loaded_keys[idx]
        file_info = self.data_loaded[file_key]
        file_path = file_info["h5_path"]

        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)

        group = self.file_cache[file_path][file_key]
        record = {
            k: (
                group[k][()].decode("utf-8")
                if isinstance(group[k].dtype.type, np.bytes_)
                else group[k][()]
            )
            for k in group.keys()
        }

        return record

    def close_files(self):
        for f in self.file_cache.values():
            f.close()
        self.file_cache.clear()

    # Add dataset_name and phase arguments
    def get_data_list(
        self, dataset_name: str, phase: str, data_usage: Optional[int]
    ) -> Dict[str, Any]:
        """
        Loads the file list for a given dataset phase from the cache.

        Args:
            dataset_name (str): Name of the dataset.
            phase (str): Data phase (e.g., 'train', 'val', 'test').
            data_usage (Optional[int]): Maximum number of samples to use (None for all).

        Returns:
            Dict[str, Any]: Dictionary mapping group names to file info.

        Raises:
            ValueError: If the cache file is not found or cannot be loaded.
        """
        # Use PathConfig method to get cache path
        cache_dir = self.paths.get_cache_path(dataset_name, phase)
        file_list_path = cache_dir / "file_list.pkl"

        # Use PathConfig method to check existence
        if self.paths.check_exists(file_list_path):
            try:
                # Use path.open()
                with file_list_path.open("rb") as f:
                    data_loaded = pickle.load(f)
            except Exception as e:
                raise ValueError(
                    f"Error loading file_list.pkl from {file_list_path}: {e}"
                )
        else:
            raise ValueError(f"Error: file_list.pkl not found at {file_list_path}")

        data_list = list(data_loaded.items())
        np.random.shuffle(data_list)
        CONSOLE.log(f"[get_data_list] total={len(data_list)}, data_usage={data_usage}")

        # only enforce a limit when data_usage is set
        if not self.config.stage and data_usage is not None:
            data_loaded = dict(data_list[:data_usage])
        else:
            data_loaded = dict(data_list)
        return data_loaded

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
        Prepare agent-centric trajectory and mask data for model input.

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

    def get_interested_agents(
        self,
        track_index_to_predict: np.ndarray,
        obj_trajs_full: np.ndarray,
        current_time_index: int,
        obj_types: np.ndarray,
        scene_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Select valid center agents for prediction based on validity and type.
        (This function might be redundant if preprocess handles filtering correctly)

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
        Transforms trajectories to be relative to center agents' coordinates.

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

    def get_map_data(
        self, center_objects: np.ndarray, map_infos: MapInfosDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare map polyline features relative to each center agent, using fixed-size points per polyline.

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

    def get_manually_split_map_data(
        self, center_objects: np.ndarray, map_infos: MapInfosDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare map polyline features relative to each center agent, manually splitting long lanes.

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

    # --- Unused/Debugging Methods ---
    # These seem like debugging or analysis methods, keep them but ensure they use CONSOLE

    def sample_from_distribution(self, original_array, m=100):
        # ... (keep original logic, replace print with CONSOLE.log/plog if needed for debugging) ...
        # Example replacement:
        # print("total sample:", len(sampled_indices)) -> CONSOLE.log(f"Total samples from distribution: {len(sampled_indices)}")
        # print(f"Bin {range_}: ...") -> CONSOLE.log(f"Bin {range_}: ...")
        distribution = [
            ("-10,0", 0),
            ("0,10", 23.952629169758517),
            ("10,20", 24.611144221251667),
            ("20,30.0", 21.142773679220554),
            ("30,40.0", 15.996653629820514),
            ("40,50.0", 9.446714336574939),
            ("50,60.0", 3.7812939732733786),
            ("60,70", 0.8821063091988663),
            ("70,80.0", 0.1533644322320915),
            ("80,90.0", 0.027777741552241064),
            ("90,100.0", 0.005542507117231198),
        ]
        bins = np.array([float(range_.split(",")[1]) for range_, _ in distribution])
        sample_sizes = np.array([round(perc / 100 * m) for _, perc in distribution])
        bin_indices = np.digitize(original_array, bins)
        sampled_indices = []
        for i, size in enumerate(sample_sizes):
            indices_in_bin = np.where(bin_indices == i)[0]
            sampled_indices_in_bin = np.random.choice(
                indices_in_bin, size=min(size, len(indices_in_bin)), replace=False
            )
            sampled_indices.extend(sampled_indices_in_bin)
        sampled_array = original_array[sampled_indices]
        CONSOLE.log(f"Total samples from distribution: {len(sampled_indices)}")
        # for i, (range_, _) in enumerate(distribution):
        #     CONSOLE.log(
        #         f"Bin {range_}: Expected {distribution[i][1]:.2f}%, Actual {len(np.where(bin_indices[sampled_indices] == i)[0]) / len(sampled_indices) * 100:.2f}%"
        #     )
        return sampled_array, sampled_indices

    def trajectory_filter(self, data: InternalFormatDict) -> Dict[str, Any]:
        """
        Filter trajectories to select valid tracks for prediction.
        (Note: This logic might be better placed or simplified within preprocess)

        Args:
            data (InternalFormatDict): Internal format scenario data.

        Returns:
            Dict[str, Any]: Dictionary of tracks to predict {object_id: info}.
        """
        tracks_to_predict_filtered = {}
        try:
            trajs = data["track_infos"]["trajs"]  # (num_obj, steps, feat)
            obj_ids = data["track_infos"]["object_id"]
            obj_types_enum = data["track_infos"]["object_type"]
            current_idx = data["current_time_index"]
            # object_summary might not exist, calculate necessary info directly
            # obj_summary = data["object_summary"]

            selected_type_enums = [object_type[x] for x in self.config.object_type]

            for i, obj_id in enumerate(obj_ids):
                obj_type_enum = obj_types_enum[i]
                if obj_type_enum not in selected_type_enums:
                    continue

                positions = trajs[i, :, 0:2]
                validity = trajs[i, :, -1]
                track_length = len(validity)
                valid_steps = validity > 0

                # Check validity at current time
                if not valid_steps[current_idx]:
                    continue

                # Check valid ratio
                valid_ratio = np.sum(valid_steps) / track_length
                if valid_ratio < 0.5:  # Configurable threshold?
                    continue

                # Check moving distance for vehicles
                if obj_type_enum == object_type["VEHICLE"]:
                    valid_positions = positions[valid_steps]
                    if len(valid_positions) > 1:
                        moving_distance = np.sum(
                            np.linalg.norm(np.diff(valid_positions, axis=0), axis=1)
                        )
                    else:
                        moving_distance = 0
                    if moving_distance < 2.0:  # Configurable threshold?
                        continue

                # If all checks pass, add to prediction list
                tracks_to_predict_filtered[obj_id] = {
                    "type_enum": obj_type_enum,
                    # Add other relevant info if needed
                }
        except KeyError as e:
            CONSOLE.error(
                f"Missing key in trajectory_filter for scenario {data.get('scenario_id', 'Unknown')}: {e}"
            )
        except Exception as e:
            CONSOLE.error(
                f"Error in trajectory_filter for scenario {data.get('scenario_id', 'Unknown')}: {e}"
            )

        return tracks_to_predict_filtered


# --- Hydra Main Functions ---
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def draw_figures(cfg):
    train_set = None  # Initialize to None
    try:
        set_seed(cfg.seed)
        OmegaConf.set_struct(cfg, False)  # Open the struct
        # Assuming cfg.method exists and contains dataset config
        # This merging strategy might need adjustment based on actual config structure
        # cfg = OmegaConf.merge(cfg, cfg.method)
        dataset_cfg = cfg.dataset  # Assuming dataset config is under 'dataset' key

        # Build dataset using the config
        train_set = build_dataset(dataset_cfg)  # build_dataset needs DatasetConfig

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=1,  # Process one by one for visualization
            shuffle=True,  # Shuffle to get diverse examples
            num_workers=0,  # Use 0 for debugging visualization
            collate_fn=train_set.collate_fn,
            # Add timeout if loading hangs
            # timeout=120
        )

        concat_list = [4, 4, 4, 4, 4, 4, 4, 4]  # Example layout
        images = []
        num_images_to_draw = sum(concat_list)
        CONSOLE.log(f"Attempting to draw {num_images_to_draw} figures...")

        for i, data in enumerate(tqdm(train_loader, desc="Generating figures")):
            if i >= num_images_to_draw:
                break
            if data is None or "input_dict" not in data:
                CONSOLE.warn(
                    f"Skipping batch {i} due to None data or missing 'input_dict'."
                )
                continue

            inp = data["input_dict"]
            try:
                # Assuming check_loaded_data returns a PIL image or similar
                img_bytes_io = check_loaded_data(
                    inp, 0
                )  # Draw the first sample in the batch
                if img_bytes_io:
                    img = Image.open(img_bytes_io)
                    images.append(img)
                else:
                    CONSOLE.warn(f"check_loaded_data returned None for batch {i}")

            except Exception as e:
                CONSOLE.error(f"Error generating figure for batch {i}: {e}")
                # Optionally log more details about the input data 'inp'
                # CONSOLE.plog(inp)

        if not images:
            CONSOLE.error("No images were generated.")
            return

        try:
            CONSOLE.log(f"Concatenating {len(images)} generated images...")
            final_image = concatenate_varying(images, concat_list)
            # Save or display the image
            output_path = Path("visualization_output.png")  # Save in current dir
            final_image.save(output_path)
            CONSOLE.log(f"Saved final visualization to {output_path.resolve()}")
            # final_image.show() # Opens in default image viewer

        except Exception as e:
            CONSOLE.error(f"Error concatenating or saving images: {e}")

    except Exception as e:
        CONSOLE.error(f"An error occurred in draw_figures: {e}")
    finally:
        # Ensure dataset files are closed if DataLoader doesn't handle it
        if train_set is not None and hasattr(train_set, "close_files"):
            train_set.close_files()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def split_data(cfg):
    """
    Example function to potentially copy raw data based on loaded keys.
    Requires careful implementation based on actual needs.
    """
    train_set = None  # Initialize to None
    try:
        set_seed(cfg.seed)
        OmegaConf.set_struct(cfg, False)
        # dataset_cfg = OmegaConf.merge(cfg, cfg.method).dataset # Adjust merge/access as needed
        dataset_cfg = cfg.dataset  # Assuming dataset config is under 'dataset' key

        # Initialize PathConfig from the dataset config
        paths = PathConfig(**OmegaConf.to_container(dataset_cfg.paths, resolve=True))

        # Build the dataset (this will load/process data based on config)
        CONSOLE.log("Building dataset to get loaded keys...")
        # Ensure build_dataset uses the correct config structure
        train_set = build_dataset(dataset_cfg)

        # Define output directory using PathConfig
        # Example: Copying to a subdirectory within split_data_output_dir
        split_output_dir = paths.split_data_output_dir / "my_split"
        paths.ensure_dir(split_output_dir)
        CONSOLE.log(f"Output directory for split data: {split_output_dir}")

        # This part is highly dependent on what "splitting" means.
        # If it means copying the *source* files corresponding to the loaded keys:
        CONSOLE.log(f"Processing {len(train_set.data_loaded_keys)} loaded data keys...")
        # Need mapping from data_loaded_key back to original source file path.
        # This mapping is not directly available in the current BaseDataset structure.
        # It might require modifications to how data is loaded/cached or a separate manifest file.

        # Placeholder: If we assume keys somehow map back to source files (e.g., via metadata)
        # for key in tqdm(train_set.data_loaded_keys, desc="Copying data"):
        #     try:
        #         # --- This part needs the logic to find the source file ---
        #         # Example: source_file_path = find_source_path_from_key(key, train_set.metadata_mapping)
        #         source_file_path = None # Replace with actual logic
        #
        #         if source_file_path and paths.check_exists(source_file_path):
        #             target_path = split_output_dir / source_file_path.name
        #             # Use shutil.copy or copy2 for copying
        #             import shutil
        #             shutil.copy2(source_file_path, target_path) # copy2 preserves metadata
        #         elif source_file_path:
        #             CONSOLE.warn(f"Source file not found for key {key}: {source_file_path}")
        #         else:
        #             CONSOLE.warn(f"Could not determine source file for key {key}")
        #
        #     except Exception as e:
        #         CONSOLE.error(f"Error processing key {key}: {e}")
        CONSOLE.warn(
            "Data splitting logic is currently a placeholder. Requires mapping keys to source files."
        )

    except Exception as e:
        CONSOLE.error(f"An error occurred in split_data: {e}")
    finally:
        if train_set is not None and hasattr(train_set, "close_files"):
            train_set.close_files()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Keep this minimal, primarily for triggering hydra functions
    # Example: Call draw_figures by default
    # draw_figures()
    # Or call split_data
    # split_data()
    CONSOLE.log(
        "BaseDataset module loaded. Run hydra functions like 'draw_figures' or 'split_data'."
    )
    # Example: python base_dataset.py --config-name your_config.yaml
