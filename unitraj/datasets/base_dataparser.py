from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
import pandas as pd
from metadrive.scenario import utils as sd_utils
from networkx import dfs_edges
from typing_extensions import Self

from ..configs.path_config import PathConfig
from ..utils.console import Console
from .common_utils import is_ddp
from .types import (
    DatasetItem,
    InternalFormatDict,
    ProcessedDataDict,
    RawScenarioDict,
    Stage,
)

if TYPE_CHECKING:
    from .dataparser import DataParserConfig


def process_data_chunk_wrapper(
    worker_idx: int,
    config: "DataParserConfig",
    starting_frame: int,
    data_root: Path,
    mapping: Dict[Any, str],
    ids: List[str],
) -> Dict[Stage, Dict[str, Dict[str, Any]]]:
    # Initialize parser once per task and delegate work
    parser = config.setup_target()
    parser.starting_frame = starting_frame
    return parser.process_data_chunk(worker_idx, data_root, mapping, ids)


class BaseDataParser(ABC):

    sample_metadata: Optional[pd.DataFrame] = None
    """
    Will hold pandas DataFrame of sample metadata for current split
    DataFrame columns:
    - h5_path: Path to the HDF5 file containing the sample data
    - scenario_id: Original scenario ID from ScenarioNet
    - kalman_difficulty: Array with Kalman filter difficulty scores for all agents
    - num_agents: Total number of agents in the scenario with valid trajectories
    - num_agents_interest: Number of agents of interest in the scenario
    - scenario_future_duration: Number of future timesteps in the scenario
    - num_map_polylines: Number of map polylines in the scenario
    - track_index_to_predict: Index of the track to predict
    - center_objects_type: Type ID of the centered object
    - dataset_name: Name of the dataset this sample belongs to
    - trajectory_type: Type ID of the trajectory (if available)

    The DataFrame index is the group_name in format: "{dataset_name}-{worker_idx}-{id_cnt}-{agent_idx}"
    where:
        - dataset_name: Name of the original dataset (e.g., "av2_scenarionet")
        - worker_idx: Index of the worker that processed this chunk
        - id_cnt: Counter for the scenario ID within this worker's chunk
        - agent_idx: Index of the agent within this scenario
    """

    def __init__(self, config: "DataParserConfig"):
        self.config = config
        self.paths: PathConfig = config.paths

        self.sample_metadata: Optional[pd.DataFrame] = None

    def load_data(self) -> Self:
        """
        Build a full index, split by Argo2 train/val/test, persist each,
        then keep only the configured stage.
        """
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "load_data", self.config.stage.name
        )

        if not self.config.rebuild_dataset:
            df = self.get_sample_metadata()
            if not df.empty:
                self.sample_metadata = df
                CONSOLE.log(
                    f"Metadata found for {self.config.stage}, skipping rebuild."
                )
        else:
            # Build the full merged index (all splits)
            self._build_full_index()

            # Load the data for the current stage
            self.sample_metadata = self.get_sample_metadata()
            if self.sample_metadata.empty:
                raise RuntimeError(
                    f"No data files found for {self.config.stage} in {self.paths.dataset_dest_dir}"
                )

        return self

    def _build_full_index(self) -> None:
        """
        Scan every ScenarioNet folder, parallel-map into HDF5 shards,
        and collect a merged mini-index of all records.
        """
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "_build_full_index", self.config.stage.name
        )
        for sn_idx, raw_dir in enumerate(self.paths.scenarionet_dirs):
            self.starting_frame = self.config.starting_frame[sn_idx]
            CONSOLE.log(f"Scanning {raw_dir.name}…")
            _, id_list, id_to_file = sd_utils.read_dataset_summary(raw_dir)

            # debug sampling
            if self.config.num_debug_samples is not None:
                CONSOLE.log(f"Limiting to {self.config.num_debug_samples} samples.")
                id_list = list(
                    np.random.choice(
                        id_list,
                        size=min(self.config.num_debug_samples, len(id_list)),
                        replace=False,
                    )
                )

            # configure number of workers & split into shards
            num_workers = 1 if self.config.is_debug else self.config.get_num_workers()
            np.random.shuffle(id_list)
            id_shards = np.array_split(id_list, num_workers)

            # prepare tasks: each tuple carries its shard index
            args = [
                (
                    worker_idx,
                    self.config,
                    self.starting_frame,
                    raw_dir,
                    id_to_file,
                    list(id_chunk),
                )
                for worker_idx, id_chunk in enumerate(id_shards)
            ]

            if self.config.is_debug:
                # serial execution
                for idx, _, _, data_root, mapping, ids in args:
                    sample_metadata = self.process_data_chunk(
                        idx, data_root, mapping, ids
                    )

            else:
                CONSOLE.log(
                    f"Processing a total of {len(id_list)} scenarios in {raw_dir.name} "
                    f"with {num_workers} workers ({len(id_shards)} shards)…"
                )

                # dispatch shards and collect per-shard sample_metadata dicts
                with Pool(num_workers) as p:
                    results = p.starmap(process_data_chunk_wrapper, args)
                # merge all shards' sample_metadata into per-split index

                sample_metadata: Dict[Stage, Dict[str, Dict[str, Any]]] = defaultdict(
                    dict
                )
                for part in results:
                    for split, info in part.items():
                        sample_metadata[split].update(info)

            # write pandas DataFrame per split
            for split, info in sample_metadata.items():
                df = pd.DataFrame.from_dict(info, orient="index")
                df.to_pickle(self.paths.get_sample_metadata_path(split))

    def process_data_chunk(
        self,
        worker_idx: int,
        data_root: Path,
        mapping: Dict[Any, str],
        ids: List[str],
    ) -> Dict[Stage, Dict[str, Dict[str, Any]]]:
        """
        Worker function:
          - reads each scenario via `read_scenario`
          - assigns it to its Argo2 split
          - runs preprocess/process/postprocess
          - returns a mini-index dict with all records info
        """
        CONSOLE = Console.with_prefix(
            self.__class__.__name__,
            "process_data_chunk",
            self.config.stage.name,
            f"worker_{worker_idx}",
        )

        # {Split: {extended_sid: {h5_path: <path>, kalman_difficulty: np.ndarray}}}
        sample_metadata: Dict[Stage, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        orig_ds_name = data_root.name
        hdf5_path = (
            self.paths.dataset_dest_dir / f"{orig_ds_name}_worker_{worker_idx}.h5"
        )

        with h5py.File(hdf5_path, "w") as hdf5_file:
            for id_cnt, sid in enumerate(ids):
                # --- read raw scenario ---
                try:
                    scen_pth = data_root / mapping[sid] / sid
                    scen = sd_utils.read_scenario_data(scen_pth)
                except Exception as e:
                    CONSOLE.warn(f"[read_scenario] {sid}: {e}")
                    continue

                # --- determine AV2 split ---
                if orig_ds_name == "av2_scenarionet":
                    sid_str = sid.replace("sd_av2_v2_", "").replace(".pkl", "")
                    split = self.paths.find_av2_split(sid_str) or Stage.TRAIN
                else:
                    raise NotImplementedError(
                        f"Dataset {orig_ds_name} not supported for split detection."
                    )

                # --- pipeline ---
                try:
                    pre_out = self.preprocess(scen)
                    assert pre_out is not None, f"preprocess() yielded None for {sid}"
                    out = self.process(pre_out)
                    assert out is not None, f"process() yielded None for {sid}"
                    out = self.postprocess(out)
                    assert out is not None, f"postprocess() yielded None for {sid}"
                except Exception as e:
                    CONSOLE.warn(f"[pipeline] {sid}: {e}")
                    continue

                # stack kalman difficulty once for this scenario
                kd_arr = np.stack(
                    [item.model_dump()["kalman_difficulty"] for item in out]
                )
                num_agents_interest = len(out)
                for agent_idx, agent_record in enumerate(out):
                    # Create a unique group name that encodes dataset, worker, scenario index, and agent index
                    # This identifier serves as the key in sample_metadata and allows tracing back to original data
                    grp_name = f"{orig_ds_name}-{worker_idx}-{id_cnt}-{agent_idx}"
                    grp = hdf5_file.create_group(grp_name)
                    for key, value in agent_record.model_dump().items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)
                    # accumulate metadata without resetting

                    meta = {
                        "h5_path": hdf5_path,
                        "scenario_id": sid,  # Original scenario ID from ScenarioNet
                        "kalman_difficulty": kd_arr,
                        "num_agents": np.sum(
                            np.any(agent_record.obj_trajs_last_pos != 0, axis=1)
                        ).item(),  # type: ignore
                        "num_agents_interest": num_agents_interest,
                        "scenario_future_duration": int(
                            agent_record.obj_trajs_future_mask.shape[1]
                        ),
                        "num_map_polylines": int(agent_record.map_polylines.shape[0]),
                        "track_index_to_predict": int(
                            agent_record.track_index_to_predict
                        ),
                        "center_objects_type": int(agent_record.center_objects_type),
                        "dataset_name": agent_record.dataset_name,
                        "trajectory_type": (
                            int(agent_record.trajectory_type)
                            if agent_record.trajectory_type is not None
                            else None
                        ),
                    }
                    sample_metadata[split][grp_name] = meta

                del out
                del scen

        return sample_metadata

    def get_sample_metadata(self, split: Optional[Stage] = None) -> pd.DataFrame:
        """
        Load the sample metadata DataFrame for the given split.
        """
        split = split or self.config.stage
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "get_sample_metadata", split.name
        )
        idx_path = self.paths.get_sample_metadata_path(split)
        try:
            df = pd.read_pickle(idx_path)
            if self.config.num_debug_samples is not None:
                CONSOLE.log(
                    f"Limiting to {self.config.num_debug_samples} / {len(df)} samples in {split.name}."
                )
                df = df.sample(n=min(self.config.num_debug_samples, len(df)))
            return df
        except Exception as e:
            CONSOLE.error(f"Failed to load sample metadata DataFrame: {idx_path}: {e}")
            return pd.DataFrame()

    @abstractmethod
    def preprocess(self, scenario: RawScenarioDict) -> Optional[ProcessedDataDict]:
        """
        Preprocess a raw scenario dictionary into an internal format.

        Args:
            scenario (RawScenarioDict): Raw scenario data.

        Returns:
            Optional[InternalFormatDict]: Processed internal format dictionary, or None if invalid.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.preprocess() not implemented. "
            "Please implement this method in a subclass."
        )

    @abstractmethod
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
        raise NotImplementedError(
            f"{self.__class__.__name__}.process() not implemented. "
            "Please implement this method in a subclass."
        )

    @abstractmethod
    def postprocess(self, output: List[DatasetItem]) -> Optional[List[DatasetItem]]:
        """
        Perform post-processing steps like calculating difficulty and trajectory type.

        Args:
            output (List[DatasetItem]): List of processed data items from the process() step.

        Returns:
            Optional[List[DatasetItem]]: List of data items with added post-processing info, or None if input is invalid.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.postprocess() not implemented. "
            "Please implement this method in a subclass."
        )
