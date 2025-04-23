# myproject/parsing/parser.py

import pickle
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
from metadrive.scenario import utils as sd_utils
from scenarionet.common_utils import read_scenario
from tqdm.auto import tqdm

from ..configs.path_config import PathConfig
from ..utils.base_config import CONSOLE
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


class BaseDataParser(ABC):
    """
    Splits raw ScenarioNet summaries into shards and runs
    BaseDataset.process_data_chunk in parallel to build cache.
    """

    def __init__(self, config: "DataParserConfig"):
        self.config = config
        self.paths = PathConfig(root=config.cache_root)

    def initialize_progress_bars(self, data_lists):
        """
        Initialize multiple progress bars for parallel processing.
        Creates one progress bar for each worker to track their individual progress.

        Args:
            data_lists (List[List[str]]): Split data lists for each worker

        Returns:
            Dict[int, tqdm]: Dictionary mapping worker indices to their progress bars
        """

        progress_bars = {}
        for worker_idx, data_list in enumerate(data_lists):
            desc = f"Worker {worker_idx}"
            position = worker_idx  # Stack bars vertically by position
            progress_bars[worker_idx] = tqdm(
                total=len(data_list),
                desc=desc,
                position=position,
                leave=True,
                unit="scenario",
                miniters=1,
                dynamic_ncols=True,
                colour="green" if worker_idx % 2 == 0 else "blue",  # Alternate colors
            )

        # Store as instance attribute to access in other methods
        self._shared_progress_bars = progress_bars
        return progress_bars

    def build_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        For each raw_data_dir:
          - read its summary (ids → files)
          - optionally sample debug subset
          - split IDs across workers
          - spawn process_data_chunk
          - collect and merge all file_list dicts
        Returns final merged file_list.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        for raw_dir in self.config.raw_data_dirs:
            split_name = raw_dir.name
            CONSOLE.log(f"scanning {raw_dir}")
            # 1) read summary
            try:
                meta_map, id_list, id_to_file = sd_utils.read_dataset_summary(raw_dir)
            except Exception as e:
                raise RuntimeError(f"Failed summary read of {raw_dir}: {e}")

            # 2) debug sampling
            if self.config.is_debug:
                np.random.seed(0)
                id_list = list(
                    np.random.choice(
                        id_list,
                        size=min(self.config.num_debug_samples, len(id_list)),
                        replace=False,
                    )
                )
                CONSOLE.log(f"debug-sampling {len(id_list)} of {len(id_list)}")

            # 3) split into shards
            n_workers = 1 if self.config.is_debug else self.config.get_num_workers()
            splits = np.array_split(id_list, n_workers)
            args = [
                (str(raw_dir), id_to_file, list(s), split_name, self.config.stage, i)
                for i, s in enumerate(splits)
            ]

            # Initialize progress bars for all workers
            if self.config.has_tqdm:
                self.initialize_progress_bars([list(s) for s in splits])

            # 4) dispatch
            if self.config.is_debug:
                results = [self.process_data_chunk(self, *a) for a in args]
            else:
                with Pool(n_workers) as pool:
                    results = pool.starmap(self.process_data_chunk, args)

            # 5) collect
            for r in results:
                if r:
                    merged.update(r)

        # 6) persist the merged index for fast reload
        idx_file = self.paths.cache / f"file_list_{self.config.stage}.pkl"
        with idx_file.open("wb") as f:
            pickle.dump(merged, f)
        CONSOLE.log(f"wrote index {len(merged)} entries → {idx_file}")
        return merged

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

        # --- Create / open HDF5 file and write each scenario's outputs ---
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

                    # Convert DatasetItem to dict (using field values, not model internals)
                    record_dict = record.model_dump()

                    # Write each field to the HDF5 group
                    for key, value in record_dict.items():
                        try:
                            # Handle string conversion explicitly
                            if isinstance(value, str):
                                value = np.bytes_(value)
                            # Attempt to create dataset
                            grp.create_dataset(key, data=value)
                        except TypeError as te:
                            CONSOLE.warn(
                                f"HDF5 TypeError for key '{key}' in group '{grp_name}': {te}. Skipping key."
                            )

                    file_info = {
                        # Get kalman_difficulty from the first record (all records in a scenario should have the same value)
                        "kalman_difficulty": (
                            record.kalman_difficulty
                            if record.kalman_difficulty is not None
                            else np.array([0.0])
                        ),
                        "h5_path": str(hdf5_path),  # Store path as string
                        "h5_group": grp_name,  # Store the group name for easier lookup
                    }
                    file_list[grp_name] = file_info  # Use group name as the key

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
