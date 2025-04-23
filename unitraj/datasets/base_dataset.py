import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import h5py
import hydra
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

# Use tqdm.auto for automatic notebook/console detection
from tqdm.auto import tqdm

from ..configs.path_config import PathConfig
from ..utils.base_config import CONSOLE, BaseConfig
from ..utils.visualization import check_loaded_data
from .dataparser import DataParserConfig

# Import everything from types except DatasetItem which we're implementing here
from .types import (
    BatchDict,
    DatasetItem,
    InternalFormatDict,
    Stage,
    object_type,
    polyline_type,
)

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
# TODO: do not overwrite cache by default


class DatasetConfig(BaseConfig["BaseDataset"]):
    """Configuration for the BaseDataset."""

    target: Type["BaseDataset"] = Field(default_factory=lambda: BaseDataset)

    stage: Stage = Field(Stage.TRAIN)
    """
    STAGE.TRAIN, STAGE.VAL, STAGE.TEST
    """

    is_debug: bool = Field(
        False, description="Debug mode (no MP, limited samples, CPU only)"
    )

    # Object and map configuration
    object_type: List[str] = Field(
        ["VEHICLE"], description="Object types included in the training set"
    )

    data_parser: DataParserConfig = Field(default_factory=DataParserConfig)
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


class BaseDataset(Dataset):

    def __init__(self, config: DatasetConfig):
        self.config = config

        self.data_loaded_memory: List[Dict[str, Any]] = []
        self.file_cache: Dict[str, Optional[h5py.File]] = {}

    def collate_fn(self, data_list: List[Optional[DatasetItem]]) -> Optional[BatchDict]:
        """
        Collates a list of DatasetItems into a BatchDict for model input.

        Args:
            data_list (List[Optional[DatasetItem]]): List of data items, potentially containing None values.

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
        if not self.config.paths.check_exists(file_path):
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
        group_name = file_info["h5_group"]  # Use the stored group name

        # Get or create the H5 file handle
        if file_path not in self.file_cache:
            self.file_cache[file_path] = self._get_file(file_path)

        if self.file_cache[file_path] is None:
            CONSOLE.error(f"Failed to open HDF5 file: {file_path}")
            # Return an empty DatasetItem or raise an error
            raise RuntimeError(f"Could not access HDF5 file: {file_path}")

        # Access the group
        h5_file = self.file_cache[file_path]
        group = h5_file[group_name]

        # Read all fields from the group into a dict
        record_dict = {
            k: (
                group[k][()].decode("utf-8")
                if isinstance(group[k].dtype.type, np.bytes_)
                else group[k][()]
            )
            for k in group.keys()
        }

        # Create and return a DatasetItem
        try:
            return DatasetItem(**record_dict)
        except Exception as e:
            CONSOLE.error(f"Error creating DatasetItem from HDF5 data: {e}")
            # Log problematic fields to help with debugging
            for k, v in record_dict.items():
                if isinstance(v, np.ndarray):
                    CONSOLE.error(f"Field {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    CONSOLE.error(f"Field {k}: type={type(v)}")
            raise

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
        cache_dir = self.config.paths.get_cache_path(dataset_name, phase)
        file_list_path = cache_dir / "file_list.pkl"

        # Use PathConfig method to check existence
        if self.config.paths.check_exists(file_list_path):
            try:
                # Use path.open()
                with file_list_path.open("rb") as f:
                    data_loaded = pickle.load(f)
            except Exception as e:
                raise ValueError(
                    f"Error loading file_list.pkl from {file_list_path}: {e}",
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

    # --- Unused/Debugging Methods ---
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

        return sampled_array, sampled_indices

    def trajectory_filter(self, data: InternalFormatDict) -> Dict[str, Any]:
        """
        Filter trajectories to select valid tracks for prediction.

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
    try:
        set_seed(cfg.seed)
        OmegaConf.set_struct(cfg, False)
        dataset_cfg = cfg.dataset  # Assuming dataset config is under 'dataset' key
        train_set = build_dataset(dataset_cfg)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
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
        except Exception as e:
            CONSOLE.error(f"Error concatenating or saving images: {e}")
    except Exception as e:
        CONSOLE.error(f"An error occurred in draw_figures: {e}")
    finally:
        if train_set is not None and hasattr(train_set, "close_files"):
            train_set.close_files()


if __name__ == "__main__":
    CONSOLE.log("BaseDataset module loaded. Run hydra functions like 'draw_figures'.")
