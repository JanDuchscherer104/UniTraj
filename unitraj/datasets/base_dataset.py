import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ..configs.path_config import PathConfig
from ..utils.base_config import BaseConfig, Console
from ..utils.visualization import check_loaded_data
from .dataparser import DataParserConfig
from .types import BatchDict, DatasetItem, InternalFormatDict, ObjectType, Stage

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

    # Paths - Ensure PathConfig is used
    paths: PathConfig = Field(default_factory=PathConfig)
    parser: DataParserConfig = Field(default_factory=DataParserConfig)
    """
    Attributes (all of type Path):
        root: Path to the root of the project
        data: path to the data root
        checkpoints
        cache
        temp_dir
        split_data_output_dir
    """

    def setup_target(self) -> "BaseDataset":
        self.parser.stage = self.stage
        return self.target(self)


class BaseDataset(Dataset):

    def __init__(self, config: DatasetConfig):
        self.config = config

        self.data_samples = (
            self.config.parser.setup_target().load_data().get_sample_metadata()
        )
        # Reset index name to group name
        self.data_samples.index.name = "group_name"

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

        # stack arrays -> Tensors, leave other objects alone
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
        return len(self.data_samples)

    @lru_cache(maxsize=None)
    def _get_file(self, file_path):
        return h5py.File(file_path, "r")

    def __getitem__(self, idx: int) -> DatasetItem:
        # Look up group metadata
        row = self.data_samples.iloc[idx]
        h5_path: Path = row["h5_path"]
        group_name: str = row.name
        # Open or retrieve cached HDF5 file
        h5f = self._get_file(h5_path)
        grp = h5f[group_name]
        # Read all datasets in the group
        data: Dict[str, Any] = {}
        for key in grp:
            arr = grp[key][()]
            if isinstance(arr, bytes) or (
                isinstance(arr, np.ndarray) and arr.dtype.type == np.bytes_
            ):
                # decode byte strings
                val = (
                    arr.decode("utf-8")
                    if isinstance(arr, (bytes, bytearray))
                    else arr.astype(str)
                )
            else:
                val = arr
            data[key] = val
        return DatasetItem(**data)

    # --- Unused/Debugging Methods ---
    def sample_from_distribution(self, original_array, m=100):
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "sample_from_distribution"
        )

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


# # --- Hydra Main Functions ---
# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def draw_figures(cfg):
#     CONSOLE = Console.with_prefix(__file__.split("/")[-1], "draw_figures")
#     try:
#         set_seed(cfg.seed)
#         OmegaConf.set_struct(cfg, False)
#         dataset_cfg = cfg.dataset  # Assuming dataset config is under 'dataset' key
#         train_set = build_dataset(dataset_cfg)
#         train_loader = torch.utils.data.DataLoader(
#             train_set,
#             batch_size=1,
#             shuffle=True,
#             num_workers=0,
#             collate_fn=train_set.collate_fn,
#         )

#         concat_list = [4, 4, 4, 4, 4, 4, 4, 4]  # Example layout
#         images = []
#         num_images_to_draw = sum(concat_list)
#         CONSOLE.log(f"Attempting to draw {num_images_to_draw} figures...")

#         for i, data in enumerate(tqdm(train_loader, desc="Generating figures")):
#             if i >= num_images_to_draw:
#                 break
#             if data is None or "input_dict" not in data:
#                 CONSOLE.warn(
#                     f"Skipping batch {i} due to None data or missing 'input_dict'."
#                 )
#                 continue
#             inp = data["input_dict"]
#             try:
#                 # Assuming check_loaded_data returns a PIL image or similar
#                 img_bytes_io = check_loaded_data(
#                     inp, 0
#                 )  # Draw the first sample in the batch
#                 if img_bytes_io:
#                     img = Image.open(img_bytes_io)
#                     images.append(img)
#                 else:
#                     CONSOLE.warn(f"check_loaded_data returned None for batch {i}")
#             except Exception as e:
#                 CONSOLE.error(f"Error generating figure for batch {i}: {e}")

#         if not images:
#             CONSOLE.error("No images were generated.")
#             return

#         try:
#             CONSOLE.log(f"Concatenating {len(images)} generated images...")
#             final_image = concatenate_varying(images, concat_list)
#             # Save or display the image
#             output_path = Path("visualization_output.png")  # Save in current dir
#             final_image.save(output_path)
#             CONSOLE.log(f"Saved final visualization to {output_path.resolve()}")
#         except Exception as e:
#             CONSOLE.error(f"Error concatenating or saving images: {e}")
#     except Exception as e:
#         CONSOLE.error(f"An error occurred in draw_figures: {e}")
#     finally:
#         if train_set is not None and hasattr(train_set, "close_files"):
#             train_set.close_files()


# if __name__ == "__main__":
#     CONSOLE.log("BaseDataset module loaded. Run hydra functions like 'draw_figures'.")
