from multiprocessing import cpu_count
from typing import Annotated, Any, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import Field, ValidationInfo, field_validator, model_validator
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from typing_extensions import Self

from ..datasets.base_dataset import BaseDataset, DatasetConfig
from ..datasets.types import BatchDict, DatasetItem, Stage
from ..utils.base_config import BaseConfig
from ..utils.console import Console


class LitDatamoduleConfig(BaseConfig["LitDatamodule"]):
    """
    Configuration for BaseLitDataModule.

    Acts as a factory for creating the BaseLitDataModule instance.
    Implements the Config-as-Factory pattern.
    """

    target: Type["LitDatamodule"] = Field(
        default_factory=lambda: LitDatamodule, exclude=True
    )

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    """Configuration for the dataset to be used."""

    batch_size: int = Field(32)
    """Batch size for training, validation, and testing."""

    num_workers: int = Field(default_factory=cpu_count)
    """Number of workers for data loading. 0 means main process, -1 for auto-detection."""

    pin_memory: bool = Field(True)
    """Whether to pin memory for data loading."""

    is_debug: bool = Field(False)
    """Debug mode flag for more verbose logging and simplified data loading."""

    @model_validator(mode="after")
    def validate(self) -> Self:
        if self.is_debug:
            CONSOLE = Console.with_prefix(self.__class__.__name__, "validate")
            CONSOLE.log("Debug mode is enabled. Setting num_workers to 0.")
            object.__setattr__(self, "num_workers", 0)

        return self


class LitDatamodule(pl.LightningDataModule):
    """
    Base PyTorch Lightning DataModule.

    Handles the creation of DataLoaders for train, validation, and test sets
    based on a provided dataset configuration.
    """

    def __init__(self, config: LitDatamoduleConfig):
        """
        Initializes the BaseLitDataModule.

        Args:
            config (LitDatamoduleConfig): The configuration object.
        """
        super().__init__()
        self.config = config
        # save_hyperparameters
        self.save_hyperparameters(
            {
                k: v
                for k, v in config.model_dump().items()
                if k not in ["target", "dataset_config"]
            }
        )

        self.train_dataset: Optional[BaseDataset] = None
        self.val_dataset: Optional[BaseDataset] = None
        self.test_dataset: Optional[BaseDataset] = None

    def setup(self, stage: Optional[Union[Stage, str]] = None):
        """
        Instantiates the datasets based on the stage.

        Args:
            stage (Optional[str]): Stage ('fit', 'validate', 'test', 'predict').
                                   Will be converted to Stage enum if needed.
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "setup")

        if stage is None:
            CONSOLE.warn("No stage provided to setup. Defaulting to 'fit' / 'TRAIN'.")
            stage = Stage.TRAIN

        assert isinstance(stage, (str, Stage)), "Stage must be a string or Stage enum"
        stage = stage if isinstance(stage, Stage) else Stage.from_str(stage)

        dataset_config = self.config.dataset.model_copy()
        dataset_config.stage = stage

        if stage == Stage.TRAIN:
            self.train_dataset = dataset_config.setup_target()
        elif stage == Stage.VAL:
            self.val_dataset = dataset_config.setup_target()
        elif stage == Stage.TEST:
            self.test_dataset = dataset_config.setup_target()
        else:
            CONSOLE.error(f"Unknown stage: {stage}. No datasets created.")

        CONSOLE.log(f"Setup dataset for stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for training.

        Returns:
            DataLoader: Configured DataLoader for training data
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "train_dataloader")
        CONSOLE.set_debug(getattr(self.config, "is_debug", False))

        if self.train_dataset is None:
            CONSOLE.log("Training dataset not initialized, calling setup('fit')")
            self.setup("fit")

        if self.train_dataset is None:
            CONSOLE.error("Training dataset is still None after setup")
            raise RuntimeError(
                "Training dataset not initialized. Check setup() implementation."
            )

        CONSOLE.log(
            f"Creating training DataLoader: batch_size={self.config.batch_size}, "
            f"workers={self.config.num_workers}, shuffle=True"
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=True,  # Drop last batch during training
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for validation.

        Returns:
            DataLoader: Configured DataLoader for validation data
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "val_dataloader")
        CONSOLE.set_debug(getattr(self.config, "is_debug", False))

        if self.val_dataset is None:
            CONSOLE.log("Validation dataset not initialized, calling setup('validate')")
            self.setup("validate")

        if self.val_dataset is None:
            CONSOLE.error("Validation dataset is still None after setup")
            raise RuntimeError(
                "Validation dataset not initialized. Check setup() implementation."
            )

        CONSOLE.log(
            f"Creating validation DataLoader: batch_size={self.config.batch_size}, "
            f"workers={self.config.num_workers}, shuffle=False"
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,  # Keep all samples during validation
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for testing.

        Returns:
            DataLoader: Configured DataLoader for test data
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "test_dataloader")
        CONSOLE.set_debug(getattr(self.config, "is_debug", False))

        if self.test_dataset is None:
            CONSOLE.log("Test dataset not initialized, calling setup('test')")
            self.setup("test")

        if self.test_dataset is None:
            CONSOLE.error("Test dataset is still None after setup")
            raise RuntimeError(
                "Test dataset not initialized. Check setup() implementation."
            )

        CONSOLE.log(
            f"Creating test DataLoader: batch_size={self.config.batch_size}, "
            f"workers={self.config.num_workers}, shuffle=False"
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,  # Keep all samples during testing
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up datasets, close files, etc.

        Args:
            stage (Optional[str]): Stage being torn down ('fit', 'validate', 'test').
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "teardown")
        CONSOLE.set_debug(getattr(self.config, "is_debug", False))

        CONSOLE.log(f"Tearing down datamodule for stage: {stage}")

        # Create a list of (dataset, name) tuples, filtering out None values
        datasets_to_close: List[Tuple[Any, str]] = [
            (ds, name)
            for ds, name in [
                (self.train_dataset, "training"),
                (self.val_dataset, "validation"),
                (self.test_dataset, "test"),
            ]
            if ds is not None
        ]

        # Close files for each dataset
        for ds, name in datasets_to_close:
            if hasattr(ds, "close_files"):
                try:
                    CONSOLE.dbg(f"Closing files for {name} dataset")
                    ds.close_files()
                except Exception as e:
                    CONSOLE.error(f"Error closing files for {name} dataset: {e}")

    def collate_fn(self, batch: List[Optional[DatasetItem]]) -> Optional[BatchDict]:
        """
        Collates a list of DatasetItems into a BatchDict for model input.

        Args:
            batch (List[Optional[DatasetItem]]): List of data items, potentially containing None values.

        Returns:
            Optional[BatchDict]: A dictionary containing batched tensors, or None if the batch is empty.
        """
        # Filter out None values with a comprehension
        data_list = [item for item in batch if item is not None]
        if not data_list:
            return None

        # Convert each DatasetItem to a tensor dictionary using the to_tensor_dict method
        batch_size = len(data_list)
        batch_dicts = [item.to_tensor_dict() for item in data_list]

        # Get all keys from the first item (all items should have the same keys)
        all_keys = batch_dicts[0].keys()

        # Prepare the batch dictionary more efficiently
        input_dict = {}

        for key in all_keys:
            values = [d[key] for d in batch_dicts]

            # Special handling for different data types
            if isinstance(values[0], torch.Tensor):
                try:
                    # Use torch.stack directly for tensors - more efficient
                    input_dict[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    # For tensors that can't be stacked, try to convert list values to tensor
                    if all(isinstance(v, (str, int, float)) for v in values):
                        try:
                            # Convert simple types to tensor when possible
                            if all(isinstance(v, int) for v in values):
                                input_dict[key] = torch.tensor(values, dtype=torch.long)
                            elif all(isinstance(v, float) for v in values):
                                input_dict[key] = torch.tensor(
                                    values, dtype=torch.float
                                )
                            else:
                                input_dict[key] = (
                                    values  # Keep as list if conversion fails
                                )
                        except:
                            input_dict[key] = values
                    else:
                        input_dict[key] = values  # Keep as list for complex objects
            elif isinstance(values[0], np.ndarray):
                try:
                    # Optimize: convert to tensor in one step
                    input_dict[key] = torch.from_numpy(np.stack(values, axis=0))
                except:
                    # If stacking fails, try to convert to tensor if elements are simple types
                    input_dict[key] = values
            elif isinstance(values[0], (int, np.integer)):
                # Convert numeric lists to tensors
                input_dict[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], (float, np.floating)):
                # Convert numeric lists to tensors
                input_dict[key] = torch.tensor(values, dtype=torch.float)
            elif all(isinstance(v, str) for v in values):
                # Keep strings as list, but ensure they're all strings
                input_dict[key] = values
            else:
                # Other types remain as lists
                input_dict[key] = values

        # Convert specific fields that might need special handling
        if "center_objects_type" in input_dict and isinstance(
            input_dict["center_objects_type"], torch.Tensor
        ):
            # Keep center_objects_type as numpy array as expected by downstream code
            input_dict["center_objects_type"] = input_dict[
                "center_objects_type"
            ].numpy()

        if "track_index_to_predict" in input_dict and isinstance(
            input_dict["track_index_to_predict"], list
        ):
            # Convert track_index_to_predict to tensor if it's still a list
            try:
                input_dict["track_index_to_predict"] = torch.tensor(
                    input_dict["track_index_to_predict"], dtype=torch.long
                )
            except:
                pass  # Keep as list if conversion fails

        if "center_gt_final_valid_idx" in input_dict and isinstance(
            input_dict["center_gt_final_valid_idx"], list
        ):
            # Convert center_gt_final_valid_idx to tensor if it's still a list
            try:
                input_dict["center_gt_final_valid_idx"] = torch.tensor(
                    input_dict["center_gt_final_valid_idx"], dtype=torch.float
                )
            except:
                pass  # Keep as list if conversion fails

        # Create the batch dictionary with the required structure
        batch_dict: BatchDict = {
            "batch_size": batch_size,
            "input_dict": input_dict,
            "batch_sample_count": batch_size,
        }

        return batch_dict
