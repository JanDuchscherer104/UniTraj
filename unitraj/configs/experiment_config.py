import os
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from ..datasets.base_dataset import DatasetConfig
from ..models.autobot.autobot import AutoBotConfig
from ..utils.base_config import CONSOLE, BaseConfig
from ..utils.utils import find_latest_checkpoint, set_seed
from .path_config import PathConfig


class ExperimentConfig(BaseConfig):
    # experiment settings
    exp_name: str = Field("test", description="Name used in wandb and checkpoints")
    ckpt_path: Optional[str] = Field(
        None, description="Checkpoint path for resume/eval"
    )
    seed: int = Field(42, description="Random seed for reproducibility")
    is_debug: bool = Field(True, description="Debug mode (CPU only)")
    devices: List[int] = Field([0], description="List of GPU device IDs")

    # data loaders & paths
    load_num_workers: int = Field(0, description="Num workers for DataLoader")
    train_data_path: List[str] = Field(
        ["/work/share/argoverse2_scenarionet/av2_scenarionet"],
        description="Paths to training dataset shards",
    )
    val_data_path: List[str] = Field(
        ["/work/share/argoverse2_scenarionet/av2_scenarionet"],
        description="Paths to validation dataset shards",
    )
    cache_path: str = Field("./cache", description="Path to cache directory")

    # dataset slicing
    max_data_num: List[Optional[int]] = Field(
        [None], description="Max samples per dataset (None = all)"
    )
    starting_frame: List[int] = Field(
        [0], description="Starting frame for each dataset"
    )

    # trajectory params
    past_len: int = Field(21, description="Length of observed history (frames)")
    future_len: int = Field(60, description="Length of prediction horizon (frames)")
    trajectory_sample_interval: int = Field(
        1, description="Sampling interval for trajectory frames"
    )

    # filtering & map inputs
    object_type: List[str] = Field(["VEHICLE"], description="Object types to include")
    line_type: List[str] = Field(
        ["lane", "stop_sign", "road_edge", "road_line", "crosswalk", "speed_bump"],
        description="Types of map elements used",
    )
    masked_attributes: List[str] = Field(
        ["z_axis", "size"], description="Attributes to mask in trajectory input"
    )
    only_train_on_ego: bool = Field(False, description="Train only on ego trajectories")
    center_offset_of_map: List[float] = Field(
        [30.0, 0.0], description="Map center offset in local frame"
    )

    # caching
    use_cache: bool = Field(False, description="Whether to use disk cache")
    overwrite_cache: bool = Field(False, description="Overwrite cached files")
    store_data_in_memory: bool = Field(
        False, description="Keep entire dataset in memory"
    )

    # evaluation settings
    nuscenes_dataroot: Optional[str] = Field(
        None,
        description="Path to nuScenes dataset root for evaluation",
    )
    eval_nuscenes: bool = Field(False, description="Use nuScenes evaluation tool")
    eval_waymo: bool = Field(False, description="Use Waymo evaluation tool")
    eval_argoverse2: bool = Field(False, description="Use Argoverse2 evaluation tool")

    # nested configs
    method: Union[AutoBotConfig, DictConfig] = Field(
        default_factory=AutoBotConfig, description="Model config"
    )
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    paths: PathConfig = Field(
        default_factory=PathConfig, description="Path configuration"
    )

    def setup_target(self, stage: str = "train"):
        torch.set_float32_matmul_precision("medium")
        set_seed(self.seed)

        CONSOLE.set_debug(self.is_debug)

        model = self.method.setup_target()
        # Override dataset paths with experiment data paths
        from pathlib import Path

        # Training data
        # train_paths = [Path(p) for p in self.train_data_path]
        # self.dataset.paths.data = train_paths
        train_set = self.dataset.setup_target()
        # Validation data
        # val_paths = [Path(p) for p in self.val_data_path]
        # self.dataset.paths.data = val_paths
        val_set = self.dataset.setup_target(is_validation=True)

        train_batch_size = max(self.method.train_batch_size // len(self.devices), 1)
        eval_batch_size = max(self.method.eval_batch_size // len(self.devices), 1)

        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=self.load_num_workers,
            drop_last=False,
            collate_fn=train_set.collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=eval_batch_size,
            num_workers=self.load_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=train_set.collate_fn,
        )

        callbacks = [
            ModelCheckpoint(
                monitor="val/brier_fde",
                filename="{epoch}-{val/brier_fde:.2f}",
                save_top_k=1,
                mode="min",
                dirpath=self.paths.checkpoints / self.exp_name,
            )
        ]

        trainer = pl.Trainer(
            max_epochs=self.method.max_epochs,
            logger=(
                None
                if self.is_debug
                else WandbLogger(
                    project="unitraj",
                    name=self.exp_name,
                    id=self.exp_name,
                )
            ),
            devices=1 if self.is_debug else self.devices,
            gradient_clip_val=self.method.grad_clip_norm,
            accelerator="cpu" if self.is_debug else "gpu",
            profiler="simple",
            strategy="auto" if self.is_debug else "ddp",
            callbacks=callbacks,
        )

        if self.ckpt_path is None and not self.is_debug:
            # search_pattern = os.path.join("./unitraj", self.exp_name, "**", "*.ckpt")
            search_pattern = self.paths.checkpoints / self.exp_name / "**" / "*.ckpt"
            ckpt_path = find_latest_checkpoint(search_pattern.as_posix())

            if ckpt_path is not None:
                ckpt_path = Path(ckpt_path)
                assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
                self.ckpt_path = ckpt_path
                CONSOLE.print(f"Found checkpoint: {self.ckpt_path}")

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=self.ckpt_path,
        )
        return trainer
