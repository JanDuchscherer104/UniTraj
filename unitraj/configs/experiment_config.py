from multiprocessing import cpu_count
from typing import List, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pydantic import Field
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from ..datasets.base_dataset import DatasetConfig
from ..datasets.dataparser import DataParserConfig
from ..models.autobot.autobot import AutoBotConfig
from ..utils.base_config import BaseConfig, Console
from ..utils.utils import find_latest_checkpoint, set_seed
from .path_config import PathConfig


class ExperimentConfig(BaseConfig):
    # experiment settings
    exp_name: str = Field("test", description="Name used in wandb and checkpoints")
    seed: int = Field(42, description="Random seed for reproducibility")
    is_debug: bool = Field(True, description="Debug mode (CPU only)")
    devices: List[int] = Field([0], description="List of GPU device IDs")
    num_workers: int = Field(-1, description="Number of workers for data loading")

    # evaluation settings
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

        CONSOLE = Console.with_prefix(self.__class__.__name__, "setup_target")
        CONSOLE.set_debug(self.is_debug)

        if self.is_debug:
            self.num_workers = 1
        else:
            if (
                self.num_workers == -1
                or self.num_workers is None
                or self.num_workers > cpu_count()
            ):
                self.num_workers = cpu_count()

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
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=train_set.collate_fn,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=eval_batch_size,
            num_workers=self.num_workers,
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

        if self.paths.checkpoints is not None and not self.is_debug:
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
