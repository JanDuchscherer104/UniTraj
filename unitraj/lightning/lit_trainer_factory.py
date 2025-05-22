import os
from pathlib import Path
from typing import List, Literal, Optional, Type, Union

import pytorch_lightning as pl
import torch
from pydantic import Field, field_validator, model_validator
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import Logger
from typing_extensions import Self

from ..configs.path_config import PathConfig
from ..configs.wandb_config import WandbConfig
from ..utils.base_config import BaseConfig
from ..utils.console import Console


# --- TrainerCallbacksConfig ---
class TrainerCallbacksConfig(BaseConfig):
    """
    Configuration for Trainer Callbacks.

    Controls which callbacks are enabled and their settings.
    """

    use_model_checkpoint: bool = True
    """"Whether to use ModelCheckpoint callback."""
    checkpoint_monitor: str = "val/brier_fde"
    """Metric to monitor for best checkpoint."""

    checkpoint_mode: Literal["min", "max"] = "min"
    """min: lower is better, max: higher is better."""

    checkpoint_filename: str = "{epoch:02d}-{val/brier_fde:.2f}"
    """Checkpoint filename template."""

    checkpoint_save_top_k: int = 1
    """Number of best models to save."""

    use_early_stopping: bool = False
    """Whether to use EarlyStopping callback."""

    early_stopping_monitor: str = "val/brier_fde"
    """Metric to monitor for early stopping."""

    early_stopping_mode: Literal["min", "max"] = "min"
    """min: lower is better, max: higher is better."""

    early_stopping_patience: int = 5
    """Number of epochs with no improvement before stopping."""

    use_lr_monitor: bool = True
    """Whether to use LearningRateMonitor callback."""

    lr_monitor_logging_interval: str = "epoch"
    """Logging interval for LR monitor (step or epoch)."""

    use_wandb: bool = True
    """Whether to use WandB logger."""

    wandb_config: WandbConfig = Field(default_factory=WandbConfig)
    """Advanced WandB configuration."""


# --- Update LitTrainerFactoryConfig to inherit from BaseConfig ---
class LitTrainerFactoryConfig(BaseConfig["LitTrainerFactory"]):
    """
    Configuration for creating a PyTorch Lightning Trainer instance.

    Acts as a factory for the Trainer. Inherits from BaseConfig.
    """

    target: Type["LitTrainerFactory"] = Field(
        default_factory=lambda: LitTrainerFactory, exclude=True
    )
    # Trainer settings
    accelerator: str = "auto"
    """Accelerator to use ('cpu', 'gpu', 'tpu', 'mps', 'auto')."""

    strategy: str = "auto"
    """Strategy for distributed training ('ddp', 'fsdp', 'auto')."""

    devices: Union[List[int], str, int] = "auto"
    """Devices to use (e.g., [0, 1], 'auto', 4)."""

    max_epochs: Optional[int] = None
    """Maximum number of epochs."""

    min_epochs: Optional[int] = None
    """Minimum number of epochs."""
    max_steps: int = -1
    """Maximum number of steps (-1 for no limit)."""

    min_steps: Optional[int] = None
    """Minimum number of steps."""

    precision: Union[str, int] = "32-true"
    """Training precision ('16-mixed', 'bf16-mixed', '32-true', '64-true')."""

    gradient_clip_val: Optional[float] = None
    """Gradient clipping value."""

    accumulate_grad_batches: int = 1
    """Accumulate gradients across batches."""

    log_every_n_steps: int = 50
    """Log metrics every N steps."""

    check_val_every_n_epoch: int = 1
    """Run validation every N epochs."""
    # Debugging and Profiling
    fast_dev_run: bool = False
    """Run a quick check with one batch. Will be set to True if is_debug=True."""

    profiler: Optional[str] = None
    """Profiler to use ('simple', 'advanced', 'pytorch')."""

    # Nested configs
    callbacks_config: TrainerCallbacksConfig = Field(
        default_factory=TrainerCallbacksConfig
    )
    """Configuration for callbacks."""

    # Use the imported PathConfig, which is a SingletonConfig
    path_config: PathConfig = Field(default_factory=PathConfig, exclude=True)
    """Path configuration for checkpoints."""

    # Experiment info (will be propagated if defined in parent ExperimentConfig)
    exp_name: str = "default_exp"
    """Experiment name for logging and checkpoints."""

    is_debug: bool = False
    """Whether running in debug mode (affects logger/callbacks)."""

    @field_validator("is_debug", mode="before")
    @classmethod
    def _propagate_debug(cls, v, info):
        if v:
            data = info.data
            # enable fast-dev-run and CPU when debugging
            data.setdefault("fast_dev_run", True)
            data.setdefault("accelerator", "cpu")
            # disable callbacks in debug
            callbacks = data.setdefault("callbacks_config", {})
            if isinstance(callbacks, dict):
                callbacks.setdefault("use_model_checkpoint", False)
                callbacks.setdefault("use_early_stopping", False)
                callbacks.setdefault("use_lr_monitor", False)
            # ensure single-batch behavior
            data.setdefault("accumulate_grad_batches", 1)
        return v

    @field_validator("precision")
    @classmethod
    def set_matmul_precision(cls, v):
        # Recommended by Lightning for TF32 on Ampere GPUs
        # Use Console for logging this setting
        CONSOLE = Console()
        if v in ["32-true", 32]:
            torch.set_float32_matmul_precision("medium")
            CONSOLE.log(
                f"Setting float32 matmul precision to 'medium' for precision='{v}'."
            )
        elif v in ["bf16-mixed"]:
            torch.set_float32_matmul_precision("medium")  # or high
            CONSOLE.log(
                f"Setting float32 matmul precision to 'medium' for precision='{v}'."
            )
        # Add other precision settings if needed
        return v

    @model_validator(mode="after")
    def validate_trainer_config(self) -> Self:
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "validate_trainer_config"
        )
        if self.is_debug:
            object.__setattr__(self, "fast_dev_run", True)
            object.__setattr__(self, "accelerator", "cpu")
            object.__setattr__(self.callbacks_config, "use_model_checkpoint", False)
            object.__setattr__(self.callbacks_config, "use_early_stopping", False)
            object.__setattr__(self.callbacks_config, "use_lr_monitor", False)
            object.__setattr__(self, "accumulate_grad_batches", 1)
            CONSOLE.log(
                "Debug mode enabled. Using CPU, fast_dev_run=True, and disabling callbacks."
            )

        return self


class LitTrainerFactory:
    """
    Factory class to create and configure PyTorch Lightning Trainer instances.
    """

    def __init__(self, config: LitTrainerFactoryConfig):
        """
        Initializes the LitTrainerFactory.

        Args:
            config (LitTrainerFactoryConfig): The configuration object.
        """
        self.config = config

    def _configure_logger(self) -> Optional[Logger]:
        """Configures the logger based on the config."""
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_configure_logger")
        cfg = self.config.callbacks_config
        if self.config.is_debug or not cfg.use_wandb:
            CONSOLE.log("Logger disabled (debug mode or use_wandb=False).")
            return None

        # Ensure wandb save directory exists using PathConfig
        wandb_config = cfg.wandb_config

        assert isinstance(wandb_config, WandbConfig)
        logger = wandb_config.setup_target()
        return logger

    def _configure_callbacks(self) -> List[Callback]:
        """Configures callbacks based on the config."""
        CONSOLE = Console.with_prefix(self.__class__.__name__, "_configure_callbacks")
        callbacks = []
        cfg = self.config.callbacks_config

        if cfg.use_model_checkpoint and not self.config.is_debug:
            # Use PathConfig for checkpoint directory
            checkpoint_dir = self.config.path_config.checkpoints / self.config.exp_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                monitor=cfg.checkpoint_monitor,
                filename=cfg.checkpoint_filename,
                save_top_k=cfg.checkpoint_save_top_k,
                mode=cfg.checkpoint_mode,
                dirpath=str(checkpoint_dir),
                save_last=True,  # Often useful to save the last epoch
            )
            callbacks.append(checkpoint_callback)
            # Use Console for logging
            CONSOLE.log(
                f"ModelCheckpoint configured: monitor='{cfg.checkpoint_monitor}', dir='{checkpoint_dir}', filename='{cfg.checkpoint_filename}'"
            )
        elif self.config.is_debug:
            # Use Console for logging
            CONSOLE.log("ModelCheckpoint disabled in debug mode.")

        # Add EarlyStopping callback if configured
        if cfg.use_early_stopping and not self.config.is_debug:
            early_stopping = EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                mode=cfg.early_stopping_mode,
                patience=cfg.early_stopping_patience,
            )
            callbacks.append(early_stopping)
            CONSOLE.log(
                f"EarlyStopping configured: monitor='{cfg.early_stopping_monitor}', patience={cfg.early_stopping_patience}"
            )

        # Add LearningRateMonitor callback if configured
        if cfg.use_lr_monitor and not self.config.is_debug:
            lr_monitor = LearningRateMonitor(
                logging_interval=cfg.lr_monitor_logging_interval
            )
            callbacks.append(lr_monitor)
            CONSOLE.log(
                f"LearningRateMonitor configured: interval='{cfg.lr_monitor_logging_interval}'"
            )

        return callbacks

    def create_trainer(self) -> pl.Trainer:
        """
        Creates and configures the PyTorch Lightning Trainer instance.

        Returns:
            pl.Trainer: The configured Trainer instance.
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "create_trainer")
        CONSOLE.log("Configuring logger...")
        logger = self._configure_logger()
        CONSOLE.log("Configuring callbacks...")
        callbacks = self._configure_callbacks()

        trainer = pl.Trainer(
            accelerator=self.config.accelerator,
            strategy=self.config.strategy,
            devices=self.config.devices,
            max_epochs=self.config.max_epochs,
            min_epochs=self.config.min_epochs,
            max_steps=self.config.max_steps,
            min_steps=self.config.min_steps,
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            log_every_n_steps=self.config.log_every_n_steps,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            fast_dev_run=self.config.fast_dev_run,
            profiler=self.config.profiler,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=str(
                self.config.path_config.checkpoints / self.config.exp_name
            ),
        )
        return trainer

    def find_latest_checkpoint(
        self, recursive: bool = True, pattern: str = "*.ckpt"
    ) -> Optional[str]:
        """
        Finds the latest checkpoint file based on the experiment name.

        This method implements a robust checkpoint search strategy:
        1. First looks for 'last.ckpt' in the experiment's checkpoint directory
        2. If not found, searches for checkpoint files matching the pattern
        3. Can search recursively through subdirectories
        4. Returns the most recently modified checkpoint file

        Args:
            recursive: Whether to search recursively in subdirectories. Defaults to True.
            pattern: File pattern to search for. Defaults to "*.ckpt".

        Returns:
            Optional[str]: Path to the latest checkpoint, or None if not found.
        """
        CONSOLE = Console.with_prefix(self.__class__.__name__, "find_latest_checkpoint")
        if self.config.is_debug:
            CONSOLE.log("Skipping checkpoint loading in debug mode.")
            return None

        # Use PathConfig for checkpoint directory
        checkpoint_dir = self.config.path_config.checkpoints / self.config.exp_name

        if not checkpoint_dir.exists():
            CONSOLE.log(f"Checkpoint directory not found: {checkpoint_dir}")
            return None

        # Look for last.ckpt first
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.is_file():
            CONSOLE.log(f"Found last checkpoint: {last_ckpt}")
            return str(last_ckpt)
        else:
            CONSOLE.dbg(f"'last.ckpt' not found or not a file in {checkpoint_dir}")

        # If no last.ckpt, find checkpoint files matching the pattern
        try:
            # Use rglob for recursive search or glob for non-recursive search
            if recursive:
                CONSOLE.log(f"Searching recursively for {pattern} in {checkpoint_dir}")
                checkpoints = list(checkpoint_dir.rglob(pattern))
            else:
                CONSOLE.log(f"Searching for {pattern} in {checkpoint_dir}")
                checkpoints = list(checkpoint_dir.glob(pattern))

            if not checkpoints:
                CONSOLE.log(
                    f"No checkpoint files matching '{pattern}' found in {checkpoint_dir}"
                )
                return None

            # Sort by modification time to find the latest
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            CONSOLE.log(f"Found latest checkpoint by time: {latest_checkpoint}")
            return str(latest_checkpoint)
        except Exception as e:
            CONSOLE.error(f"Error finding latest checkpoint in {checkpoint_dir}: {e}")
            return None
