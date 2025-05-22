from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

from pydantic import Field
from pytorch_lightning.loggers import WandbLogger

import wandb

from ..utils.base_config import BaseConfig
from .path_config import PathConfig


class WandbConfig(BaseConfig):
    # Essential fields directly matching WandbLogger parameters
    name: Optional[str] = Field(None, description="Display name for the run.")
    project: Optional[str] = "unitraj"
    """Name of the wandb project."""
    save_dir: Path = Field(
        default_factory=lambda: PathConfig().wandb,
        serialization_alias="dir",
        description="Directory to save wandb logs.",
    )
    offline: Optional[bool] = Field(
        False,
        description="Run offline mode.",
    )
    log_model: Union[Literal["all"], bool] = Field(
        False,
        description="Log model checkpoints as wandb artifacts.",
    )
    prefix: Optional[str] = Field(
        "",
        description="Str prefix for beginning of metric keys.",
    )
    checkpoint_name: Optional[str] = Field(
        None, description="Name of model checkpoint artifact."
    )
    tags: Optional[list[str]] = Field(
        None, description="List of tags for easier filtering."
    )

    group: Optional[str] = Field(None, description="Group name for multiple runs.")

    def setup_target(self, **kwargs: Any) -> WandbLogger:
        """
        Initializes the [WandbLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) with the specified configurations.

        For kwargs refer to [wandb.init](https://docs.wandb.ai/ref/python/init/)
        """
        logger = WandbLogger(
            name=self.name,
            save_dir=self.save_dir,
            offline=self.offline,
            project=self.project,
            log_model=self.log_model,
            prefix=self.prefix,
            experiment=wandb.run,
            checkpoint_name=self.checkpoint_name,
            tags=self.tags,
            group=self.group,
            **(kwargs or {}),
        )

        return logger
