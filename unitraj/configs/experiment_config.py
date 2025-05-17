from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from ..datasets.types import Stage
from ..lightning.lit_datamodule import LitDatamodule, LitDatamoduleConfig
from ..lightning.lit_trainer_factory import LitTrainerFactory, LitTrainerFactoryConfig
from ..models import AutoBotConfig
from ..models.base_model.base_model import BaseModel
from ..utils.base_config import BaseConfig
from ..utils.console import Console
from ..utils.utils import set_seed
from .path_config import PathConfig


class ExperimentConfig(BaseConfig):
    # experiment settings
    seed: Optional[int] = 42
    """Random seed for reproducibility"""

    is_debug: bool = True
    """Debug mode (affects logging, checkpoints, etc.)"""

    # evaluation settings
    eval_nuscenes: bool = False
    """Use nuScenes evaluation tool"""

    eval_waymo: bool = False
    """Use Waymo evaluation tool"""

    eval_argoverse2: bool = False
    """Use Argoverse2 evaluation tool"""

    # Use new Lightning configs
    model: AutoBotConfig = Field(default_factory=AutoBotConfig)
    """Lightning Module configuration (specific model config)."""

    datamodule: LitDatamoduleConfig = Field(default_factory=LitDatamoduleConfig)
    """Lightning DataModule configuration."""

    trainer: LitTrainerFactoryConfig = Field(default_factory=LitTrainerFactoryConfig)
    """Lightning Trainer factory configuration."""

    # Keep PathConfig
    paths: PathConfig = Field(default_factory=PathConfig)
    """Path configuration"""

    ckpt_path: Optional[Annotated[Path, str]] = Field(None)
    """Path to a checkpoint to load. Relative to paths.checkpoints."""

    @field_validator(
        "ckpt_path",
        mode="before",
    )
    @classmethod
    def _convert_to_path(
        cls, v: Union[str, Path], info: ValidationInfo
    ) -> Optional[Path]:
        if v is None:
            return None
        if isinstance(v, (str, Path)):
            paths = info.data.get("paths")
            assert (
                paths is not None
            ), "PathConfig must be initialized before using them."
            assert isinstance(paths, PathConfig)

            v = paths.checkpoints / v if not Path(v).is_absolute() else Path(v)
        v = v.resolve()

        assert v.exists(), f"Checkpoint path {v} does not exist."

        return v

    @model_validator(mode="before")
    @classmethod
    def _propagate_is_debug_to_data(
        cls, data: Dict[str, Any]
    ) -> Dict[str, Any]:  # Renamed for clarity
        is_debug = data.get("is_debug", False)  # Default to False if not present

        # Ensure 'method' is a dict and set 'is_debug' if not already present
        method_cfg = data.get("method", {})
        if isinstance(method_cfg, dict):
            method_cfg.setdefault("is_debug", is_debug)
            data["method"] = method_cfg
        elif isinstance(method_cfg, BaseModel) and hasattr(
            method_cfg, "is_debug"
        ):  # If already an instance
            if getattr(method_cfg, "is_debug", None) is None:  # only set if not set
                setattr(method_cfg, "is_debug", is_debug)

        # Ensure 'datamodule' is a dict and set 'is_debug' if not already present
        datamodule_cfg = data.get("datamodule", {})
        if isinstance(datamodule_cfg, dict):
            datamodule_cfg.setdefault("is_debug", is_debug)
            data["datamodule"] = datamodule_cfg
        elif isinstance(datamodule_cfg, BaseModel) and hasattr(
            datamodule_cfg, "is_debug"
        ):
            if getattr(datamodule_cfg, "is_debug", None) is None:
                setattr(datamodule_cfg, "is_debug", is_debug)

        # Ensure 'trainer' is a dict and set 'is_debug' if not already present
        trainer_cfg = data.get("trainer", {})
        if isinstance(trainer_cfg, dict):
            trainer_cfg.setdefault("is_debug", is_debug)
            data["trainer"] = trainer_cfg
        elif isinstance(trainer_cfg, BaseModel) and hasattr(trainer_cfg, "is_debug"):
            if getattr(trainer_cfg, "is_debug", None) is None:
                setattr(trainer_cfg, "is_debug", is_debug)

        return data

    @model_validator(mode="after")
    def _propagate_path_config_to_children(self) -> Self:
        """
        Propagates the main 'paths' config to children that have a 'paths' attribute.
        For this to work optimally with TOML serialization (avoiding duplicates),
        the 'paths' field in child configurations (e.g., AutoBotConfig,
        LitDatamoduleConfig, LitTrainerFactoryConfig) should be defined as:
        `paths: Optional[PathConfig] = Field(None, exclude=True)`
        """
        if self.paths:  # Ensure self.paths itself is initialized
            sub_configs_to_update = []
            if hasattr(self, "method") and self.model is not None:
                sub_configs_to_update.append(self.model)
            if hasattr(self, "datamodule") and self.datamodule is not None:
                sub_configs_to_update.append(self.datamodule)
            if hasattr(self, "trainer") and self.trainer is not None:
                sub_configs_to_update.append(self.trainer)

            for sub_cfg in sub_configs_to_update:
                if hasattr(sub_cfg, "paths"):
                    setattr(sub_cfg, "paths", self.paths)
        return self

    @model_validator(mode="after")
    def _set_seed(self) -> Self:
        """
        Sets the random seed for reproducibility.
        """
        if self.seed is not None:
            Console.with_prefix(self.__class__.__name__, "_set_seed").log(
                f"Setting random seed to {self.seed} (np, torch, cuda, random, pl)"
            )
            set_seed(self.seed)
        return self

    def setup_target(self) -> Tuple[pl.Trainer, BaseModel, LitDatamodule]:
        """
        Sets up and runs the training/evaluation process using PyTorch Lightning.

        Args:
            stage (str): The stage to run ('fit', 'test', 'validate', 'predict'). Defaults to 'fit'.
        """

        CONSOLE = Console.with_prefix(self.__class__.__name__, "setup_target")

        # Instantiate components
        CONSOLE.log("Instantiating lightning components...")
        datamodule = self.datamodule.setup_target()
        model = self.model.setup_target()
        trainer_factory = self.trainer.setup_target()

        trainer: pl.Trainer = trainer_factory.create_trainer()

        return trainer, model, datamodule

    def setup_target_and_run(
        self, stage: Union[Stage, Literal["train", "fit", "val", "test"]] = Stage.TRAIN
    ) -> pl.Trainer:
        stage = stage if isinstance(stage, Stage) else Stage.from_str(stage)
        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "setup_target_and_run", stage.name
        )

        ckpt_path = None
        if self.ckpt_path is not None:
            CONSOLE.log(f"Will contunue from checkpoint: {self.ckpt_path}")
            ckpt_path = self.ckpt_path

        trainer, model, datamodule = self.setup_target()

        if stage == Stage.TRAIN:
            CONSOLE.log("Starting training (fit)...")
            trainer.fit(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
        elif stage == Stage.VAL:
            CONSOLE.log("Starting validation...")
            trainer.validate(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
        elif stage == Stage.TEST:
            CONSOLE.log("Starting testing...")
            trainer.test(
                model=model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
        else:
            CONSOLE.error(f"Unsupported stage: {stage}")
            raise ValueError(f"Unsupported stage: {stage}")

        # # Determine checkpoint path
        # ckpt_path_to_load = self.ckpt_path
        # if ckpt_path_to_load is None:
        #     CONSOLE.log("Searching for the latest checkpoint...")
        #     ckpt_path_to_load = trainer_factory.find_latest_checkpoint()
        #     if ckpt_path_to_load:
        #         CONSOLE.log(f"Found checkpoint to resume/load: {ckpt_path_to_load}")
        #     else:
        #         CONSOLE.log("No checkpoint found, starting training from scratch.")

        # # Run the specified stage
        # if stage == Stage.TRAIN:
        #     CONSOLE.log("Starting training (fit)...")
        #     trainer.fit(
        #         model=model,
        #         datamodule=datamodule,
        #         ckpt_path=ckpt_path_to_load,
        #     )
        # elif stage == Stage.TEST:
        #     CONSOLE.log("Starting testing...")
        #     if ckpt_path_to_load is None:
        #         CONSOLE.warn(
        #             "Testing requires a checkpoint. Please provide ckpt_path or ensure a checkpoint exists."
        #         )
        #     trainer.test(
        #         model=model, datamodule=datamodule, ckpt_path=ckpt_path_to_load
        #     )
        # elif stage == Stage.VAL:
        #     CONSOLE.log("Starting validation...")
        #     if ckpt_path_to_load is None:
        #         CONSOLE.warn(
        #             "Validation requires a checkpoint. Please provide ckpt_path or ensure a checkpoint exists."
        #         )
        #     trainer.validate(
        #         model=model, datamodule=datamodule, ckpt_path=ckpt_path_to_load
        #     )
        # else:
        #     CONSOLE.error(f"Unsupported stage: {stage}")
        #     raise ValueError(f"Unsupported stage: {stage}")

        # CONSOLE.log(f"Stage '{stage}' finished.")
        # return trainer
