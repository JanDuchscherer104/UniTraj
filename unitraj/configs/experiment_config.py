from typing import Any, Dict, List, Optional, Tuple, Union

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
    seed: int = Field(42, description="Random seed for reproducibility")
    is_debug: bool = Field(
        True, description="Debug mode (affects logging, checkpoints, etc.)"
    )

    # evaluation settings
    eval_nuscenes: bool = Field(False, description="Use nuScenes evaluation tool")
    eval_waymo: bool = Field(False, description="Use Waymo evaluation tool")
    eval_argoverse2: bool = Field(False, description="Use Argoverse2 evaluation tool")

    # Use new Lightning configs
    method: AutoBotConfig = Field(
        default_factory=AutoBotConfig,
        description="Lightning Module configuration (specific model config).",
    )
    datamodule: LitDatamoduleConfig = Field(
        default_factory=LitDatamoduleConfig,
    )
    trainer: LitTrainerFactoryConfig = Field(
        default_factory=LitTrainerFactoryConfig,
        description="Lightning Trainer factory configuration.",
    )

    # Keep PathConfig
    paths: PathConfig = Field(
        default_factory=PathConfig, description="Path configuration"
    )

    # Optional: Checkpoint path override
    ckpt_path: Optional[str] = Field(
        None, description="Explicit path to a checkpoint to load."
    )

    @model_validator(mode="before")
    @classmethod
    def propagate_is_debug(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        is_debug = data.get("is_debug", False)

        if is_debug:
            # Ensure 'datamodule' is a dict and set 'is_debug'
            datamodule = data.get("datamodule", {})
            if isinstance(datamodule, dict):
                datamodule.setdefault("is_debug", True)
                data["datamodule"] = datamodule

            # Ensure 'trainer' is a dict and set 'is_debug'
            trainer = data.get("trainer", {})
            if isinstance(trainer, dict):
                trainer.setdefault("is_debug", True)
                data["trainer"] = trainer

        return data

    def setup_target(
        self, stage: Union[Stage, str] = Stage.TRAIN
    ) -> Tuple[pl.Trainer, BaseModel, LitDatamodule]:
        """
        Sets up and runs the training/evaluation process using PyTorch Lightning.

        Args:
            stage (str): The stage to run ('fit', 'test', 'validate', 'predict'). Defaults to 'fit'.
        """
        stage = stage if isinstance(stage, Stage) else Stage.from_str(stage)
        assert isinstance(stage, Stage)
        set_seed(self.seed)

        CONSOLE = Console.with_prefix(
            self.__class__.__name__, "setup_target"
        ).set_debug(self.is_debug)

        # Instantiate components
        CONSOLE.log("Instantiating lightning components...")
        datamodule = self.datamodule.setup_target()
        model = self.method.setup_target()
        trainer_factory = self.trainer.setup_target()

        trainer: pl.Trainer = trainer_factory.create_trainer()

        return trainer, model, datamodule

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
