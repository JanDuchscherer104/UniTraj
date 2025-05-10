from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pydantic import Field
from torch import Tensor

# --- Import BaseConfig and Console ---
from ..utils.base_config import BaseConfig, Console

# Placeholder for actual loss function import
def placeholder_loss_fn(pred, target):
    return torch.tensor(0.0, requires_grad=True)

# --- Optimizer and Scheduler Configs remain BaseModel ---
# They don't need setup_target or propagation logic directly
from pydantic import BaseModel

class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""
    lr: float = Field(1e-4, description="Learning rate.")
    weight_decay: float = Field(0.0, description="Weight decay.")
    # Add other optimizer-specific parameters if needed (e.g., betas for Adam)

class SchedulerConfig(BaseModel):
    """Configuration for the learning rate scheduler."""
    # Example: StepLR configuration
    use_scheduler: bool = Field(False, description="Whether to use a learning rate scheduler.")
    step_size: int = Field(10, description="Step size for StepLR scheduler.")
    gamma: float = Field(0.1, description="Gamma factor for StepLR scheduler.")
    # Add fields for other scheduler types and their parameters as needed

# --- Update BaseLitModuleConfig to inherit from BaseConfig ---
class BaseLitModuleConfig(BaseConfig["BaseLitModule"]):
    """
    Base configuration for LightningModules.

    Acts as a factory for creating the BaseLitModule instance.
    Inherits from BaseConfig for setup_target and potential propagation.
    """
    target: Type["BaseLitModule"] = Field(default_factory=lambda: BaseLitModule)
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler_config: SchedulerConfig = Field(default_factory=SchedulerConfig)
    # Add model-specific hyperparameters here, e.g.:
    # feature_dim: int = Field(128, description="Dimension of internal features.")

    # setup_target is inherited from BaseConfig

# --- Update BaseLitModule ---
class BaseLitModule(pl.LightningModule):
    """
    Base class for PyTorch Lightning modules in this project.

    Implements basic training, validation, and test loops,
    and optimizer/scheduler configuration. Subclasses should override
    `forward`, `compute_loss`, and potentially the step methods
    for model-specific logic.
    """
    def __init__(self, config: BaseLitModuleConfig):
        """
        Initializes the BaseLitModule.

        Args:
            config (BaseLitModuleConfig): The configuration object.
        """
        super().__init__()
        # Use save_hyperparameters for Lightning's tracking mechanisms
        # config.model_dump() converts Pydantic model to dict
        self.save_hyperparameters(config.model_dump())
        self.config = config # Keep the pydantic config for easy access

        # --- Instantiate Console ---
        self.CONSOLE = Console.with_prefix(self.__class__.__name__)
        self.CONSOLE.set_debug(getattr(config, 'is_debug', False)) # Set debug based on config if available

        # Example: Define parts of the model - subclasses should do this
        # self.model = torch.nn.Linear(config.feature_dim, 1) # Placeholder

    def forward(self, batch: Dict[str, Any]) -> Any:
        """
        Forward pass of the model. Subclasses must implement this.

        Args:
            batch (Dict[str, Any]): The input batch dictionary, typically from the DataLoader.

        Returns:
            Any: The model's output.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def compute_loss(self, prediction: Any, batch: Dict[str, Any]) -> Tensor:
        """
        Computes the loss for a given prediction and batch.
        Subclasses should implement their specific loss calculation here.

        Args:
            prediction (Any): The output from the forward pass.
            batch (Dict[str, Any]): The input batch dictionary.

        Returns:
            Tensor: The computed loss value.
        """
        # Example placeholder - subclasses need to implement this
        # target = batch['target'] # Assuming target is in the batch
        # loss = placeholder_loss_fn(prediction, target)
        # return loss
        raise NotImplementedError("Subclasses must implement the compute_loss method.")

    def _common_step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> Tensor:
        """
        Common logic for training, validation, and test steps.

        Args:
            batch (Dict[str, Any]): The input batch.
            batch_idx (int): The index of the batch.
            stage (str): The current stage ('train', 'val', 'test').

        Returns:
            Tensor: The computed loss for the batch.
        """
        # Example: Assumes forward and compute_loss are implemented by subclass
        try:
            # The input batch might be nested, e.g., inside 'input_dict'
            # Handle potential nesting as seen in BaseDataset collate_fn
            input_data = batch.get('input_dict', batch)
            if not isinstance(input_data, dict):
                 self.CONSOLE.warn(f"Input data for {stage} step is not a dict, using raw batch.")
                 input_data = batch # Fallback if not nested as expected

            prediction = self.forward(input_data)
            loss = self.compute_loss(prediction, input_data)

            # Log the loss, prepending the stage
            # Use batch.get('batch_size') from the original collated batch
            batch_size = batch.get('batch_size', 1) # Get batch size safely
            self.log(f"{stage}/loss", loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

            # Log other metrics if needed
            # self.log(f"{stage}/some_metric", metric_value, ...)
            return loss
        except NotImplementedError as e:
            # Use CONSOLE for error logging
            self.CONSOLE.error(f"Error in _common_step ({stage}): {e}. Ensure forward() and compute_loss() are implemented.")
            # Return a dummy loss or re-raise depending on desired behavior
            return torch.tensor(0.0, device=self.device, requires_grad=True) # Dummy loss
        except Exception as e:
            # Use CONSOLE for error logging
            self.CONSOLE.error(f"Unexpected error during {stage} step for batch {batch_idx}: {e}")
            # Handle other potential errors gracefully
            return torch.tensor(0.0, device=self.device, requires_grad=True) # Dummy loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """
        Performs a single training step.

        Args:
            batch (Dict[str, Any]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The loss for the training step.
        """
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            batch (Dict[str, Any]): The input batch.
            batch_idx (int): The index of the batch.
        """
        self._common_step(batch, batch_idx, "val") # Loss is logged internally

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Performs a single test step.

        Args:
            batch (Dict[str, Any]): The input batch.
            batch_idx (int): The index of the batch.
        """
        self._common_step(batch, batch_idx, "test") # Loss is logged internally

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: Configuration dictionary for the optimizer and scheduler.
        """
        opt_cfg = self.config.optimizer_config
        sched_cfg = self.config.scheduler_config
        self.CONSOLE.log(f"Configuring optimizer: AdamW(lr={opt_cfg.lr}, wd={opt_cfg.weight_decay})")

        # Filter parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = optim.AdamW(
            trainable_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay
        )

        optimizer_config = {"optimizer": optimizer}

        if sched_cfg.use_scheduler:
            self.CONSOLE.log(f"Configuring scheduler: StepLR(step_size={sched_cfg.step_size}, gamma={sched_cfg.gamma})")
            # Example using StepLR, adapt as needed for other schedulers
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma
            )
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch", # or "step"
                "frequency": 1,
                "monitor": "val/loss", # Optional: Monitor a metric for ReduceLROnPlateau
            }
        else:
             self.CONSOLE.log("No learning rate scheduler configured.")

        return optimizer_config
