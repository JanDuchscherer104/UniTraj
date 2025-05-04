# filepath: /home/duchscherer/repos/Video-Analysis/src/external/UniTraj/unitraj/train.py
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# Or import specific model configs if build_model is removed/changed
# from .models.autobot import AutoBotConfig
# from .models.wayformer import WayformerConfig # Example
from unitraj.utils.utils import (
    set_seed,
)  # Assuming this is a utility function for setting seeds

from .configs.path_config import PathConfig  # Import PathConfig

# Import your Pydantic Config classes
from .datasets.base_dataset import DatasetConfig, Stage
from .datasets.types import BatchDict  # Assuming BatchDict is defined here or elsewhere
from .models import build_model  # Keep if build_model handles Pydantic config


# Assuming build_dataset is potentially replaced by DatasetConfig.setup_target()
# from .datasets import build_dataset # Remove this if using setup_target directly
@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):  # cfg is OmegaConf DictConfig
    # Optional: Log the resolved config from Hydra
    print("--- Hydra Resolved Config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")

    # --- Convert Hydra config to Pydantic configs ---

    # Path Config (Assuming it's nested under 'paths' or top-level)
    # If PathConfig is Singleton, instantiation might need care, but from_hydra should work per run.
    path_config = PathConfig.from_hydra(cfg.paths if "paths" in cfg else cfg)

    # Dataset Config (Assuming it's under 'dataset')
    # Pass the validated path_config instance
    dataset_config = DatasetConfig.from_hydra(cfg.dataset)
    dataset_config.paths = path_config  # Assign the validated PathConfig instance
    # Propagate top-level flags if needed and defined in DatasetConfig
    if hasattr(dataset_config, "is_debug"):
        dataset_config.is_debug = cfg.debug

    # Model Config (Assuming it's under 'model' or 'method')
    # Determine the correct Pydantic Model Config class based on Hydra config
    # Option 1: Use a known key like 'model_name'
    model_cfg_hydra = (
        cfg.model if "model" in cfg else cfg.method
    )  # Adjust based on your structure
    model_name = model_cfg_hydra.get("model_name", "unknown")

    # You'll need to import the specific Pydantic config classes
    if model_name == "autobot":
        from .models.autobot import AutoBotConfig as ModelConfigClass
    elif model_name == "wayformer":
        from .models.wayformer import (
            WayformerConfig as ModelConfigClass,
        )  # Make sure this exists
    else:
        # Attempt to dynamically load based on target if specified, otherwise raise error
        target_str = model_cfg_hydra.get("_target_", None)  # Hydra often adds _target_
        if target_str:
            # Assuming target string points to the *model class*, infer config from it (needs convention)
            # This is more complex, sticking to model_name is easier
            raise NotImplementedError(
                f"Dynamic config class loading based on target '{target_str}' not implemented."
            )
        else:
            raise ValueError(
                f"Unknown or missing model_name: '{model_name}' in config section {cfg.model if 'model' in cfg else cfg.method}"
            )

    model_config = ModelConfigClass.from_hydra(model_cfg_hydra)
    # Propagate top-level flags if needed
    if hasattr(model_config, "is_debug"):
        model_config.is_debug = cfg.debug

    # --- Use Pydantic configs and factory pattern ---
    set_seed(cfg.seed)  # Use top-level simple values directly from Hydra cfg

    # Instantiate dataset using Pydantic config's factory method
    # Set stage for training set
    dataset_config.stage = Stage.TRAIN
    train_set = dataset_config.setup_target()  # Calls BaseDataset(dataset_config)

    # Create and configure validation set config
    # Create a new instance or deep copy to avoid modifying train config
    val_dataset_config = DatasetConfig.from_hydra(
        cfg.dataset
    )  # Re-validate from Hydra source
    val_dataset_config.paths = path_config  # Re-assign paths
    val_dataset_config.stage = Stage.VAL  # Set stage to VAL
    if hasattr(val_dataset_config, "is_debug"):
        val_dataset_config.is_debug = cfg.debug
    val_set = val_dataset_config.setup_target()  # Calls BaseDataset(val_dataset_config)

    # Instantiate model using Pydantic config's factory method
    # The build_model function might become redundant if model_config.setup_target() does the job
    # model = build_model(model_config) # If build_model now accepts Pydantic config
    model = model_config.setup_target()  # Directly use the factory pattern

    # --- Setup DataLoaders and Trainer ---
    # Use values from the validated Pydantic config objects
    train_batch_size = max(model_config.train_batch_size // len(cfg.devices), 1)
    eval_batch_size = max(model_config.eval_batch_size // len(cfg.devices), 1)

    call_backs = []
    checkpoint_callback = ModelCheckpoint(
        monitor="val/brier_fde",
        filename="{epoch}-{val/brier_fde:.2f}",
        save_top_k=1,
        mode="min",
        # Use path_config for directory structure
        dirpath=str(path_config.checkpoints / cfg.exp_name),  # Ensure path is string
    )
    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        num_workers=cfg.load_num_workers,  # Use directly from Hydra cfg
        drop_last=False,
        collate_fn=train_set.collate_fn,  # Use method from instantiated dataset
        shuffle=True,  # Usually shuffle training data
    )

    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        num_workers=cfg.load_num_workers,  # Use directly from Hydra cfg
        shuffle=False,
        drop_last=False,
        collate_fn=val_set.collate_fn,  # Use method from instantiated dataset
    )

    trainer = pl.Trainer(
        max_epochs=model_config.max_epochs,  # Use Pydantic config
        logger=(
            None
            if cfg.debug
            # Use path_config for potential save dirs if needed by logger
            else WandbLogger(
                project="unitraj",
                name=cfg.exp_name,
                id=cfg.exp_name,
                save_dir=str(path_config.root / ".logs"),
            )  # Example save_dir
        ),
        devices=1 if cfg.debug else cfg.devices,  # Use directly from Hydra cfg
        gradient_clip_val=model_config.grad_clip_norm,  # Use Pydantic config
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        # strategy="auto" if cfg.debug else "ddp", # ddp might need find_unused_parameters=True depending on model
        strategy=(
            "ddp_find_unused_parameters_true"
            if not cfg.debug and len(cfg.devices) > 1
            else "auto"
        ),  # More robust DDP default
        callbacks=call_backs,
    )

    # --- Resume and Train ---
    ckpt_path_to_resume = cfg.ckpt_path  # Use directly from Hydra cfg

    # automatically resume training (logic seems incomplete in original, adjust as needed)
    # Example: Check if a checkpoint exists in the dirpath
    last_ckpt = None
    if not cfg.debug and checkpoint_callback.dirpath:
        # Logic to find the latest checkpoint in checkpoint_callback.dirpath
        # For example:
        ckpt_dir = Path(checkpoint_callback.dirpath)
        if ckpt_dir.exists():
            ckpts = sorted(
                ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if ckpts:
                last_ckpt = str(ckpts[0])
                print(f"Found last checkpoint: {last_ckpt}")

    # Prioritize explicit ckpt_path over automatic last_ckpt
    ckpt_to_use = ckpt_path_to_resume if ckpt_path_to_resume else last_ckpt

    print(f"Starting training for experiment: {cfg.exp_name}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_to_use)


if __name__ == "__main__":
    train()
