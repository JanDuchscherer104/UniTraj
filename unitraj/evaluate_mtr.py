#!/usr/bin/env python3
"""
Script for evaluating a trained MTR model using the new configuration system
"""
import os
from pathlib import Path

from configs.mtr_config import MTRConfig
from utils.console import Console

if __name__ == "__main__":
    console = Console().log

    # Create an instance of the MTR config
    cfg = MTRConfig(
        is_debug=False,
        
        # Specify checkpoint path for a trained model
        ckpt_path="default_exp/last-v13.ckpt",
    )

    # Enable visualization
    cfg.model.visualize_eval_predictions = True

    # Print configuration summary
    console("Evaluation Configuration:")
    console(f"Model: {cfg.model.model_name}")
    console(f"Checkpoint: {cfg.ckpt_path}")
    console(f"Evaluation batch size: {cfg.model.eval_batch_size}")
    
    # Evaluate the model
    console("Starting evaluation...")
    # This will set up the trainer, model, and datamodule, and run evaluation
    cfg.setup_target_and_run("val")
    
    console("Evaluation completed!")
