#!/usr/bin/env python3
"""
Script for training the MTR model using the new configuration system
"""
import os
import sys
from pathlib import Path

# Import UniTraj modules with proper path handling
from unitraj.configs.mtr_config import MTRConfig
from unitraj.utils.console import Console

def main():
    console = Console().log

    # Create an instance of the MTR config with default values
    cfg = MTRConfig(
        is_debug=False
        )
    
    # Limit samples for faster training (if needed)
    cfg.datamodule.dataset_config.max_num_samples = 1024
    
    # Adjust based on your GPU memory
    cfg.model.train_batch_size = 32
    
    # Print configuration summary
    console("MTR Training Configuration:")
    console(f"Model: {cfg.model.model_name}")
    console(f"Training batch size: {cfg.model.train_batch_size}")
    console(f"Number of epochs: {cfg.model.max_epochs}")
    console(f"Learning rate: {cfg.model.learning_rate}")
    console(f"Object types: {cfg.model.object_type}")
    console(f"Future trajectory length: {cfg.model.future_len} frames")
    
    # Train the model
    console("Starting training...")
    # This will set up the trainer, model, and datamodule, and run training
    cfg.setup_target_and_run("train")
    
    console("Training completed!")

if __name__ == "__main__":
    main()
