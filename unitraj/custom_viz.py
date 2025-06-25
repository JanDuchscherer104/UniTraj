#!/usr/bin/env python3
"""
Script for visualizing samples and predictions using the UniTraj framework
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Get the absolute path to the unitraj directory
current_dir = Path(__file__).parent.absolute()
unitraj_dir = current_dir
project_dir = unitraj_dir.parent  # UniTraj directory

# Add paths to system path to ensure imports work properly
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
if str(unitraj_dir) not in sys.path:
    sys.path.insert(0, str(unitraj_dir))

# These imports should now work regardless of where the script is run from
from unitraj.configs.mtr_config import MTRConfig
from unitraj.utils.console import Console
from unitraj.utils.visualization import check_loaded_data, visualize_prediction, configure_plotting, save_high_res_figure, plot_dataset_item, plot_prediction_item

def main():
    console = Console().log
    
    # Create config
    cfg = MTRConfig(
        is_debug=False,
        # Specify checkpoint path for a trained model
        ckpt_path="default_exp/last-v13.ckpt",
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_dir, ".logs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup trainer, model, and datamodule
    console(f"Setting up model from checkpoint: {cfg.ckpt_path}")
    # Use setup_target_and_run to initialize all components properly
    trainer, model_instance, datamodule = cfg.setup_target()
    
    # Instead, use the model's class to load from checkpoint
    # Get the model class from the instance
    model_class = model_instance.__class__
    
    # Load model from checkpoint if checkpoint path is specified
    if cfg.ckpt_path:
        # Correctly call the class method on the class, not the instance
        model = model_class.load_from_checkpoint(cfg.ckpt_path)
        console(f"Model loaded from checkpoint: {cfg.ckpt_path}")
    else:
        # If no checkpoint, use the instance
        model = model_instance
    
    # Check if CUDA is available and move model to appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    console(f"Model moved to device: {device}")
    
    # Load a few samples
    console("Loading validation data...")
    datamodule.setup(stage="validate")
    val_dataloader = datamodule.val_dataloader()
    batch = next(iter(val_dataloader))
    
    # Move batch to the same device as the model
    # Add this recursive function to move nested tensors to device
    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(o, device) for o in obj]
        elif isinstance(obj, tuple):
            return tuple(move_to_device(o, device) for o in obj)
        else:
            return obj
    
    # Replace your current batch device moving code with this:
    batch = move_to_device(batch, device)
    
    configure_plotting()
    
    # 1. Visualize raw data
    console("Creating data visualization...")
    sample = datamodule.val_dataset[0]
    fig = check_loaded_data(sample)
    data_viz_path = os.path.join(output_dir, "data_visualization.png")
    save_high_res_figure(fig, data_viz_path)
    console(f"Raw data visualization saved to {data_viz_path}")
    
    # 2. Get predictions and visualize them
    console("Creating prediction visualization...")
    with torch.no_grad():
        model.eval()
        prediction, _ = model(batch)
    
    fig = visualize_prediction(batch, prediction)
    pred_viz_path = os.path.join(output_dir, "prediction_visualization.png")
    save_high_res_figure(fig, pred_viz_path)
    console(f"Prediction visualization saved to {pred_viz_path}")
    
    # 3. Use plot_dataset_item for enhanced visualization of dataset items
    console("Creating enhanced dataset visualization...")
    sample = datamodule.val_dataset[0]
    # Include more agents in the visualization and show bounding boxes
    fig = plot_dataset_item(
        data=sample,
        include_agents_in_range=True,
        show_bboxes=True,
        show_timestep_info=True
    )
    enhanced_data_viz_path = os.path.join(output_dir, "enhanced_data_visualization.png")
    save_high_res_figure(fig, enhanced_data_viz_path)
    console(f"Enhanced dataset visualization saved to {enhanced_data_viz_path}")
    
    # 4. Use plot_prediction_item for enhanced visualization with predictions
    console("Creating enhanced prediction visualization...")
    # Since plot_prediction_item works with dataset items and not batches,
    # we need to extract the first item from validation dataset
    sample = datamodule.val_dataset[0]
    
    # Run model on individual sample
    # We need to prepare the sample for the model first
    sample_tensor = datamodule.collate_fn([sample])
    sample_tensor = move_to_device(sample_tensor, device)
    
    # Get prediction for this sample
    with torch.no_grad():
        model.eval()
        sample_prediction, _ = model(sample_tensor)

    # from pprint import pprint
    # pprint(sample_prediction)

    # Create enhanced visualization with predictions
    fig = plot_prediction_item(
        data=sample,
        prediction=sample_prediction,
        include_agents_in_range=True,
        show_bboxes=False,
        include_all_predictions=True,
        show_ids=False,
        show_probabilities=False
        # prediction_top_k=5,
        # show_probabilities=True,
        # show_timestep_info=True,
        # prediction_colormap="rainbow"
    )
    enhanced_pred_viz_path = os.path.join(output_dir, "enhanced_prediction_visualization.png")
    save_high_res_figure(fig, enhanced_pred_viz_path)
    console(f"Enhanced prediction visualization saved to {enhanced_pred_viz_path}")
    
    console("Visualizations completed successfully!")

if __name__ == "__main__":
    main()