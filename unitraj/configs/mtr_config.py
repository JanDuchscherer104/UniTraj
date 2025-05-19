"""
Configuration file for the MTR (Motion Transformer) model
"""
from pathlib import Path
from typing import List, Optional

from pydantic import Field

from unitraj.configs.experiment_config import ExperimentConfig
from unitraj.configs.path_config import PathConfig
from unitraj.lightning.lit_datamodule import LitDatamoduleConfig
from unitraj.lightning.lit_trainer_factory import LitTrainerFactoryConfig
from unitraj.datasets.base_dataset import DatasetConfig
from unitraj.models.mtr.MTR_config import MTRConfig as BaseMTRConfig


class MTRConfig(ExperimentConfig):
    """
    Configuration for MTR, extending the base ExperimentConfig
    """

    # Model specific configuration
    model: BaseMTRConfig = Field(
        default_factory=lambda: BaseMTRConfig(
            model_name="MTR",
            # Context Encoder settings - both lowercase and uppercase for compatibility
            context_encoder=dict(
                name="MTREncoder",
                num_of_attn_neighbors=7,
                num_input_attr_agent=39,
                num_input_attr_map=29,
                num_channel_in_mlp_agent=256,
                num_channel_in_mlp_map=64,
                num_layer_in_mlp_agent=3,
                num_layer_in_mlp_map=5,
                num_layer_in_pre_mlp_map=3,
                d_model=256,
                num_attn_layers=6,
                num_attn_head=8,
                dropout_of_attn=0.1,
                use_local_attn=True,
            ),
            # Define uppercase versions for direct model access
            CONTEXT_ENCODER=dict(
                NAME="MTREncoder",
                NUM_OF_ATTN_NEIGHBORS=7,
                NUM_INPUT_ATTR_AGENT=39,
                NUM_INPUT_ATTR_MAP=29,
                NUM_CHANNEL_IN_MLP_AGENT=256,
                NUM_CHANNEL_IN_MLP_MAP=64,
                NUM_LAYER_IN_MLP_AGENT=3,
                NUM_LAYER_IN_MLP_MAP=5,
                NUM_LAYER_IN_PRE_MLP_MAP=3,
                D_MODEL=256,
                NUM_ATTN_LAYERS=6,
                NUM_ATTN_HEAD=8,
                DROPOUT_OF_ATTN=0.1,
                USE_LOCAL_ATTN=True,
            ),
            # Motion Decoder settings
            motion_decoder=dict(
                name="MTRDecoder",
                num_motion_modes=6,
                intention_points_file="models/mtr/cluster_64_center_dict_6s.pkl",
                d_model=512,
                num_decoder_layers=6,
                num_attn_head=8,
                map_d_model=256,
                dropout_of_attn=0.1,
                num_base_map_polylines=256,
                num_waypoint_map_polylines=128,
                loss_weights={
                    'cls': 1.0,
                    'reg': 1.0,
                    'vel': 0.5
                },
                nms_dist_thresh=2.5,
            ),
            # Define uppercase versions for direct model access
            MOTION_DECODER=dict(
                NAME="MTRDecoder",
                NUM_MOTION_MODES=6,
                INTENTION_POINTS_FILE="models/mtr/cluster_64_center_dict_6s.pkl",
                D_MODEL=512,
                NUM_DECODER_LAYERS=6,
                NUM_ATTN_HEAD=8,
                MAP_D_MODEL=256,
                DROPOUT_OF_ATTN=0.1,
                NUM_BASE_MAP_POLYLINES=256,
                NUM_WAYPOINT_MAP_POLYLINES=128,
                LOSS_WEIGHTS={
                    'cls': 1.0,
                    'reg': 1.0,
                    'vel': 0.5
                },
                NMS_DIST_THRESH=2.5,
            ),
            # Training settings
            max_epochs=40,
            learning_rate=0.0001,
            learning_rate_sched=[22, 24, 26, 28],
            optimizer="AdamW",
            scheduler="lambdaLR",
            grad_clip_norm=1000.0,
            weight_decay=0.01,
            lr_decay=0.5,
            lr_clip=0.000001,
            train_batch_size=32,  # Adjust based on GPU memory
            eval_batch_size=32,   # Adjust based on GPU memory
            # Data settings
            max_num_agents=64,
            map_range=100,
            max_num_roads=768,
            max_points_per_lane=20,
            manually_split_lane=True,
            point_sampled_interval=1,
            num_points_each_polyline=20,
            vector_break_dist_thresh=1.0,
            # Other parameters
            # Filter to only handle vehicles since intention points only exist for vehicles
            object_type=['VEHICLE'],
            center_offset_of_map=[30.0, 0.0],
            future_len=60,
        )
    )

    # DataModule configuration
    datamodule: LitDatamoduleConfig = Field(
        default_factory=lambda: LitDatamoduleConfig(
            dataset_config=DatasetConfig(
                max_num_samples=None,  # Use None for all data
                cache_dir="./cache",
                use_cache=True,
            ),
            num_workers=31,  # Use maximum recommended workers
            pin_memory=True,  # Enable pin memory for faster data transfer to GPU
        )
    )

    # Trainer configuration
    trainer: LitTrainerFactoryConfig = Field(
        default_factory=lambda: LitTrainerFactoryConfig(
            max_epochs=40, 
            accelerator="gpu",
            devices=[0],  # Specify GPU ids
            precision=16,  # Use mixed precision for faster training
            log_every_n_steps=10,
        )
    )

    # Path configuration
    paths: PathConfig = Field(default_factory=PathConfig)
