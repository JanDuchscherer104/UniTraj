"""
Configuration for the MTR (Motion Transformer) model
"""
from typing import Dict, List, Optional, Type, Union, Any, TYPE_CHECKING
from pydantic import Field

from ..base_model.base_model import BaseModelConfig

# Import directly to avoid the need for model_rebuild()
from .MTR import MotionTransformer


class MTRConfig(BaseModelConfig["MotionTransformer"]):
    """
    Configuration for the MTR model
    """
    target: Type["MotionTransformer"] = Field(default_factory=lambda: MotionTransformer)
    model_name: str = Field(default="MTR")
    
    # Context Encoder configuration - using uppercase to match MTR expected format
    CONTEXT_ENCODER: Dict = Field(default_factory=dict)
    
    # Motion Decoder configuration - using uppercase to match MTR expected format
    MOTION_DECODER: Dict = Field(default_factory=dict)
    
    # Alias properties to maintain compatibility with both naming styles
    @property
    def context_encoder(self) -> Dict:
        """Alias for CONTEXT_ENCODER to maintain compatibility with existing code"""
        return self.CONTEXT_ENCODER
        
    @context_encoder.setter
    def context_encoder(self, value: Dict):
        self.CONTEXT_ENCODER = value
        
    @property
    def motion_decoder(self) -> Dict:
        """Alias for MOTION_DECODER to maintain compatibility with existing code"""
        return self.MOTION_DECODER
        
    @motion_decoder.setter
    def motion_decoder(self, value: Dict):
        self.MOTION_DECODER = value
    
    # Training parameters
    max_epochs: int = Field(default=40)
    learning_rate: float = Field(default=0.0001)
    learning_rate_sched: List[int] = Field(default_factory=lambda: [22, 24, 26, 28])
    optimizer: str = Field(default="AdamW")
    scheduler: str = Field(default="lambdaLR")
    grad_clip_norm: float = Field(default=1000.0)
    weight_decay: float = Field(default=0.01)
    lr_decay: float = Field(default=0.5)
    lr_clip: float = Field(default=0.000001)
    train_batch_size: int = Field(default=32)
    eval_batch_size: int = Field(default=32)
    
    # Data related parameters
    max_num_agents: int = Field(default=64)
    map_range: int = Field(default=100)
    max_num_roads: int = Field(default=768)
    max_points_per_lane: int = Field(default=20)
    manually_split_lane: bool = Field(default=True)
    point_sampled_interval: int = Field(default=1)
    num_points_each_polyline: int = Field(default=20)
    vector_break_dist_thresh: float = Field(default=1.0)
    
    # Trajectory parameters
    object_type: List[str] = Field(default_factory=lambda: ['VEHICLE'])
    center_offset_of_map: List[float] = Field(default_factory=lambda: [30.0, 0.0])
    future_len: int = Field(default=60)  # 6 seconds with 10Hz
    
    # Evaluation parameters
    eval_nuscenes: bool = Field(default=False)  # Flag for NuScenes evaluation

    def setup_target(self):
        """
        Sets up and returns the MTR model
        """
        from .MTR import MotionTransformer
        
        # Create a simple dictionary with the required structure
        # Focusing on getting a working configuration first
        config = {
            # Direct model parameters
            "model_name": self.model_name,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "learning_rate_sched": self.learning_rate_sched,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "grad_clip_norm": self.grad_clip_norm,
            "weight_decay": self.weight_decay,
            "lr_decay": self.lr_decay,
            "lr_clip": self.lr_clip,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "max_num_agents": self.max_num_agents,
            "map_range": self.map_range,
            "max_num_roads": self.max_num_roads,
            "max_points_per_lane": self.max_points_per_lane,
            "manually_split_lane": self.manually_split_lane,
            "point_sampled_interval": self.point_sampled_interval,
            "num_points_each_polyline": self.num_points_each_polyline,
            "vector_break_dist_thresh": self.vector_break_dist_thresh,
            "object_type": self.object_type,
            "center_offset_of_map": self.center_offset_of_map,
            "future_len": self.future_len,
            "eval_nuscenes": self.eval_nuscenes,
            
            # Context Encoder and Motion Decoder in uppercase as expected by MTR.py
            "CONTEXT_ENCODER": self.CONTEXT_ENCODER,
            "MOTION_DECODER": self.MOTION_DECODER
        }
        
        # Create the model with the simplified config dict
        model = self.target(config)
        return model
