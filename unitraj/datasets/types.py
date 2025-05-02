from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
from metadrive.scenario.scenario_description import MetaDriveType
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor, int64

# -----------------------------------------------------------------------------
# Shape symbol key used throughout docstrings
#   T – number of timesteps in raw data
#   P – past trajectory length (past_len)
#   F – future trajectory length (future_len)
#   S – total trajectory steps (S = P + F)
#   N – number of objects / agents in a scenario
#   A – maximum number of agents kept per sample (max_num_agents)
#   R – maximum number of road polylines kept (max_roads)
#   L – number of points per polyline segment (num_points_each_polyline)
#   C – coordinate dimension (2 = x,y or 3 = x,y,z)
#   D – generic feature dimension
#   B – batch size / number of “center” agents drawn from a scenario
# -----------------------------------------------------------------------------

# Type alias for object ID (usually string)
ObjectID = str


# --- Raw Scenario Data (Input to preprocess) ---
class RawStateDict(TypedDict):
    """Represents the state of an object at multiple timesteps."""

    position: np.ndarray
    """[T, 3] (x, y, z)"""
    length: np.ndarray
    """[T, 1]"""
    width: np.ndarray
    """[T, 1]"""
    height: np.ndarray
    """[T, 1]"""
    heading: np.ndarray
    """[T, 1]"""
    velocity: np.ndarray
    """[T, 2] (vx, vy)"""
    valid: np.ndarray
    """[T, 1] (bool)"""
    # Potentially other fields like acceleration


class RawTrackDict(TypedDict):
    """Represents a single track in the raw scenario."""

    state: RawStateDict
    """State information for the track over time."""
    type: str
    """Object type string, e.g., 'VEHICLE', 'PEDESTRIAN'."""


class RawMapFeatureDict(TypedDict):
    """Represents a single map feature in the raw scenario."""

    polyline: np.ndarray
    """[N, C] polyline coordinates for this feature (N = # points)."""
    type: str
    """Map feature type string, e.g., 'LANE', 'ROAD_LINE'."""


class RawDynamicState(TypedDict, total=False):
    """Represents the dynamic state of a map element (e.g., traffic light)."""

    state: Any
    """Structure depends on the dataset (e.g., int for TL state)."""
    stop_point: Optional[np.ndarray]
    """[3,] (x, y, z) Optional stop point associated with the state."""


class RawScenarioDict(TypedDict):
    """Represents the raw data structure for a single scenario."""

    scenario_id: str
    """Unique identifier for the scenario."""
    timestamps_seconds: np.ndarray
    """[T] Timestamps for each step (seconds)."""
    tracks: Dict[ObjectID, RawTrackDict]
    """Dictionary mapping object IDs to their track data."""
    map_features: Dict[str, RawMapFeatureDict]
    """Dictionary mapping map feature IDs to their data."""
    dynamic_map_states: Optional[Dict[int, Dict[str, RawDynamicState]]]
    """Optional dynamic states keyed by timestamp index, then lane_id."""
    metadata: Dict[str, Any]
    """Additional metadata, including sdc_id, potentially map_center, etc."""
    # Optional fields depending on dataset format
    tracks_to_predict: Optional[Dict[ObjectID, Any]]
    """Optional dictionary indicating tracks to predict (e.g., Waymo format)."""
    objects_of_interest: Optional[List[ObjectID]]
    """Optional list of object IDs considered 'of interest' (e.g., Argoverse format)."""
    dataset: Optional[str]
    """Name of the dataset this scenario belongs to (added during loading)."""


# --- Intermediate Internal Format (Output of preprocess, Input to process) ---


class TrackInfosDict(TypedDict):
    """Processed information about all valid tracks in the scenario."""

    object_id: List[ObjectID]
    """List of object IDs for all tracks."""
    object_type: List[int]
    """List of object types (mapped to integers) corresponding to object_id."""
    trajs: np.ndarray
    """[N, S, 10] Trajectories for all objects (features as listed)."""


class MapInfoEntry(TypedDict):
    """Entry for a specific map polyline type."""

    id: str
    """Unique ID of the map feature."""
    type: str
    """Original type string of the map feature."""
    polyline_index: Tuple[int, int]
    """Start and end index in the 'all_polylines' array for this feature."""


class MapInfosDict(TypedDict):
    """Processed information about map features."""

    lane: List[MapInfoEntry]
    """List of lane features."""
    road_line: List[MapInfoEntry]
    """List of road line features."""
    road_edge: List[MapInfoEntry]
    """List of road edge features."""
    stop_sign: List[MapInfoEntry]
    """List of stop sign features."""
    crosswalk: List[MapInfoEntry]
    """List of crosswalk features."""
    speed_bump: List[MapInfoEntry]
    """List of speed bump features."""
    all_polylines: np.ndarray
    """[L, 7] Stacked polyline points for the entire map."""


class DynamicMapInfosDict(TypedDict):
    """Processed dynamic map information (potentially simplified)."""

    lane_id: List[str]
    """List of lane IDs associated with dynamic states."""
    state: List[Any]
    """List of processed dynamic states."""
    stop_point: List[np.ndarray]
    """List of stop points associated with the dynamic states."""


class TracksToPredictDict(TypedDict):
    """Information about which tracks should be predicted."""

    track_index: List[int]
    """Indices within TrackInfosDict['trajs'] corresponding to tracks to predict."""
    object_type: List[int]
    """Object types (integer) for the tracks to predict."""


class InternalFormatDict(TypedDict):
    """Intermediate data structure after preprocessing a scenario."""

    track_infos: TrackInfosDict
    """Processed track information."""
    dynamic_map_infos: DynamicMapInfosDict
    """Processed dynamic map information."""
    map_infos: MapInfosDict
    """Processed static map information."""
    scenario_id: str
    """Unique identifier for the scenario."""
    timestamps_seconds: np.ndarray
    """[total_steps,] Timestamps for each step in seconds."""
    current_time_index: int
    """Index separating past and future steps in the trajectories."""
    sdc_track_index: int
    """Index of the Self-Driving Car (SDC) within TrackInfosDict['trajs'."""
    tracks_to_predict: TracksToPredictDict
    """Information about which tracks need prediction."""
    map_center: np.ndarray
    """[1, 3] (x, y, z) Center coordinates of the map area."""
    track_length: int
    """Total number of steps (past + future) in the trajectories."""
    dataset: str
    """Name of the source dataset."""


# --- Processed Data (Output of process, Input to postprocess/getitem) ---
class ProcessedDataDict(TypedDict):
    """Structure holding processed data for multiple agents (batch-like before splitting)."""

    scenario_id: np.ndarray
    """[B] Scenario ID for each center agent."""
    obj_trajs: np.ndarray
    """[B, A, P, D] Past trajectories of all objects relative to each center agent."""
    obj_trajs_mask: np.ndarray
    """[B, A, P] Mask for past trajectories."""
    track_index_to_predict: np.ndarray
    """[B] Index of the center agent within A."""
    obj_trajs_pos: np.ndarray
    """[B, A, P, 3] Past positions (x, y, z)."""
    obj_trajs_last_pos: np.ndarray
    """[B, A, 3] Last valid past position for each object."""
    center_objects_world: np.ndarray
    """[B, 10] World‑frame state of each center agent at t=0."""
    center_objects_id: np.ndarray
    """[B] Object ID of each center agent."""
    center_objects_type: np.ndarray
    """[B] Object type enum of each center agent."""
    map_center: np.ndarray
    """[B, 1, 3] Map center used for normalization."""
    obj_trajs_future_state: np.ndarray
    """[B, A, F, 4] Future states (x, y, vx, vy)."""
    obj_trajs_future_mask: np.ndarray
    """[B, A, F] Mask for future states."""
    center_gt_trajs: np.ndarray
    """[B, F, 4] Ground‑truth future trajectory of center agents."""
    center_gt_trajs_mask: np.ndarray
    """[B, F] Mask for ground‑truth future points."""
    center_gt_final_valid_idx: np.ndarray
    """[B] Index of last valid GT point."""
    center_gt_trajs_src: np.ndarray
    """[B, S, 10] Original world‑frame trajectory of center agents."""
    map_polylines: np.ndarray
    """[B, R, L, 29] Map polyline features relative to each center agent."""
    map_polylines_mask: np.ndarray
    """[B, R, L] Mask for map polylines."""
    map_polylines_center: np.ndarray
    """[B, R, 3] Center of each polyline segment."""
    dataset_name: List[str]
    """[B] Name of the source dataset."""


class DatasetItem(BaseModel):
    """
    Strongly-typed data model for a single sample.
    """

    @field_validator("scenario_id", "center_objects_id", "dataset_name", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        # Handle numpy scalar/array
        if isinstance(v, (np.generic, np.ndarray)):
            try:
                v = v.item()
            except Exception:
                pass
        # Decode bytes to str
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8")
        # If already str, leave unchanged
        if isinstance(v, str):
            return v
        return v

    scenario_id: str = Field(..., description="Scenario unique ID (|S36)")

    obj_trajs: np.ndarray = Field(
        ..., description="Past trajectories: shape (A, P, 39)"
    )
    obj_trajs_mask: np.ndarray = Field(
        ..., description="Mask for past traj points: shape (A, P)"
    )
    track_index_to_predict: np.int64 = Field(
        ..., description="Index of agent-to-predict in [0,A)"
    )

    obj_trajs_pos: np.ndarray = Field(
        ..., description="Past positions (x,y,z): shape (A, P, 3)"
    )
    obj_trajs_last_pos: np.ndarray = Field(
        ..., description="Last valid pos per agent: shape (A, 3)"
    )

    center_objects_world: np.ndarray = Field(
        ..., description="World-frame state of predicted agent: shape (10,)"
    )
    center_objects_id: str = Field(..., description="Agent ID (|S4)")
    center_objects_type: np.int64 = Field(..., description="Agent type enum")

    map_center: np.ndarray = Field(
        ..., description="Map center for normalization: shape (3,)"
    )

    obj_trajs_future_state: np.ndarray = Field(
        ..., description="Future states (x,y,vx,vy): shape (A, F, 4)"
    )
    obj_trajs_future_mask: np.ndarray = Field(
        ..., description="Mask for future traj points: shape (A, F)"
    )

    center_gt_trajs: np.ndarray = Field(
        ..., description="GT future traj (x,y,vx,vy): shape (F, 4)"
    )
    center_gt_trajs_mask: np.ndarray = Field(
        ..., description="Mask for GT future points: shape (F,)"
    )
    center_gt_final_valid_idx: float = Field(
        ..., description="Index of last valid GT point"
    )
    center_gt_trajs_src: np.ndarray = Field(
        ..., description="Original world traj: shape (S, 10)"
    )

    map_polylines: np.ndarray = Field(
        ..., description="Map polyline feats: shape (R, L, 29)"
    )
    map_polylines_mask: np.ndarray = Field(
        ..., description="Mask for map polyline points: shape (R, L)"
    )
    map_polylines_center: np.ndarray = Field(
        ..., description="Center (x,y,z) per polyline: shape (R, 3)"
    )

    dataset_name: str = Field(..., description="Source dataset name (|S38)")

    # Added by postprocess
    kalman_difficulty: Optional[np.ndarray] = Field(
        None, description="Optional Kalman difficulty: shape (K,)"
    )
    trajectory_type: Optional[int] = Field(
        None, description="Optional trajectory type enum"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def to_tensor_dict(self) -> Dict[str, Any]:
        """
        Convert all numpy arrays to torch.Tensors.
        """
        import torch

        tdict = {}
        for name, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                tdict[name] = torch.from_numpy(val)
            else:
                tdict[name] = val
        return tdict

    def summary(self) -> str:
        """
        Returns a one-line summary: scenario_id, dataset_name, #agents, #past, #future, #map_polylines.
        """
        # decode bytes fields for readability
        sid = (
            self.scenario_id.decode("utf-8")
            if isinstance(self.scenario_id, (bytes, bytearray))
            else str(self.scenario_id)
        )
        ds = (
            self.dataset_name.decode("utf-8")
            if isinstance(self.dataset_name, (bytes, bytearray))
            else str(self.dataset_name)
        )
        n_agents, n_past, _ = self.obj_trajs.shape
        n_future = self.center_gt_trajs.shape[0]
        # number of static map polylines
        map_count = self.map_polylines.shape[0]
        return (
            f"<DatasetItem {sid!r} @ {ds!r}: "
            f"agents={n_agents}, past={n_past}, future={n_future}, "
            f"map_polylines={map_count}>"
        )


# --- Batch Data (Output of collate_fn) ---
class BatchInputDict(TypedDict):
    """Structure of the input_dict within BatchDict (usually tensors)."""

    scenario_id: List[str]
    """List of scenario IDs in the batch."""
    obj_trajs: Tensor
    """[batch_size, max_num_agents, past_len, num_features] Batched past trajectories."""
    obj_trajs_mask: Tensor
    """[batch_size, max_num_agents, past_len] Batched mask for past trajectories."""
    track_index_to_predict: Tensor
    """[batch_size,] Batched index of the agent-to-predict."""
    obj_trajs_pos: Tensor
    """[batch_size, max_num_agents, past_len, 3] Batched past positions."""
    obj_trajs_last_pos: Tensor
    """[batch_size, max_num_agents, 3] Batched last valid past positions."""
    center_objects_world: Tensor
    """[batch_size, 10] Batched world-frame state of the agent-to-predict."""
    center_objects_id: List[ObjectID]
    """List of object IDs for the agents-to-predict in the batch."""
    center_objects_type: Tensor
    """[batch_size,] Batched object type of the agent-to-predict."""
    map_center: Tensor
    """[batch_size, 1, 3] Batched map centers."""
    obj_trajs_future_state: Tensor
    """[batch_size, max_num_agents, future_len, 4] Batched future states."""
    obj_trajs_future_mask: Tensor
    """[batch_size, max_num_agents, future_len] Batched mask for future states."""
    center_gt_trajs: Tensor
    """[batch_size, future_len, 4] Batched ground truth future trajectories."""
    center_gt_trajs_mask: Tensor
    """[batch_size, future_len] Batched mask for ground truth future trajectories."""
    center_gt_final_valid_idx: Tensor
    """[batch_size,] Batched index of the last valid ground truth future point."""
    center_gt_trajs_src: Tensor
    """[batch_size, total_steps, 10] Batched original world-frame trajectories."""
    map_polylines: Tensor
    """[batch_size, max_roads, max_points, map_feat_dim] Batched map polyline features."""
    map_polylines_mask: Tensor
    """[batch_size, max_roads, max_points] Batched mask for map polyline points."""
    map_polylines_center: Tensor
    """[batch_size, max_roads, 3] Batched center points for map polylines."""
    dataset_name: List[str]
    """List of dataset names in the batch."""
    kalman_difficulty: Tensor
    """[batch_size,] Batched Kalman difficulty scores."""
    trajectory_type: Tensor
    """[batch_size,] Batched trajectory type classifications."""


class BatchDict(TypedDict):
    """Structure returned by the collate_fn."""

    batch_size: int
    """Number of samples in the batch."""
    input_dict: BatchInputDict
    """Dictionary containing the batched input tensors and metadata."""
    batch_sample_count: int
    """Often the same as batch_size, represents the number of samples processed."""


class Stage(Enum):
    """
    (TRAIN, VAL, TEST) = ("train", "val", "test")
    """

    TRAIN = ("train", "fit")
    VAL = ("val", "validate")
    TEST = ("test",)

    def __str__(self):
        return self.value[0]

    @classmethod
    def from_str(cls, value: Optional[str]) -> Optional["Stage"]:
        for member in cls:
            if value in member.value:
                return member
        return None


traffic_light_state_to_int = {
    None: 0,
    MetaDriveType.LANE_STATE_UNKNOWN: 0,
    # // States for traffic signals with arrows.
    MetaDriveType.LANE_STATE_ARROW_STOP: 1,
    MetaDriveType.LANE_STATE_ARROW_CAUTION: 2,
    MetaDriveType.LANE_STATE_ARROW_GO: 3,
    # // Standard round traffic signals.
    MetaDriveType.LANE_STATE_STOP: 4,
    MetaDriveType.LANE_STATE_CAUTION: 5,
    MetaDriveType.LANE_STATE_GO: 6,
    # // Flashing light signals.
    MetaDriveType.LANE_STATE_FLASHING_STOP: 7,
    MetaDriveType.LANE_STATE_FLASHING_CAUTION: 8,
}


class ObjectType(Enum):
    """
    Mapping from MetaDriveType to integer labels for objects.
    """

    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    @classmethod
    def from_metadrive_type(cls, md_type: Any) -> "ObjectType":
        """
        Convert a MetaDriveType value or string to an ObjectType enum.
        """
        # md_type may be a string constant or a MetaDriveType member.
        name = str(md_type).upper()
        return cls.__members__.get(name, cls.UNSET)


class PolylineType(Enum):
    """
    Mapping from MetaDriveType (and some strings) to integer labels for map polylines.
    """

    UNSET = 0

    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    LINE_BROKEN_SINGLE_WHITE = 6
    LINE_SOLID_SINGLE_WHITE = 7
    LINE_SOLID_DOUBLE_WHITE = 8
    LINE_BROKEN_SINGLE_YELLOW = 9
    LINE_BROKEN_DOUBLE_YELLOW = 10
    LINE_SOLID_SINGLE_YELLOW = 11
    LINE_SOLID_DOUBLE_YELLOW = 12
    LINE_PASSING_DOUBLE_YELLOW = 13

    BOUNDARY_LINE = 15
    BOUNDARY_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19

    @classmethod
    def from_metadrive_type(cls, md_type: Any) -> "PolylineType":
        """
        Convert a MetaDriveType value or string to a PolylineType enum.
        """
        # md_type may be a string constant or a MetaDriveType member.
        name = str(md_type).upper()
        # Handle known aliases
        alias = {
            "LANE_SURFACE_UNSTRUCTURE": "LANE_SURFACE_STREET",
            "ROAD_EDGE_SIDEWALK": "LINE_SOLID_SINGLE_WHITE",
        }
        name = alias.get(name, name)
        return cls.__members__.get(name, cls.UNSET)
