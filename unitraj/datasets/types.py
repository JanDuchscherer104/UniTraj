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
#   Tp – past trajectory length (past_len) (prev P)
#   Tf – future trajectory length (future_len) (prev F)
#   T – total trajectory steps (S = Tp + Tf) (prev S)
#   N – number of active agents in a scenario
#   A - number of center agents (whose future we want to predict) (prev B)
#   Nmax – maximum number of agents kept per sample (max_num_agents) (prev A)
#   K – maximum number of road polylines kept (max_roads)
#   L – number of points per polyline segment (num_points_each_polyline)
#   C – coordinate dimension (2 = x,y or 3 = x,y,z)
#   Fap – feature dimension for all agents in historical data.
#   Faf - feature dimension for all agents in future data.
#   Fmap - feature dimension for map polylines
#   B – batch size
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
    """[A] Scenario ID for each center agent.

    Each entry uniquely identifies the scenario from which the corresponding center agent is drawn. This allows for tracking and grouping predictions by scenario during evaluation and debugging.
    """
    obj_trajs: np.ndarray
    """[A, Nmax, Tp, Fap] Past trajectories of all objects relative to each center agent.

    Captures the historical movement data for all agents in a scene, for each center agent in the batch. These trajectories are crucial for learning spatiotemporal patterns, which underpin the prediction of future behaviors. Each trajectory is normalized relative to the center agent at the current timestep, ensuring consistency across samples. The feature dimension Fap may include positions, velocities, headings, etc.
    """
    obj_trajs_mask: np.ndarray
    """[A, Nmax, Tp] Mask for past trajectories.

    Boolean mask indicating the validity of each past trajectory entry. Used to ignore padded or missing data during training and evaluation.
    """
    track_index_to_predict: np.ndarray
    """[A] Index of the center agent within Nmax.

    For each sample in the batch, this gives the index of the "center" agent (the agent for which prediction is made) within the set of Nmax agents considered for that sample. This enables models to distinguish the target agent from its context and is essential for supervised learning and evaluation.
    """
    obj_trajs_pos: np.ndarray
    """[A, Nmax, Tp, C] Past positions (x, y, z).

    The raw spatial locations of all agents for each timestep in the past window, relative to the center agent.
    """
    obj_trajs_last_pos: np.ndarray
    """[A, Nmax, C] Last valid past position for each object.

    For each agent, provides the most recent valid position in the past window, often used for initialization or as a reference point for future prediction.
    """
    center_objects_world: np.ndarray
    """[A, 10] World-frame state of each center agent at t=0.

    The complete state vector (positions, velocities, heading, etc.) of the center agent at the current time in global/world coordinates. This serves as a reference for denormalization and for model input features.
    """
    center_objects_id: np.ndarray
    """[A] Object ID of each center agent.

    Unique identifier for the center agent in each sample, useful for mapping predictions back to the original data.
    """
    center_objects_type: np.ndarray
    """[A] Object type enum of each center agent.

    Integer code representing the semantic type of the center agent (e.g., vehicle, pedestrian, cyclist).
    """
    map_center: np.ndarray
    """[A, 1, C] Map center used for normalization.

    The central point of the map region for each sample, used to normalize coordinates and ensure translation invariance. Normalizing by the map center helps models generalize across different map locations and scales.
    """
    obj_trajs_future_state: np.ndarray
    """[A, Nmax, Tf, Faf] Future states (x, y, vx, vy).

    Ground-truth or reference future states for all agents, including position and velocity, relative to the center agent's frame.
    """
    obj_trajs_future_mask: np.ndarray
    """[A, Nmax, Tf] Mask for future states.

    Boolean mask indicating valid future trajectory entries for each agent.
    """
    center_gt_trajs: np.ndarray
    """[A, Tf, Faf] Ground-truth future trajectory of center agents.

    The true future trajectory (position and velocity) for each center agent in the batch. This is the primary supervision target for trajectory prediction models.
    """
    center_gt_trajs_mask: np.ndarray
    """[A, Tf] Mask for ground-truth future points.

    Boolean mask indicating which points in the ground-truth future trajectory are valid and should be used for loss computation.
    """
    center_gt_final_valid_idx: np.ndarray
    """[A] Index of last valid GT point.

    For each center agent, gives the index of the last valid point in its future trajectory. Useful for variable-length trajectory handling.
    """
    center_gt_trajs_src: np.ndarray
    """[A, T, 10] Original world-frame trajectory of center agents.

    The full trajectory (past + future) for each center agent in world coordinates, including all available features (positions, velocities, etc.). Useful for evaluation and visualization.
    """
    map_polylines: np.ndarray
    """[A, K, L, Fmap] Map polyline features relative to each center agent.

    Contains the geometric and semantic features of the map (e.g., lane boundaries, crosswalks) for each sample, centered and normalized to the center agent. Each polyline is represented by L points, and K is the maximum number of polylines considered. The feature dimension (Fmap) may include coordinates, heading, type encoding, and other attributes.
    """
    map_polylines_mask: np.ndarray
    """[A, K, L] Mask for map polylines.

    Boolean mask indicating valid points within each map polyline segment.
    """
    map_polylines_center: np.ndarray
    """[A, K, C] Center of each polyline segment.

    For each map polyline segment, provides the centroid or reference point in normalized coordinates. Useful for spatial reasoning and pooling operations.
    """
    dataset_name: List[str]
    """[A] Name of the source dataset.

    Indicates the origin dataset for each sample in the batch. This is important for multi-dataset training or cross-dataset evaluation.
    """


class DatasetItem(BaseModel):
    """
    Strongly-typed data model for a single sample.
    """

    @field_validator("scenario_id", "center_objects_id", "dataset_name", mode="before")
    @classmethod
    def _validate_str_fields(cls, v):
        if isinstance(v, (np.generic, np.ndarray)):
            try:
                v = v.item()
            except Exception:
                pass
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8")
        if isinstance(v, str):
            return v
        return v

    @field_validator("track_index_to_predict", "center_objects_type", mode="before")
    @classmethod
    def _validate_int64_fields(cls, v):
        """Convert NumPy arrays and other numeric types to np.int64."""
        if isinstance(v, np.ndarray):
            try:
                return np.int64(v.item())
            except Exception:
                pass
        if isinstance(v, (int, float, np.number)):
            return np.int64(v)
        return v

    scenario_id: str = Field(...)
    """Scenario unique ID (|S36)

    A unique string identifier for the scenario from which this sample is drawn. Used for grouping, tracking, and evaluation.
    """

    obj_trajs: np.ndarray = Field(...)
    """[Nmax, Tp, Fap] Past trajectories of all agents.

    Historical trajectory data for Nmax agents in the sample, for Tp past steps, and Fap features.
    These are normalized relative to the center agent, which is critical for learning scene-invariant motion patterns.
    The Fap dimension (e.g., 39) consists of:
    - [0:3] Relative Position: x, y, z relative to center agent at t=0.
    - [3:6] Size length, width, height of the object.
    - [6:11] Object Type & Role Mask (5 features): One-hot encoding (VEHICLE, PEDESTRIAN, CYCLIST, CENTER, SDC).
    - [11:11+Tp] One-hot Time Embedding; A := 11 + Tp
    - [A:A+2] Heading Embedding (2 features): Sine and cosine of the agent's heading relative to the center agent's heading at t=0.
    - [A+2:A+4] Relative Velocity (2 features): vx, vy relative to center agent frame.
    - [A+4:Fap] Relative Acceleration (2 features): ax, ay relative to center agent frame.
    """

    obj_trajs_mask: np.ndarray = Field(...)
    """[Nmax, Tp] Mask for past trajectory entries.

    Boolean mask indicating which entries in obj_trajs are valid (not padded or missing).
    """

    track_index_to_predict: np.int64 = Field(...)
    """Index of agent-to-predict in [0, Nmax). Typically 0.

    The integer index of the "center" age!nt (the prediction target) within the set of all Nmax agents in this sample.
    This index is crucial for extracting the correct ground-truth trajectory and for associating predictions with the
    correct agent.
    """

    obj_trajs_pos: np.ndarray = Field(...)
    """[Nmax, Tp, C] Past positions (x, y, z).

    The actual spatial positions for all agents and all past steps, with C=3 for 3D coordinates.
    """

    obj_trajs_last_pos: np.ndarray = Field(...)
    """[Nmax, C] Last valid position per agent.

    The most recent valid position in the past window for each agent, useful for initializing predictions or for
    computing relative motion.
    """

    center_objects_world: np.ndarray = Field(...)
    """[10] World-frame state of center agent at t=0.

    The full state vector (positions, velocities, heading, etc.) for the center agent at the current time, in
    absolute/world coordinates. This is essential for denormalizing predictions and for model input.

    (x, y, z, length, width, height, heading, vx, vy, valid)
    """

    center_objects_id: str = Field(...)
    """Center agent ID within the scenario.

    Unique identifier for the center agent in this sample, used for mapping predictions back to the original scenario data.
    """

    center_objects_type: np.int64 = Field(...)
    """Center agent type.

    Integer code for the semantic type (e.g., vehicle, pedestrian) of the center agent.
    0: UNSET, 1: VEHICLE, 2: PEDESTRIAN, 3: CYCLIST, 4: OTHER
    """

    map_center: np.ndarray = Field(...)
    """[C] Map center for normalization. Typically (0, 0, 0) (agent-centered).

    The reference point in map/world coordinates used to normalize all positions in this sample, ensuring translation invariance. This helps models generalize across locations.
    """

    obj_trajs_future_state: np.ndarray = Field(...)
    """[Nmax, Tf, Faf] Future states Faf = (x, y, vx, vy).

    The ground-truth or reference future states for all agents, including position and velocity, for Tf future steps.
    """

    obj_trajs_future_mask: np.ndarray = Field(...)
    """[Nmax, Tf] Mask for future trajectory entries.

    Boolean mask indicating which entries of obj_trajs_future_state are valid.
    """

    center_gt_trajs: np.ndarray = Field(...)
    """[Tf, Faf] GT future trajectory (x, y, vx, vy).

    The ground-truth future trajectory for the center agent, including position and velocity at each of Tf future timesteps. This is the main supervision signal for trajectory prediction.
    """

    center_gt_trajs_mask: np.ndarray = Field(...)
    """[Tf] Mask for GT future trajectory entries.

    Boolean mask indicating which future trajectory points for the center agent are valid (not padded).
    """

    center_gt_final_valid_idx: float = Field(...)
    """Index of last valid GT point.

    The (possibly fractional) index of the last valid ground-truth future point for the center agent. This is useful for handling variable-length future trajectories.
    """

    center_gt_trajs_src: np.ndarray = Field(...)
    """[T, 10] Original world-frame trajectory.

    The entire (past + future) trajectory of the center agent in world coordinates, including all 10 features. Used for visualization and evaluation.
    """

    map_polylines: np.ndarray = Field(...)
    """[K, L, Fmap] Map polyline features.

    Contains K polylines (e.g., lanes, crosswalks), each with L points, and Fmap features per point, all normalized
    relative to the center agent.

    The Fmap dimension consists of:
    - [0:3] Position (x, y, z): Spatial coordinates of the polyline point.
    - [3:6] Direction (x, y, z): Direction vector at the polyline point.
    - [6:9] Previous point position (x, y, z): Position of the previous point in the polyline.
    - [9:29] Lane type one-hot encoding: Represents different lane types as defined in PolylineType enum:
        - UNSET (0)
        - LANE_FREEWAY (1)
        - LANE_SURFACE_STREET (2)
        - LANE_BIKE_LANE (3)
        - LINE_BROKEN_SINGLE_WHITE (6)
        - LINE_SOLID_SINGLE_WHITE (7)
        - LINE_SOLID_DOUBLE_WHITE (8)
        - LINE_BROKEN_SINGLE_YELLOW (9)
        - LINE_BROKEN_DOUBLE_YELLOW (10)
        - LINE_SOLID_SINGLE_YELLOW (11)
        - LINE_SOLID_DOUBLE_YELLOW (12)
        - LINE_PASSING_DOUBLE_YELLOW (13)
        - BOUNDARY_LINE (15)
        - BOUNDARY_MEDIAN (16)
        - STOP_SIGN (17)
        - CROSSWALK (18)
        - SPEED_BUMP (19)
    """

    map_polylines_mask: np.ndarray = Field(...)
    """[K, L] Mask for map polylines.

    Boolean mask indicating which points in the map polylines are valid.
    """

    map_polylines_center: np.ndarray = Field(...)
    """[K, C] Center point per polyline.

    The centroid or reference point for each polyline segment, used for pooling and spatial reasoning.
    """

    dataset_name: str = Field(...)
    """Source dataset name (|S38)

    Indicates the name of the dataset from which this sample was drawn, which is important for multi-dataset experiments or analysis.
    """

    kalman_difficulty: Optional[np.ndarray] = Field(None)
    """[K] Optional Kalman difficulty scores.

    If available, provides a difficulty estimate for the sample, e.g., based on Kalman filter residuals.
    """

    trajectory_type: Optional[int] = Field(None)
    """Optional trajectory type enum.

    Integer code indicating a trajectory class (e.g., straight, turning, etc.), if provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def to_tensor_dict(self) -> Dict[str, Any]:
        import torch

        tdict = {}
        for name, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                tdict[name] = torch.from_numpy(val)
            else:
                tdict[name] = val
        return tdict

    def summary(self) -> str:
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
        n_agents_max, n_past, agent_past_feat_dim = self.obj_trajs.shape
        n_active_agents = np.sum(np.any(self.obj_trajs_last_pos != 0, axis=1)).item()  # type: ignore
        _, n_future, agent_future_feat_dim = self.obj_trajs_future_state.shape

        map_count, map_points, map_feat_dim = self.map_polylines.shape

        kd_info = (
            f", kd={self.kalman_difficulty.shape}"
            if self.kalman_difficulty is not None
            else ""
        )
        traj_type_info = (
            f", traj_type={self.trajectory_type}"
            if self.trajectory_type is not None
            else ""
        )

        return (
            f"<DatasetItem {sid!r} @ {ds!r}: "
            f"Agents={n_active_agents}/{n_agents_max}, "
            f"Traj(P={n_past}, F={n_future}, D_past={agent_past_feat_dim}, D_future={agent_future_feat_dim}), "
            f"Map(R={map_count}, L={map_points}, D_map={map_feat_dim})"
            f"{kd_info}{traj_type_info}>"
        )

    def print_shapes(self):
        """Prints the name and shape (for numpy arrays) or type (for others) of each field."""
        print(f"--- Shapes for DatasetItem ({self.scenario_id}) ---")
        for name, value in self.model_dump().items():
            if isinstance(value, np.ndarray):
                print(f"{name}: {value.shape}")
            else:
                print(f"{name}: {type(value)}")
        print("-------------------------------------------------")


# --- Batch Data (Output of collate_fn) ---
class BatchInputDict(TypedDict):
    """Structure of the input_dict within BatchDict (usually tensors)."""

    scenario_id: List[str]
    """[B] List of scenario IDs in the batch."""
    obj_trajs: Tensor
    """[B, Nmax, Tp, Fap] Batched past trajectories."""
    obj_trajs_mask: Tensor
    """[B, Nmax, Tp] Batched mask for past trajectories."""
    track_index_to_predict: Tensor
    """[B] Batched index of the agent-to-predict."""
    obj_trajs_pos: Tensor
    """[B, Nmax, Tp, C] Batched past positions."""
    obj_trajs_last_pos: Tensor
    """[B, Nmax, C] Batched last valid past positions."""
    center_objects_world: Tensor
    """[B, 10] Batched world-frame state of each center agent at t=0."""
    center_objects_id: List[ObjectID]
    """[B] List of center agent IDs."""
    center_objects_type: Tensor
    """[B] Batched object type enums."""
    map_center: Tensor
    """[B, 1, C] Batched map centers."""
    obj_trajs_future_state: Tensor
    """[B, Nmax, Tf, Faf] Batched future states (x, y, vx, vy)."""
    obj_trajs_future_mask: Tensor
    """[B, Nmax, Tf] Batched mask for future states."""
    center_gt_trajs: Tensor
    """[B, Tf, Faf] Batched ground truth future trajectories."""
    center_gt_trajs_mask: Tensor
    """[B, Tf] Batched mask for ground truth."""
    center_gt_final_valid_idx: Tensor
    """[B] Batched index of last valid ground truth point."""
    center_gt_trajs_src: Tensor
    """[B, T, 10] Batched original world-frame trajectories."""
    map_polylines: Tensor
    """[B, K, L, Fmap] Batched map polyline features.

    Contains K polylines (e.g., lanes, crosswalks), each with L points, and Fmap features per point, all normalized
    relative to the center agent.

    The Fmap dimension consists of:
    - [0:3] Position (x, y, z): Spatial coordinates of the polyline point.
    - [3:6] Direction (x, y, z): Direction vector at the polyline point.
    - [6:9] Previous point position (x, y, z): Position of the previous point in the polyline.
    - [9:29] Lane type one-hot encoding: Represents different lane types as defined in PolylineType enum:
        - UNSET (0)
        - LANE_FREEWAY (1)
        - LANE_SURFACE_STREET (2)
        - LANE_BIKE_LANE (3)
        - LINE_BROKEN_SINGLE_WHITE (6)
        - LINE_SOLID_SINGLE_WHITE (7)
        - LINE_SOLID_DOUBLE_WHITE (8)
        - LINE_BROKEN_SINGLE_YELLOW (9)
        - LINE_BROKEN_DOUBLE_YELLOW (10)
        - LINE_SOLID_SINGLE_YELLOW (11)
        - LINE_SOLID_DOUBLE_YELLOW (12)
        - LINE_PASSING_DOUBLE_YELLOW (13)
        - BOUNDARY_LINE (15)
        - BOUNDARY_MEDIAN (16)
        - STOP_SIGN (17)
        - CROSSWALK (18)
        - SPEED_BUMP (19)
    """
    map_polylines_mask: Tensor
    """[B, K, L] Batched polyline masks."""
    map_polylines_center: Tensor
    """[B, K, C] Batched polyline centers."""


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


TRAFFIC_LIGHT_STATE_TO_INT = {
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

TRAJECTORY_TYPE_MAP = {
    0: "stationary",
    1: "straight",
    2: "straight_right",
    3: "straight_left",
    4: "right_u_turn",
    5: "right_turn",
    6: "left_u_turn",
    7: "left_turn",
}
AGENT_TYPE_MAP = {0: "unset", 1: "vehicle", 2: "pedestrian", 3: "bicycle", 4: "other"}


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
