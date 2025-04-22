from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
from metadrive.scenario.scenario_description import MetaDriveType

# Type alias for object ID (usually string)
ObjectID = str


# --- Raw Scenario Data (Input to preprocess) ---
class RawStateDict(TypedDict):
    """Represents the state of an object at multiple timesteps."""

    position: np.ndarray
    """[num_timesteps, 3] (x, y, z)"""
    length: np.ndarray
    """[num_timesteps, 1]"""
    width: np.ndarray
    """[num_timesteps, 1]"""
    height: np.ndarray
    """[num_timesteps, 1]"""
    heading: np.ndarray
    """[num_timesteps, 1]"""
    velocity: np.ndarray
    """[num_timesteps, 2] (vx, vy)"""
    valid: np.ndarray
    """[num_timesteps, 1] (boolean/int)"""
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
    """[num_points, 2 or 3] (x, y, [z]) representing the feature geometry."""
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
    """[num_timesteps,] Timestamps for each step in seconds."""
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
    """[num_objects, total_steps, 10] Array containing trajectory data (x,y,z,l,w,h,heading,vx,vy,valid)."""


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
    """[total_points, 7] Array containing all map polyline points (x,y,z, dx,dy,dz, type_int)."""


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
    """[num_agents_in_scenario,] Scenario ID for each agent instance."""
    obj_trajs: np.ndarray
    """[num_agents_in_scenario, max_num_agents, past_len, num_features] Past trajectories of all objects relative to each agent."""
    obj_trajs_mask: np.ndarray
    """[num_agents_in_scenario, max_num_agents, past_len] Boolean mask for valid past trajectory points."""
    track_index_to_predict: np.ndarray
    """[num_agents_in_scenario,] Index of the agent-to-predict within the max_num_agents dimension."""
    obj_trajs_pos: np.ndarray
    """[num_agents_in_scenario, max_num_agents, past_len, 3] Past positions (x, y, z) of all objects."""
    obj_trajs_last_pos: np.ndarray
    """[num_agents_in_scenario, max_num_agents, 3] Last valid position (x, y, z) for each object in the past."""
    center_objects_world: np.ndarray
    """[num_agents_in_scenario, 10] World-frame state of the agent-to-predict at the current time step."""
    center_objects_id: np.ndarray
    """[num_agents_in_scenario,] Object ID of the agent-to-predict."""
    center_objects_type: np.ndarray
    """[num_agents_in_scenario,] Object type (int) of the agent-to-predict."""
    map_center: np.ndarray
    """[num_agents_in_scenario, 1, 3] Map center used for normalization for each agent."""
    obj_trajs_future_state: np.ndarray
    """[num_agents_in_scenario, max_num_agents, future_len, 4] Future states (x, y, vx, vy) of all objects."""
    obj_trajs_future_mask: np.ndarray
    """[num_agents_in_scenario, max_num_agents, future_len] Boolean mask for valid future trajectory points."""
    center_gt_trajs: np.ndarray
    """[num_agents_in_scenario, future_len, 4] Ground truth future trajectory (x, y, vx, vy) for the agent-to-predict."""
    center_gt_trajs_mask: np.ndarray
    """[num_agents_in_scenario, future_len] Boolean mask for valid ground truth future points."""
    center_gt_final_valid_idx: np.ndarray
    """[num_agents_in_scenario,] Index of the last valid point in the ground truth future trajectory."""
    center_gt_trajs_src: np.ndarray
    """[num_agents_in_scenario, total_steps, 10] Original world-frame trajectory of the agent-to-predict."""
    map_polylines: np.ndarray
    """[num_agents_in_scenario, max_roads, max_points, map_feat_dim] Map polyline features relative to each agent."""
    map_polylines_mask: np.ndarray
    """[num_agents_in_scenario, max_roads, max_points] Boolean mask for valid map polyline points."""
    map_polylines_center: np.ndarray
    """[num_agents_in_scenario, max_roads, 3] Center point (x, y, z) for each map polyline."""
    dataset_name: List[str]
    """[num_agents_in_scenario,] Name of the source dataset for each agent."""


class DatasetItem(TypedDict):
    """Structure for a single data sample returned by __getitem__."""

    scenario_id: np.bytes_
    """Scenario unique ID. shape: (), dtype: np.bytes_ (|S36)"""

    obj_trajs: np.ndarray
    """Past trajectories of surrounding agents. shape: (128, 21, 39), dtype: float32"""

    obj_trajs_mask: np.ndarray
    """Mask for valid past trajectory points. shape: (128, 21), dtype: bool"""

    track_index_to_predict: np.int64
    """Index of the agent-to-predict within the max_num_agents dimension. shape: (), dtype: int64"""

    obj_trajs_pos: np.ndarray
    """Past positions (x, y, z) of surrounding agents. shape: (128, 21, 3), dtype: float32"""

    obj_trajs_last_pos: np.ndarray
    """Last valid position (x, y, z) for each surrounding agent. shape: (128, 3), dtype: float32"""

    center_objects_world: np.ndarray
    """World-frame state of the agent-to-predict at the current time step. shape: (10,), dtype: float32"""

    center_objects_id: np.bytes_
    """Object ID of the agent-to-predict. shape: (), dtype: np.bytes_ (|S4)"""

    center_objects_type: np.int64
    """Object type (int) of the agent-to-predict. shape: (), dtype: int64"""

    map_center: np.ndarray
    """Map center used for normalization. shape: (3,), dtype: float32"""

    obj_trajs_future_state: np.ndarray
    """Future states (x, y, vx, vy) of surrounding agents. shape: (128, 60, 4), dtype: float32"""

    obj_trajs_future_mask: np.ndarray
    """Mask for valid future trajectory points. shape: (128, 60), dtype: float32"""

    center_gt_trajs: np.ndarray
    """Ground truth future trajectory (x, y, vx, vy) for the agent-to-predict. shape: (60, 4), dtype: float32"""

    center_gt_trajs_mask: np.ndarray
    """Mask for valid ground truth future points. shape: (60,), dtype: float32"""

    center_gt_final_valid_idx: np.float32
    """Index of the last valid point in the ground truth future trajectory. shape: (), dtype: float32"""

    center_gt_trajs_src: np.ndarray
    """Original world-frame trajectory of the agent-to-predict. shape: (81, 10), dtype: float32"""

    map_polylines: np.ndarray
    """Map polyline features. shape: (256, 20, 29), dtype: float32"""

    map_polylines_mask: np.ndarray
    """Mask for valid map polyline points. shape: (256, 20), dtype: bool"""

    map_polylines_center: np.ndarray
    """Center point (x, y, z) for each map polyline. shape: (256, 3), dtype: float32"""

    dataset_name: np.bytes_
    """Name of the source dataset. shape: (), dtype: np.bytes_ (|S38)"""

    # Added by postprocess
    kalman_difficulty: Optional[np.ndarray]
    """Optional Kalman filter based difficulty score. shape: (3,), dtype: float64"""

    trajectory_type: Optional[np.int64]
    """Optional classification of the trajectory type. shape: (), dtype: int64"""


# --- Batch Data (Output of collate_fn) ---

# Using Any for Tensor type to avoid torch dependency here
Tensor = Any


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


object_type = {
    MetaDriveType.UNSET: 0,
    MetaDriveType.VEHICLE: 1,
    MetaDriveType.PEDESTRIAN: 2,
    MetaDriveType.CYCLIST: 3,
    MetaDriveType.OTHER: 4,
}

# lane_type = {
#     0: MetaDriveType.LANE_UNKNOWN,
#     1: MetaDriveType.LANE_FREEWAY,
#     2: MetaDriveType.LANE_SURFACE_STREET,
#     3: MetaDriveType.LANE_BIKE_LANE
# }
#
# road_line_type = {
#     0: MetaDriveType.LINE_UNKNOWN,
#     1: MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
#     2: MetaDriveType.LINE_SOLID_SINGLE_WHITE,
#     3: MetaDriveType.LINE_SOLID_DOUBLE_WHITE,
#     4: MetaDriveType.LINE_BROKEN_SINGLE_YELLOW,
#     5: MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW,
#     6: MetaDriveType.LINE_SOLID_SINGLE_YELLOW,
#     7: MetaDriveType.LINE_SOLID_DOUBLE_YELLOW,
#     8: MetaDriveType.LINE_PASSING_DOUBLE_YELLOW
# }
#
# road_edge_type = {
#     0: MetaDriveType.LINE_UNKNOWN,
#     # // Physical road boundary that doesn't have traffic on the other side (e.g.,
#     # // a curb or the k-rail on the right side of a freeway).
#     1: MetaDriveType.BOUNDARY_LINE,
#     # // Physical road boundary that separates the car from other traffic
#     # // (e.g. a k-rail or an island).
#     2: MetaDriveType.BOUNDARY_MEDIAN
# }

polyline_type = {
    # for lane
    MetaDriveType.LANE_FREEWAY: 1,
    MetaDriveType.LANE_SURFACE_STREET: 2,
    "LANE_SURFACE_UNSTRUCTURE": 2,
    MetaDriveType.LANE_BIKE_LANE: 3,
    # for roadline
    MetaDriveType.LINE_BROKEN_SINGLE_WHITE: 6,
    MetaDriveType.LINE_SOLID_SINGLE_WHITE: 7,
    "ROAD_EDGE_SIDEWALK": 7,
    MetaDriveType.LINE_SOLID_DOUBLE_WHITE: 8,
    MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: 9,
    MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW: 10,
    MetaDriveType.LINE_SOLID_SINGLE_YELLOW: 11,
    MetaDriveType.LINE_SOLID_DOUBLE_YELLOW: 12,
    MetaDriveType.LINE_PASSING_DOUBLE_YELLOW: 13,
    # for roadedge
    MetaDriveType.BOUNDARY_LINE: 15,
    MetaDriveType.BOUNDARY_MEDIAN: 16,
    # for stopsign
    MetaDriveType.STOP_SIGN: 17,
    # for crosswalk
    MetaDriveType.CROSSWALK: 18,
    # for speed bump
    MetaDriveType.SPEED_BUMP: 19,
}

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
