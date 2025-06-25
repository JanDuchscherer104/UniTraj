from typing import Any, Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
import matplotlib.transforms as mtransforms

from PIL import Image

from ..datasets.types import (
    AGENT_TYPE_MAP,
    TRAJECTORY_TYPE_MAP,
    BatchInputDict,
    DatasetItem,
    PolylineType,
)

# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements


def plot_dataset_item(
    data: DatasetItem,
    prediction: Optional[Dict[str, Any]] = None,  # Add prediction parameter
    agent_indices: Optional[List[int]] = None,
    show_legend: bool = True,
    show_map_legend: bool = True,
    vis_range: float = 50,
    include_agents_in_range: bool = False,
    show_timestep_info: bool = False,
    show_bboxes: bool = False,
    prediction_top_k: int = 3,  # Add parameter for top predictions
    show_probabilities: bool = True,  # Add parameter to show probabilities
    prediction_colormap: str = "rainbow",  # Add colormap parameter
) -> plt.Figure:
    """
    Improved static visualization for a single DatasetItem with optional predictions.

    Args:
        data: A single DatasetItem to visualize.
        prediction: Optional dictionary containing prediction outputs with keys:
            - 'predicted_probability': tensor of shape [N, M] where N is batch size and M is number of modes
            - 'predicted_trajectory': tensor of shape [N, M, T, 2+] where T is trajectory length
        agent_indices: List of agent-indices (0…Nmax-1) whose trajectories we plot.
            If None, defaults to the "ego" agent given by data.track_index_to_predict.
        show_legend: Whether to draw a legend for agent trajectories.
        show_map_legend: Whether to draw a legend for map features.
        vis_range: sets half-width/height of the visible region (in normalized coordinates).
        include_agents_in_range: Whether to automatically plot all agents whose trajectory passes within `vis_range`.
        show_timestep_info: Whether to annotate the number of past, future, and total timesteps on the plot.
        show_bboxes: If True, draw each object's 2D bounding box at the start of its trajectory instead of a start dot.
        prediction_top_k: Number of top probability predictions to include.
        show_probabilities: Whether to annotate prediction endpoints with their probabilities.
        prediction_colormap: Matplotlib colormap name to use for predictions.

    Returns:
        A matplotlib Figure with the plotted scene.
    """
    # 1) Determine which agents to plot
    past_xy = data.obj_trajs[..., :2]  # [Nmax, Tp, 2]
    fut_xy = data.obj_trajs_future_state[..., :2]  # [Nmax, Tf, 2]
    trajs = np.concatenate([past_xy, fut_xy], axis=1)  # [Nmax, Tp+Tf, 2]

    past_mask = data.obj_trajs_mask  # [Nmax, Tp]
    fut_mask = data.obj_trajs_future_mask  # [Nmax, Tf]
    masks = np.concatenate([past_mask, fut_mask], axis=1)  # [Nmax, Tp+Tf]

    Tp = past_xy.shape[1]
    T = trajs.shape[1]

    # Extract prediction data (simplified from plot_prediction_item)
    pred_probs = np.array([])
    pred_trajs = np.array([])
    if prediction is not None and isinstance(prediction, dict):
        if "predicted_probability" in prediction and "predicted_trajectory" in prediction:
            pred_probs = prediction["predicted_probability"].detach().cpu().numpy()
            pred_trajs = prediction["predicted_trajectory"].detach().cpu().numpy()
            
            # Handle batch dimension if present
            if len(pred_probs.shape) > 1 and pred_probs.shape[0] > 1:
                pred_probs = pred_probs[0]
                pred_trajs = pred_trajs[0]
            elif len(pred_probs.shape) > 1 and pred_probs.shape[0] == 1:
                pred_probs = pred_probs.squeeze(0)
                pred_trajs = pred_trajs.squeeze(0)
            
            # Remove last point if it causes issues
            if pred_trajs.shape[-2] > 1:
                pred_trajs = pred_trajs[:, :-1]
            
            # Select top predictions
            if pred_probs.shape[0] > prediction_top_k:
                top_indices = np.argsort(pred_probs)[-prediction_top_k:]
                pred_probs = pred_probs[top_indices]
                pred_trajs = pred_trajs[top_indices]
            
            # Extract only x, y coordinates
            if pred_trajs.shape[-1] > 2:
                pred_trajs = pred_trajs[..., :2]

    # Determine which agents to plot
    Nmax = trajs.shape[0]
    ego_idx = int(data.track_index_to_predict)
    if include_agents_in_range:
        agent_indices = []
        for i in range(Nmax):
            coords = trajs[i]  # [T,2]
            mask_i = masks[i]  # [T]
            cond_x = np.abs(coords[:, 0]) <= vis_range
            cond_y = np.abs(coords[:, 1]) <= vis_range
            in_box = np.logical_and(cond_x, cond_y)
            valid_mask = np.logical_and(mask_i.astype(bool), in_box)
            if np.any(valid_mask):
                agent_indices.append(i)
    elif agent_indices is None:
        agent_indices = [ego_idx]

    # 3) Decode agent semantic types (only VEHICLE/PEDESTRIAN/CYCLIST bits)
    sem_onehot = data.obj_trajs[:, -1, 6:9]  # [Nmax, 3]
    sem_idx = np.argmax(sem_onehot, axis=-1)  # 0=vehicle,1=pedestrian,2=cyclist
    # Map to AGENT_TYPE_MAP keys (1=vehicle,2=pedestrian,3=bicycle)
    agent_type_str = [AGENT_TYPE_MAP.get(int(idx) + 1, "unset") for idx in sem_idx]

    # 4) Set up color mapping for agents
    agent_color_map = {
        "unset": "gray",
        "vehicle": "tab:blue",
        "pedestrian": "tab:orange",
        "bicycle": "tab:green",
    }

    # 5) Prepare the figure
    fig, ax = plt.subplots(figsize=(10, 4))
    # Reserve right margin for legend; trim top/bottom margins to reduce whitespace
    fig.subplots_adjust(left=0.02, right=0.55, top=0.98, bottom=0.02)
    ax.set_aspect("equal")

    # 6) Plot map polylines (enhanced from plot_prediction_item)
    K, L, _ = data.map_polylines.shape
    map_xy = data.map_polylines[..., :2]  # [K, L, 2]
    map_mask = data.map_polylines_mask  # [K, L]
    map_type_oh = data.map_polylines[:, 0, 9:29]  # [K, 20]
    map_type_idx = np.argmax(map_type_oh, axis=-1)  # [K]

    map_legend_items = {}
    for k in range(K):
        coords = map_xy[k]
        valid = map_mask[k]
        pidx = int(map_type_idx[k])
        ptype = (
            PolylineType(pidx)
            if pidx in PolylineType._value2member_map_
            else PolylineType.UNSET
        )

        # choose style by feature
        if ptype.name.startswith("LANE_"):
            color, ls, label = "lightgray", "-", "Lane"
        elif ptype == PolylineType.CROSSWALK:
            color, ls, label = "black", "--", "Crosswalk"
        elif ptype == PolylineType.SPEED_BUMP:
            color, ls, label = "magenta", "-", "SpeedBump"
        elif ptype == PolylineType.STOP_SIGN:
            color, ls, label = "red", "-", "StopSign"
        else:
            color, ls, label = "dimgray", "-", ptype.name.title().replace("_", " ")

        # for the legend, draw one invisible line per new label
        if show_map_legend and label not in map_legend_items:
            map_legend_items[label] = ax.plot(
                [], [], color=color, ls=ls, lw=2, label=label
            )[0]

        # draw the actual polyline segments
        for i in range(L - 1):
            if valid[i] and valid[i + 1]:
                ax.plot(
                    [coords[i, 0], coords[i + 1, 0]],
                    [coords[i, 1], coords[i + 1, 1]],
                    color=color,
                    ls=ls,
                    lw=1,
                )

    # 7) Plot agent trajectories
    # Ensure ego is plotted last so it's on top
    ego_idx = int(data.track_index_to_predict)
    # Separate ego and non-ego indices, preserving order
    non_ego_indices = [idx for idx in agent_indices if idx != ego_idx]
    plot_indices = non_ego_indices + ([ego_idx] if ego_idx in agent_indices else [])

    legend_handles = []
    for idx in plot_indices:
        if idx < 0 or idx >= trajs.shape[0]:
            continue
        pts = trajs[idx]
        m = masks[idx]
        typ = agent_type_str[idx]

        is_ego = idx == ego_idx
        if is_ego:
            col = "tab:red"
        else:
            col = agent_color_map.get(typ, "gray")

        # draw line-by-line, solid for past, dashed for future
        for t in range(T - 1):
            if m[t] and m[t + 1]:
                if is_ego:
                    lw = 4 if t < Tp - 1 else 2
                else:
                    lw = 2 if t < Tp - 1 else 1
                ls = "-" if t < Tp - 1 else "--"
                ax.plot(
                    [pts[t, 0], pts[t + 1, 0]],
                    [pts[t, 1], pts[t + 1, 1]],
                    color=col,
                    ls=ls,
                    lw=lw,
                )
        # indicate bounding box or direction marker at the start
        valid_idxs = np.where(m)[0]
        if valid_idxs.size >= 2:
            start_i = valid_idxs[0]
            end_i = valid_idxs[-1]
            # start position
            sx, sy = pts[start_i]
            if show_bboxes:
                # extract object dimensions and heading
                length = float(data.obj_trajs[idx, start_i, 3])
                width = float(data.obj_trajs[idx, start_i, 4])
                # approximate heading from first motion vector
                dx = pts[start_i + 1, 0] - sx
                dy = pts[start_i + 1, 1] - sy
                angle = np.arctan2(dy, dx)
                # create rectangle centered at (sx, sy)
                rect = Rectangle(
                    (sx - length / 2, sy - width / 2),
                    length,
                    width,
                    edgecolor=col,
                    facecolor="none",
                    lw=lw,
                    zorder=5,
                )
                # rotate around center
                trans = (
                    mtransforms.Affine2D().rotate_around(sx, sy, angle) + ax.transData
                )
                rect.set_transform(trans)
                ax.add_patch(rect)
            else:
                # Use the same dot visualization as in plot_dataset_item
                # Dot at start position
                ax.scatter(sx, sy, marker="o", color=col, s=30, zorder=5)
                
                # Arrow from penultimate to last point to show direction
                prev = pts[end_i - 1]
                curr = pts[end_i]
                ax.annotate(
                    "",
                    xy=(curr[0], curr[1]),
                    xytext=(prev[0], prev[1]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=lw),
                    zorder=5,
                )

    # 8) Draw predictions (simplified from plot_prediction_item)
    if len(pred_probs) > 0 and len(pred_trajs) > 0:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(prediction_colormap)
        
        # Sort predictions by probability
        sort_indices = np.argsort(pred_probs)
        
        for idx in sort_indices:
            prob = pred_probs[idx]
            traj = pred_trajs[idx]
            
            # Color and line width based on probability
            color = cmap(prob * 5)  # Scale for visibility
            linewidth = 1 + prob * 2
            
            # Draw prediction trajectory
            for i in range(len(traj) - 1):
                if (np.abs(traj[i]) < 1000).all() and (np.abs(traj[i+1]) < 1000).all():
                    ax.plot(
                        [traj[i, 0], traj[i+1, 0]],
                        [traj[i, 1], traj[i+1, 1]],
                        linewidth=linewidth,
                        color=color,
                        linestyle=':',
                        alpha=0.6 + 0.4*prob,
                        zorder=10
                    )
            
            # Mark endpoint
            if len(traj) > 0 and (np.abs(traj[-1]) < 1000).all():
                ax.scatter(
                    traj[-1, 0], traj[-1, 1], 
                    s=20 + 50*prob, 
                    color=color, 
                    marker='x',
                    linewidth=2,
                    zorder=11
                )
                
                # Add probability text
                if show_probabilities and prob > 0.1:
                    ax.text(
                        traj[-1, 0], traj[-1, 1] + 0.5,
                        f"{prob:.2f}",
                        color=color,
                        fontsize=8,
                        ha='center',
                        va='bottom',
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, pad=0.5, boxstyle='round'),
                        zorder=12
                    )

    # 9) Finalize
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    ax.axis("off")
    if show_legend or show_map_legend:
        handles = []
        labels = []
        if show_legend:
            # Ego entry
            ego_type = agent_type_str[ego_idx]
            ego_label = f"Ego ({ego_type})"
            handles.append(Line2D([0], [0], color="tab:red", lw=4))
            labels.append(ego_label)
            # Other agent types
            seen = set()
            for idx in non_ego_indices:
                typ = agent_type_str[idx]
                if typ not in seen:
                    seen.add(typ)
                    col = agent_color_map.get(typ, "gray")
                    handles.append(Line2D([0], [0], color=col, lw=2))
                    labels.append(typ.capitalize())
            
            # Add prediction entries if predictions exist
            if len(pred_probs) > 0:
                highest_prob = np.max(pred_probs)
                handles.append(Line2D([0], [0], color=cmap(highest_prob * 5), lw=2, ls=':'))
                labels.append(f"Prediction (p={highest_prob:.2f})")
                
        if show_map_legend:
            # map legend items
            for lbl, item in map_legend_items.items():
                handles.append(item)
                labels.append(lbl)
        # place legend outside to the right
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="small",
        )
    if show_timestep_info:
        # annotate timestep info in top-left corner of axes
        info_text = f"Past steps: {Tp}, Future steps: {T - Tp}, Total: {T}"
        x0, y1 = ax.get_xlim()[0], ax.get_ylim()[1]
        ax.text(
            x0,
            y1,
            info_text,
            fontsize="small",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
        )
    # Add title with scenario, dataset, trajectory type, and ego index
    sc_id = data.scenario_id
    if isinstance(sc_id, (bytes, bytearray)):
        try:
            sc_id = sc_id.decode("utf-8")
        except:
            sc_id = str(sc_id)
    dataset_name = data.dataset_name
    if isinstance(dataset_name, (bytes, bytearray)):
        try:
            dataset_name = dataset_name.decode("utf-8")
        except:
            dataset_name = str(dataset_name)
    traj_label = None
    if getattr(data, "trajectory_type", None) is not None:
        traj_label = TRAJECTORY_TYPE_MAP.get(
            int(data.trajectory_type), str(data.trajectory_type)
        )
    title = f"Scenario: {sc_id} | Dataset: {dataset_name}"
    if traj_label:
        title += f" | Traj: {traj_label}"
    
    # Add prediction info to title
    if len(pred_probs) > 0:
        title += f" | {len(pred_probs)} predictions"
        
    fig.suptitle(title, fontsize="small")
    # Tighten layout within reserved margins
    fig.tight_layout(rect=[0, 0, 0.55, 0.98])
    return fig

def plot_prediction_item(
    data: DatasetItem,
    prediction: Dict[str, Any],
    agent_indices: Optional[List[int]] = None,
    show_legend: bool = True,
    show_map_legend: bool = True,
    vis_range: float = 50,
    include_agents_in_range: bool = False,
    show_timestep_info: bool = False,
    show_bboxes: bool = True,
    include_all_predictions: bool = True,
    prediction_top_k: int = 5,
    show_probabilities: bool = True,
    prediction_colormap: str = "rainbow",
    show_ids: bool = True,  # Add this parameter to control ID display
) -> plt.Figure:
    """
    Enhanced visualization for predictions that combines the style of plot_dataset_item
    with trajectory prediction visualization.

    Args:
        data: A single DatasetItem to visualize
        prediction: Dictionary containing prediction outputs with keys:
            - 'predicted_probability': tensor of shape [N, M] where N is batch size and M is number of modes
            - 'predicted_trajectory': tensor of shape [N, M, T, 2+] where T is trajectory length
        agent_indices: List of agent-indices (0…Nmax-1) whose trajectories we plot.
            If None, defaults to the "ego" agent given by data.track_index_to_predict.
        show_legend: Whether to draw a legend for agent trajectories.
        show_map_legend: Whether to draw a legend for map features.
        vis_range: Sets half-width/height of the visible region (in normalized coordinates).
        include_agents_in_range: Whether to automatically plot all agents whose trajectory passes within `vis_range`.
        show_timestep_info: Whether to annotate the number of past, future, and total timesteps on the plot.
        show_bboxes: If True, draw each object's 2D bounding box at the start of its trajectory instead of a start dot.
        include_all_predictions: Whether to show all prediction modes. If False, only top_k modes are shown.
        prediction_top_k: Number of top probability predictions to include if include_all_predictions is False.
        show_probabilities: Whether to annotate prediction endpoints with their probabilities.
        prediction_colormap: Matplotlib colormap name to use for predictions.

    Returns:
        A matplotlib Figure with the plotted scene including predictions.
    """
    # 1) Determine which agents to plot
    past_xy = data.obj_trajs[..., :2]  # [Nmax, Tp, 2]
    fut_xy = data.obj_trajs_future_state[..., :2]  # [Nmax, Tf, 2]
    trajs = np.concatenate([past_xy, fut_xy], axis=1)  # [Nmax, Tp+Tf, 2]

    past_mask = data.obj_trajs_mask  # [Nmax, Tp]
    fut_mask = data.obj_trajs_future_mask  # [Nmax, Tf]
    masks = np.concatenate([past_mask, fut_mask], axis=1)  # [Nmax, Tp+Tf]

    Tp = past_xy.shape[1]
    T = trajs.shape[1]

    # Extract prediction data
    if isinstance(prediction, dict):
        if "predicted_probability" in prediction and "predicted_trajectory" in prediction:
            pred_probs = prediction["predicted_probability"].detach().cpu().numpy()
            pred_trajs = prediction["predicted_trajectory"].detach().cpu().numpy()
            
            # Handle batch dimension if present
            if len(pred_probs.shape) > 1 and pred_probs.shape[0] > 1:
                # Use only the first element of the batch
                pred_probs = pred_probs[0]
                pred_trajs = pred_trajs[0]
            elif len(pred_probs.shape) > 1 and pred_probs.shape[0] == 1:
                # Squeeze singleton batch dimension
                pred_probs = pred_probs.squeeze(0)
                pred_trajs = pred_trajs.squeeze(0)
            
            # Remove the last point of each trajectory which appears to be invalid
            # and causes the trajectory to loop back
            if pred_trajs.shape[-2] > 1:  # Make sure there are enough points
                pred_trajs = pred_trajs[:, :-1]  # Remove last point of each trajectory
            
            # Select modes to visualize
            if not include_all_predictions and pred_probs.shape[0] > prediction_top_k:
                top_indices = np.argsort(pred_probs)[-prediction_top_k:]
                pred_probs = pred_probs[top_indices]
                pred_trajs = pred_trajs[top_indices]
            
            # Extract only the x, y coordinates from predictions
            if pred_trajs.shape[-1] > 2:
                pred_trajs = pred_trajs[..., :2]
        else:
            pred_probs = np.array([])
            pred_trajs = np.array([])
    else:
        pred_probs = np.array([])
        pred_trajs = np.array([])

    # Determine which agents to plot
    Nmax = trajs.shape[0]
    ego_idx = int(data.track_index_to_predict)
    if include_agents_in_range:
        agent_indices = []
        for i in range(Nmax):
            coords = trajs[i]  # [T,2]
            mask_i = masks[i]  # [T]
            cond_x = np.abs(coords[:, 0]) <= vis_range
            cond_y = np.abs(coords[:, 1]) <= vis_range
            in_box = np.logical_and(cond_x, cond_y)
            valid_mask = np.logical_and(mask_i.astype(bool), in_box)
            if np.any(valid_mask):
                agent_indices.append(i)
    elif agent_indices is None:
        agent_indices = [ego_idx]

    # Decode agent semantic types (only VEHICLE/PEDESTRIAN/CYCLIST bits)
    sem_onehot = data.obj_trajs[:, -1, 6:9]  # [Nmax, 3]
    sem_idx = np.argmax(sem_onehot, axis=-1)  # 0=vehicle,1=pedestrian,2=cyclist
    # Map to AGENT_TYPE_MAP keys (1=vehicle,2=pedestrian,3=bicycle)
    agent_type_str = [AGENT_TYPE_MAP.get(int(idx) + 1, "unset") for idx in sem_idx]

    # Set up color mapping for agents
    agent_color_map = {
        "unset": "gray",
        "vehicle": "tab:blue",
        "pedestrian": "tab:orange",
        "bicycle": "tab:green",
    }

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    # Reserve right margin for legend; trim top/bottom margins to reduce whitespace
    fig.subplots_adjust(left=0.02, right=0.75, top=0.98, bottom=0.02)
    ax.set_aspect("equal")

    # Plot map polylines
    K, L, _ = data.map_polylines.shape
    map_xy = data.map_polylines[..., :2]  # [K, L, 2]
    map_mask = data.map_polylines_mask  # [K, L]
    map_type_oh = data.map_polylines[:, 0, 9:29]  # [K, 20]
    map_type_idx = np.argmax(map_type_oh, axis=-1)  # [K]

    map_legend_items = {}
    for k in range(K):
        coords = map_xy[k]
        valid = map_mask[k]
        pidx = int(map_type_idx[k])
        ptype = (
            PolylineType(pidx)
            if pidx in PolylineType._value2member_map_
            else PolylineType.UNSET
        )

        # choose style by feature
        if ptype.name.startswith("LANE_"):
            color, ls, label = "lightgray", "-", "Lane"
        elif ptype == PolylineType.CROSSWALK:
            color, ls, label = "black", "--", "Crosswalk"
        elif ptype == PolylineType.SPEED_BUMP:
            color, ls, label = "magenta", "-", "SpeedBump"
        elif ptype == PolylineType.STOP_SIGN:
            color, ls, label = "red", "-", "StopSign"
        else:
            color, ls, label = "dimgray", "-", ptype.name.title().replace("_", " ")

        # for the legend, draw one invisible line per new label
        if show_map_legend and label not in map_legend_items:
            map_legend_items[label] = ax.plot(
                [], [], color=color, ls=ls, lw=2, label=label
            )[0]

        # draw the actual polyline segments
        for i in range(L - 1):
            if valid[i] and valid[i + 1]:
                ax.plot(
                    [coords[i, 0], coords[i + 1, 0]],
                    [coords[i, 1], coords[i + 1, 1]],
                    color=color,
                    ls=ls,
                    lw=1,
                    zorder=1,
                )

    # Plot agent trajectories
    # Ensure ego is plotted last so it's on top
    ego_idx = int(data.track_index_to_predict)
    # Separate ego and non-ego indices, preserving order
    non_ego_indices = [idx for idx in agent_indices if idx != ego_idx]
    plot_indices = non_ego_indices + ([ego_idx] if ego_idx in agent_indices else [])
    
    # Define color gradients for trajectories - MODIFIED THESE FUNCTIONS
    def interpolate_color(t, total_t):
        # Better color for non-ego vehicles (orange to blue gradient)
        return (0.9 - 0.9 * t / total_t, 0.6 - 0.3 * t / total_t, 0.1 + 0.7 * t / total_t)

    def interpolate_color_ego(t, total_t):
        # Keep the red to purple gradient for ego
        return (1.0 - 0.4 * t / total_t, 0, 0.5 * t / total_t)

    agent_legend_handles = []
    for idx in plot_indices:
        if idx < 0 or idx >= trajs.shape[0]:
            continue
        pts = trajs[idx]
        m = masks[idx]
        typ = agent_type_str[idx]

        is_ego = idx == ego_idx
        if is_ego:
            col = "tab:red"
            zorder = 5
        else:
            col = agent_color_map.get(typ, "gray")
            zorder = 3

        # draw line-by-line, solid for past, dashed for future
        for t in range(T - 1):
            if m[t] and m[t + 1]:
                if is_ego:
                    lw = 4 if t < Tp - 1 else 2
                    # Use color gradient
                    color = interpolate_color_ego(t, T)
                else:
                    lw = 2 if t < Tp - 1 else 1
                    # Use color gradient
                    color = interpolate_color(t, T)
                    
                ls = "-" if t < Tp - 1 else "--"
                ax.plot(
                    [pts[t, 0], pts[t + 1, 0]],
                    [pts[t, 1], pts[t + 1, 1]],
                    color=color,
                    ls=ls,
                    lw=lw,
                    zorder=zorder,
                    alpha=0.8,
                )
                
                # Add direction arrows every few segments for ego
                if is_ego and t % 3 == 0:
                    dx = pts[t + 1, 0] - pts[t, 0]
                    dy = pts[t + 1, 1] - pts[t, 1]
                    if dx*dx + dy*dy > 0.1:  # Only if segment is long enough
                        ax.arrow(
                            pts[t, 0], pts[t, 1], 
                            dx*0.5, dy*0.5, 
                            head_width=0.6, head_length=0.8, 
                            fc=color, ec=color, 
                            alpha=0.8,
                            zorder=zorder+1
                        )
                
        # indicate bounding box or direction marker at the start
        valid_idxs = np.where(m)[0]
        if valid_idxs.size >= 2:
            start_i = valid_idxs[0]
            end_i = valid_idxs[-1]
            # start position
            sx, sy = pts[start_i]
            if show_bboxes:
                # extract object dimensions and heading
                length = float(data.obj_trajs[idx, start_i, 3])
                width = float(data.obj_trajs[idx, start_i, 4])
                # approximate heading from first motion vector
                dx = pts[start_i + 1, 0] - sx
                dy = pts[start_i + 1, 1] - sy
                angle = np.arctan2(dy, dx)
                # create rectangle centered at (sx, sy)
                rect = Rectangle(
                    (sx - length / 2, sy - width / 2),
                    length,
                    width,
                    edgecolor=col,
                    facecolor=col if is_ego else "none",
                    alpha=0.3 if is_ego else 1.0,
                    lw=lw,
                    zorder=zorder + 2,
                )
                # rotate around center
                trans = (
                    mtransforms.Affine2D().rotate_around(sx, sy, angle) + ax.transData
                )
                rect.set_transform(trans)
                ax.add_patch(rect)
                
                # Add a direction indicator (triangle) for better visibility
                if is_ego:
                    triangle = plt.Polygon(
                        [[sx + length/2, sy],
                         [sx + length/3, sy + width/4], 
                         [sx + length/3, sy - width/4]],
                        color=col,
                        alpha=0.7,
                        zorder=zorder + 3
                    )
                    triangle.set_transform(trans)
                    ax.add_patch(triangle)
            else:
                # Use the same dot visualization as in plot_dataset_item
                # Dot at start position
                ax.scatter(sx, sy, marker="o", color=col, s=30, zorder=zorder + 2)
                
                # Arrow from penultimate to last point to show direction
                prev = pts[end_i - 1]
                curr = pts[end_i]
                ax.annotate(
                    "",
                    xy=(curr[0], curr[1]),
                    xytext=(prev[0], prev[1]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=lw),
                    zorder=zorder + 2,
                )

    # Draw the predictions
    if len(pred_probs) > 0 and len(pred_trajs) > 0:
        # Create colormap for predictions
        cmap = cm.get_cmap(prediction_colormap)
        
        # Sort predictions by probability for better visualization
        sort_indices = np.argsort(pred_probs)
        
        # Draw predictions from lowest to highest probability so highest are on top
        for idx in sort_indices:
            prob = pred_probs[idx]
            traj = pred_trajs[idx]
            
            # Color based on probability, but scaled to be more visible
            color = cmap(prob * 5)  # Multiply by 5 to make even low probabilities visible
            linewidth = 1 + prob * 3  # Scale width by probability
            
            # Draw prediction with dotted line
            for i in range(len(traj) - 1):
                # Check for valid trajectory segments (both points should have reasonable values)
                if (np.abs(traj[i]) < 1000).all() and (np.abs(traj[i+1]) < 1000).all():
                    ax.plot(
                        [traj[i, 0], traj[i+1, 0]],
                        [traj[i, 1], traj[i+1, 1]],
                        linewidth=linewidth,
                        color=color,
                        linestyle=':',
                        alpha=0.6 + 0.4*prob,  # Higher probability = more opaque
                        zorder=10
                    )
                
            # Mark the endpoint with a cross only if it's valid
            if len(traj) > 0 and (np.abs(traj[-1]) < 1000).all():
                ax.scatter(
                    traj[-1, 0], traj[-1, 1], 
                    s=30 + 70*prob, 
                    color=color, 
                    marker='x',
                    linewidth=2,
                    zorder=11
                )
            
            # Add probability text for significant predictions
            if show_probabilities and prob > 0.05 and len(traj) > 0 and (np.abs(traj[-1]) < 1000).all():
                ax.text(
                    traj[-1, 0], traj[-1, 1] + 1,
                    f"{prob:.2f}",
                    color=color,
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round'),
                    zorder=12
                )

    # Finalize
    # Adjust view based on trajectories and predictions
    all_points = []
    
    # Add trajectory points
    for idx in plot_indices:
        if idx < 0 or idx >= trajs.shape[0]:
            continue
        pts = trajs[idx]
        m = masks[idx]
        valid_points = pts[m.astype(bool)]
        all_points.extend(valid_points)
    
    # Add prediction points
    if len(pred_trajs) > 0:
        for traj in pred_trajs:
            # Only add valid points
            valid_points = [p for p in traj if (np.abs(p) < 1000).all()]
            all_points.extend(valid_points)
    
    if all_points:
        points = np.array(all_points)
        x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
        x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
        
        # Add margin
        margin = max(5, (x_max - x_min) * 0.1)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_xlim(-vis_range, vis_range)
        ax.set_ylim(-vis_range, vis_range)
        
    ax.axis("off")
    
    if show_legend or show_map_legend:
        handles = []
        labels = []
        
        if show_legend:
            # Ego entry
            ego_type = agent_type_str[ego_idx]
            ego_label = f"Ego ({ego_type})"
            handles.append(Line2D([0], [0], color="tab:red", lw=3))
            labels.append(ego_label)
            
            # Past/future trajectory entries
            ego_start_color = interpolate_color_ego(0, T)
            ego_end_color = interpolate_color_ego(T-1, T)
            handles.append(Line2D([0], [0], color=ego_start_color, lw=3, ls='-'))
            handles.append(Line2D([0], [0], color=ego_end_color, lw=2, ls='--'))
            labels.append("Past trajectory")
            labels.append("Future trajectory")
            
            # Other agent types
            seen = set()
            for idx in non_ego_indices:
                typ = agent_type_str[idx]
                if typ not in seen:
                    seen.add(typ)
                    col = agent_color_map.get(typ, "gray")
                    handles.append(Line2D([0], [0], color=col, lw=2))
                    labels.append(typ.capitalize())
            
            # Add prediction entries to legend
            if len(pred_probs) > 0:
                # Get highest and lowest probabilities for legend
                highest_prob = np.max(pred_probs)
                lowest_prob = np.min(pred_probs)
                
                if highest_prob > lowest_prob:
                    handles.append(Line2D([0], [0], color=cmap(highest_prob * 5), lw=3, ls=':'))
                    labels.append(f"Prediction (p={highest_prob:.2f})")
                    
                    handles.append(Line2D([0], [0], color=cmap(lowest_prob * 5), lw=1, ls=':'))
                    labels.append(f"Prediction (p={lowest_prob:.2f})")
                else:
                    handles.append(Line2D([0], [0], color=cmap(highest_prob * 5), lw=2, ls=':'))
                    labels.append(f"Prediction (p={highest_prob:.2f})")
                    
        if show_map_legend:
            # map legend items
            for lbl, item in map_legend_items.items():
                handles.append(item)
                labels.append(lbl)
                
        # place legend outside to the right
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize="medium",
        )
        
    if show_timestep_info:
        # annotate timestep info in top-left corner of axes
        info_text = f"Past steps: {Tp}, Future steps: {T - Tp}, Total: {T}"
        x0, y1 = ax.get_xlim()[0], ax.get_ylim()[1]
        ax.text(
            x0,
            y1,
            info_text,
            fontsize="small",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
        )
        
    # Add title with scenario, dataset, trajectory type info
    sc_id = data.scenario_id
    if isinstance(sc_id, (bytes, bytearray)):
        try:
            sc_id = sc_id.decode("utf-8")
        except:
            sc_id = str(sc_id)
            
    dataset_name = data.dataset_name
    if isinstance(dataset_name, (bytes, bytearray)):
        try:
            dataset_name = dataset_name.decode("utf-8")
        except:
            dataset_name = str(dataset_name)
            
    traj_label = None
    if getattr(data, "trajectory_type", None) is not None:
        traj_label = TRAJECTORY_TYPE_MAP.get(
            int(data.trajectory_type), str(data.trajectory_type)
        )
        
    title = f"Scenario: {sc_id} | Dataset: {dataset_name}"
    if traj_label:
        title += f" | Traj: {traj_label}"
        
    # Add number of predictions to title
    if len(pred_probs) > 0:
        title += f" | {len(pred_probs)} predictions"
        
    fig.suptitle(title, fontsize="medium")
    # Tighten layout within reserved margins
    fig.tight_layout(rect=[0, 0, 0.75, 0.98])
    
    # Add a mini-map in the corner showing the full scene
    ax_inset = fig.add_axes([0.05, 0.05, 0.15, 0.15])
    ax_inset.set_aspect('equal')
    
    # Draw map in mini-map
    for k in range(K):
        coords = map_xy[k]
        valid = map_mask[k]
        for i in range(L - 1):
            if valid[i] and valid[i + 1]:
                ax_inset.plot(
                    [coords[i, 0], coords[i + 1, 0]],
                    [coords[i, 1], coords[i + 1, 1]],
                    color="gray", 
                    linewidth=0.5,
                    alpha=0.5
                )
    
    # Show all agents in mini-map
    for idx in range(trajs.shape[0]):
        pts = trajs[idx]
        m = masks[idx]
        is_ego = idx == ego_idx
        
        valid_points = pts[m.astype(bool)]
        if len(valid_points) > 0:
            if is_ego:
                ax_inset.plot(valid_points[:, 0], valid_points[:, 1], 'r-', linewidth=1.5)
                ax_inset.scatter(valid_points[0, 0], valid_points[0, 1], color='red', s=20)
            else:
                ax_inset.plot(valid_points[:, 0], valid_points[:, 1], 'b-', linewidth=0.5, alpha=0.5)
    
    # Mark the current view area in the mini-map
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    rect = Rectangle(
        (xlim[0], ylim[0]), 
        xlim[1]-xlim[0], 
        ylim[1]-ylim[0], 
        fill=False, 
        edgecolor='purple', 
        linewidth=1.5
    )
    ax_inset.add_patch(rect)
    
    # Set mini-map limits to show a wider area
    if all_points:
        points = np.array(all_points)
        x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
        x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
        
        # Expand view
        margin = max(20, (x_max - x_min) * 0.5)
        ax_inset.set_xlim(x_min - margin, x_max + margin)
        ax_inset.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_inset.set_xlim(-vis_range*2, vis_range*2)
        ax_inset.set_ylim(-vis_range*2, vis_range*2)
        
    ax_inset.axis('off')
    
    return fig

def check_loaded_data(data: DatasetItem, index: int = 0) -> plt.Figure:
    """
    Visualizes agent trajectories and map data for a single sample.

    Args:
        data (DatasetItem): Object containing the data to visualize with attributes:
            - obj_trajs (np.ndarray): Past trajectories.
            - obj_trajs_future_state (np.ndarray): Future trajectories.
            - map_polylines (np.ndarray): Map polylines.
            - track_index_to_predict (np.ndarray or int): Indices of agents to predict.
        index (int, optional): Index of the agent to center the visualization on. Defaults to 0.

    Returns:
        matplotlib.pyplot.Figure: The matplotlib Figure object with the visualization.
    """
    fig, ax = plt.subplots()
    agents = np.concatenate(
        [data.obj_trajs[..., :2], data.obj_trajs_future_state[..., :2]], axis=-2
    )
    map_polylines = data.map_polylines

    if len(agents.shape) == 4:
        agents = agents[index]
        map_polylines = map_polylines[index]
        ego_index = data.track_index_to_predict[index]
        ego_agent = agents[ego_index]
    else:
        ego_index = data.track_index_to_predict
        ego_agent = agents[ego_index]

    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot(
            [point1[0], point2[0]],
            [point1[1], point2[1]],
            linewidth=line_width,
            color=color,
        )

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Plot the map with mask check
    for lane in map_polylines:
        for i in range(len(lane) - 1):
            draw_line_with_mask(lane[i, :2], lane[i, 6:8], color="grey", line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(
                        trajectory[t],
                        trajectory[t + 1],
                        color=color,
                        line_width=line_width,
                    )
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(
                        trajectory[t],
                        trajectory[t + 1],
                        color=color,
                        line_width=line_width,
                    )

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)

    # Create legend
    legend_handles = []
    legend_labels = []
    
    # Add map entry
    legend_handles.append(Line2D([0], [0], color='grey', lw=1))
    legend_labels.append("Map")
    
    # Add ego trajectory entry
    ego_start_color = interpolate_color_ego(0, 10)
    ego_end_color = interpolate_color_ego(9, 10)
    legend_handles.append(Line2D([0], [0], color=ego_start_color, lw=2))
    legend_handles.append(Line2D([0], [0], color=ego_end_color, lw=2))
    legend_labels.append("Ego (start)")
    legend_labels.append("Ego (end)")
    
    # Add other agents entry
    agent_start_color = interpolate_color(0, 10)
    agent_end_color = interpolate_color(9, 10)
    legend_handles.append(Line2D([0], [0], color=agent_start_color, lw=2))
    legend_handles.append(Line2D([0], [0], color=agent_end_color, lw=2))
    legend_labels.append("Other (start)")
    legend_labels.append("Other (end)")
    
    # Add the legend to the figure
    ax.legend(
        legend_handles, 
        legend_labels, 
        loc="upper right", 
        fontsize="small"
    )

    ax.axis("off")
    ax.axis("equal")
    
    # Set reasonable axis limits based on data
    if agents.size > 0:
        all_points = np.vstack(agents)
        x_min, y_min = np.min(all_points, axis=0) - 10
        x_max, y_max = np.max(all_points, axis=0) + 10
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    return fig


def visualize_batch_data(data: BatchInputDict) -> plt.Axes:
    """
    Visualizes batch data including agent trajectories and map information.

    Args:
        data (BatchInputDict): Dictionary containing batch data.
            - 'obj_trajs' (np.ndarray): Past trajectories.
            - 'obj_trajs_future_state' (np.ndarray): Future trajectories.
            - 'map_polylines' (np.ndarray): Map polylines.
            - 'obj_trajs_mask' (np.ndarray): Mask for past trajectories.
            - 'track_index_to_predict' (np.ndarray): Indices of agents to predict.

    Returns:
        matplotlib.axes.Axes: The axes object with the visualization.
    """
    fig, ax = plt.subplots()

    def decode_obj_trajs(obj_trajs):
        obj_trajs_xy = obj_trajs[..., :2]
        obj_lw = obj_trajs[..., -1, 3:5]
        obj_type_onehot = obj_trajs[..., -1, 6:9]
        obj_type = np.argmax(obj_type_onehot, axis=-1)
        obj_heading_encoding = obj_trajs[..., -1, 33:35]
        return obj_trajs_xy, obj_lw, obj_type, obj_heading_encoding

    def decode_map(map):
        map_xy = map[..., :2]
        map_type = map[..., 0, 9:29]
        map_type = np.argmax(map_type, axis=-1)
        return map_xy, map_type

    def plot_objects(obj_xy, obj_lw, obj_heading, obj_mask):
        for i in range(len(obj_lw)):
            if obj_mask[i]:
                length, width = obj_lw[i]
                sin_angle, cos_angle = obj_heading[i]
                angle = np.arctan2(sin_angle, cos_angle)
                x, y = obj_xy[i]
                rect = plt.Rectangle(
                    (-length / 2, -width / 2),
                    length,
                    width,
                    angle=0,
                    facecolor="none",
                    edgecolor="grey",
                    linewidth=1,
                )
                t = ax.transData
                rot = (
                    plt.matplotlib.transforms.Affine2D()
                    .rotate_around(0, 0, angle)
                    .translate(x, y)
                    + t
                )
                rect.set_transform(rot)
                ax.add_patch(rect)

    def draw_trajectory(trajectory, line_width, ego=False):
        def interpolate_color(start_color, end_color, t, total_t):
            return [
                (1 - t / total_t) * start + (t / total_t) * end
                for start, end in zip(start_color, end_color)
            ]

        def draw_line_with_mask(point1, point2, color, line_width=4):
            ax.plot(
                [point1[0], point2[0]],
                [point1[1], point2[1]],
                linewidth=line_width,
                color=color,
                alpha=0.5,
            )

        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(
                        trajectory[t],
                        trajectory[t + 1],
                        color=color,
                        line_width=line_width,
                    )
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(
                        trajectory[t],
                        trajectory[t + 1],
                        color=color,
                        line_width=line_width,
                    )

    obj_trajs = data["obj_trajs"]
    map = data["map_polylines"]

    obj_trajs_xy, obj_lw, obj_type, obj_heading = decode_obj_trajs(obj_trajs)
    obj_trajs_future_state = data["obj_trajs_future_state"][..., :2]
    all_traj = np.concatenate([obj_trajs_xy, obj_trajs_future_state], axis=-2)

    for i in range(obj_trajs.shape[0]):
        if i == data["track_index_to_predict"]:
            ego = True
        else:
            ego = False
        draw_trajectory(all_traj[i], line_width=3, ego=ego)

    map_xy, map_type = decode_map(map)
    obj_mask = data["obj_trajs_mask"]
    plot_objects(obj_trajs_xy[:, -1], obj_lw, obj_heading, obj_mask[:, -1])

    for indx, type in enumerate(map_type):
        lane = map_xy[indx]
        if type == 0:
            continue
        if type in [1, 2, 3]:
            color = "grey"
            linestyle = "dotted"
            linewidth = 1
        else:
            color = "grey"
            linestyle = "-"
            linewidth = 0.2
        for i in range(len(lane) - 1):
            if lane[i, 0] and lane[i + 1, 0]:
                ax.plot(
                    [lane[i, 0], lane[i + 1, 0]],
                    [lane[i, 1], lane[i + 1, 1]],
                    linewidth=linewidth,
                    color=color,
                    linestyle=linestyle,
                )

    vis_range = 35
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(True)
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    return ax


def concatenate_images(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """
    Concatenates a list of images into a grid.

    Args:
        images (List[PIL.Image.Image]): List of PIL Image objects.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        PIL.Image.Image: The concatenated image.
    """
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new("RGB", (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(
    image_list: List[Image.Image], column_counts: List[int]
) -> Image.Image:
    """
    Concatenates images into rows with varying numbers of columns.

    Args:
        image_list (List[PIL.Image.Image]): List of PIL Image objects.
        column_counts (List[int]): List where each element specifies the number of images in the corresponding row.

    Returns:
        PIL.Image.Image: The concatenated image.
    """
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = (
        original_height * column_counts[0]
    )  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new("RGB", (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new("RGB", (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image


def visualize_prediction(
    batch: Dict[str, Any], prediction: Dict[str, Any], 
    draw_index: int = 0, show_ids: bool = True, 
    show_objects: bool = True, default_length: float = 4.0, 
    default_width: float = 2.0
) -> plt.Figure:
    """
    Visualizes the prediction results for a given batch item.

    Args:
        batch (Dict[str, Any]): The input batch data.
        prediction (Dict[str, Any]): The model's prediction output.
        draw_index (int, optional): The index within the batch to visualize. Defaults to 0.

    Returns:
        matplotlib.pyplot.Figure: The matplotlib Figure object with the visualization.
    """

    def draw_line_with_mask(point1, point2, color, line_width=4):
        ax.plot(
            [point1[0], point2[0]],
            [point1[1], point2[1]],
            linewidth=line_width,
            color=color,
        )

    def interpolate_color(t, total_t):
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        return (1 - t / total_t, 0, t / total_t)

    def draw_trajectory(trajectory, line_width, ego=False, past=True):
        total_t = len(trajectory)
        # Skip drawing if too few valid points
        valid_points = [t for t in range(total_t) if trajectory[t, 0]]
        if len(valid_points) < 2:
            return
            
        # Use different line style for past vs future
        linestyle = '-' if past else '--'
        
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    # Draw line segment
                    ax.plot(
                        [trajectory[t, 0], trajectory[t + 1, 0]],
                        [trajectory[t, 1], trajectory[t + 1, 1]],
                        linewidth=line_width,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.8,
                        zorder=3 if ego else 2
                    )
                    
                    # Add arrow for direction every few segments
                    if t % 3 == 0:
                        dx = trajectory[t + 1, 0] - trajectory[t, 0]
                        dy = trajectory[t + 1, 1] - trajectory[t, 1]
                        if dx*dx + dy*dy > 0.1:  # Only if segment is long enough
                            ax.arrow(
                                trajectory[t, 0], trajectory[t, 1], 
                                dx*0.7, dy*0.7, 
                                head_width=0.8, head_length=0.8, 
                                fc=color, ec=color, 
                                alpha=0.8,
                                zorder=4 if ego else 3
                            )
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    # Draw line segment
                    ax.plot(
                        [trajectory[t, 0], trajectory[t + 1, 0]],
                        [trajectory[t, 1], trajectory[t + 1, 1]],
                        linewidth=line_width,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.7,
                        zorder=2
                    )
        
        # Add start/end markers
        first_valid = next((t for t in range(total_t) if trajectory[t, 0]), None)
        last_valid = next((t for t in range(total_t-1, -1, -1) if trajectory[t, 0]), None)
        
        if first_valid is not None:
            ax.scatter(
                trajectory[first_valid, 0], 
                trajectory[first_valid, 1], 
                s=50 if ego else 30,
                c='red' if ego else 'green',
                marker='o',
                zorder=5
            )
            
        if last_valid is not None and last_valid != first_valid:
            ax.scatter(
                trajectory[last_valid, 0], 
                trajectory[last_valid, 1], 
                s=50 if ego else 30,
                c='blue',
                marker='s' if ego else 'o',
                zorder=5
            )

    batch = batch["input_dict"]
    map_lanes = batch["map_polylines"][draw_index].cpu().numpy()
    map_mask = batch["map_polylines_mask"][draw_index].cpu().numpy()
    past_traj = batch["obj_trajs"][draw_index].cpu().numpy()
    future_traj = batch["obj_trajs_future_state"][draw_index].cpu().numpy()
    past_traj_mask = batch["obj_trajs_mask"][draw_index].cpu().numpy()
    future_traj_mask = batch["obj_trajs_future_mask"][draw_index].cpu().numpy()
    pred_future_prob = (
        prediction["predicted_probability"][draw_index].detach().cpu().numpy()
    )
    pred_future_traj = (
        prediction["predicted_trajectory"][draw_index].detach().cpu().numpy()
    )

    map_xy = map_lanes[..., :2]
    map_type = map_lanes[..., 0, -20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    
    # Draw map
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color="grey", line_width=1.5)

    # Draw past trajectories
    # for idx, traj in enumerate(past_traj):
    #     is_ego = idx == batch["track_index_to_predict"][draw_index].item()
    #     draw_trajectory(traj, line_width=3 if is_ego else 2, ego=is_ego, past=True)

    # Draw future trajectories  
    for idx, traj in enumerate(future_traj):
        is_ego = idx == batch["track_index_to_predict"][draw_index].item()
        draw_trajectory(traj, line_width=3 if is_ego else 2, ego=is_ego, past=False)
    
    # Draw predicted trajectories with better distinguishable styles
    for idx, traj in enumerate(pred_future_traj):
        prob = pred_future_prob[idx]
        color = cm.hot(prob)
        
        # Make higher probability predictions more visible
        linewidth = 1 + prob * 2
        
        # Draw prediction with dashed style
        for i in range(len(traj) - 1):
            ax.plot(
                [traj[i, 0], traj[i+1, 0]],
                [traj[i, 1], traj[i+1, 1]],
                linewidth=linewidth,
                color=color,
                linestyle=':',
                alpha=0.7 + 0.3*prob,
                zorder=10
            )
            
        # Add endpoint marker
        ax.scatter(traj[-1, 0], traj[-1, 1], s=30, color=color, marker='x', zorder=11)
        
        # # Add probability text for significant predictions
        # if prob > 0.2:  # Only for higher probability predictions
        #     ax.text(
        #         traj[-1, 0], traj[-1, 1] + 1,
        #         f"{prob:.2f}",
        #         color=color,
        #         fontsize=8,
        #         ha='center',
        #         va='bottom',
        #         bbox=dict(facecolor='white', alpha=0.5, pad=1)
            # )

    # Create legend
    legend_handles = []
    legend_labels = []
    
    # Map entry
    legend_handles.append(Line2D([0], [0], color='grey', lw=1.5))
    legend_labels.append("Map")
    
    # Observed trajectories entries
    ego_start = interpolate_color_ego(0, 10)
    ego_end = interpolate_color_ego(9, 10)
    other_start = interpolate_color(0, 10)
    other_end = interpolate_color(9, 10)
    
    legend_handles.append(Line2D([0], [0], color=ego_start, lw=3))
    legend_handles.append(Line2D([0], [0], color=ego_end, lw=3))
    legend_labels.append("Ego trajectory (start)")
    legend_labels.append("Ego trajectory (end)")
    
    legend_handles.append(Line2D([0], [0], color=other_start, lw=2))
    legend_handles.append(Line2D([0], [0], color=other_end, lw=2))
    legend_labels.append("Other agent (start)")
    legend_labels.append("Other agent (end)")
    
    # Prediction entries
    # Add a few samples from the colormap
    if len(pred_future_prob) > 0:
        # Sort probabilities to get highest and lowest for legend
        sorted_indices = np.argsort(pred_future_prob)
        highest_prob = pred_future_prob[sorted_indices[-1]]
        lowest_prob = pred_future_prob[sorted_indices[0]]
        
        legend_handles.append(Line2D([0], [0], color=cm.hot(highest_prob), lw=2))
        legend_labels.append(f"Prediction (p={highest_prob:.2f})")
        
        if len(pred_future_prob) > 1:
            legend_handles.append(Line2D([0], [0], color=cm.hot(lowest_prob), lw=2))
            legend_labels.append(f"Prediction (p={lowest_prob:.2f})")
    
    # Extract agent types if available (typically positions 6-8 of feature vector)
    agent_types = []
    if past_traj.shape[-1] >= 9:
        for idx in range(len(past_traj)):
            # Extract one-hot encoded agent type from features (vehicle, pedestrian, cyclist)
            agent_type_onehot = past_traj[idx, 0, 6:9]
            agent_type = np.argmax(agent_type_onehot)
            agent_types.append(agent_type)
    else:
        # Default to vehicle type if not available
        agent_types = [0] * len(past_traj)
    
    # Define colors and shapes for different agent types
    agent_colors = {
        0: 'tab:blue',      # vehicle
        1: 'tab:orange',    # pedestrian
        2: 'tab:green'      # cyclist
    }
    
    agent_markers = {
        0: 's',  # square for vehicles
        1: 'o',  # circle for pedestrians
        2: '^'   # triangle for cyclists
    }
    
    # When drawing objects, use agent type:
    for idx, traj in enumerate(past_traj):
        is_ego = idx == batch["track_index_to_predict"][draw_index].item()
        agent_type = agent_types[idx]
        
        # Use appropriate marker based on agent type
        marker = agent_markers.get(agent_type, 'o')
        
        # Generate color based on agent type (but override for ego)
        if is_ego:
            color = 'red'
        else:
            color = agent_colors.get(agent_type, 'gray')
            
        # Mark the start position with agent type-specific marker
        if past_traj_mask[idx, 0]:
            ax.scatter(
                traj[0, 0], traj[0, 1],
                s=100 if is_ego else 50,
                marker=marker,
                color=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.8,
                zorder=10
            )
    
    # Update legend to include agent types
    legend_handles.append(Line2D([0], [0], marker='s', color='tab:blue', 
                         linestyle='None', markersize=10, markeredgecolor='black'))
    legend_labels.append("Vehicle")
    
    legend_handles.append(Line2D([0], [0], marker='o', color='tab:orange', 
                         linestyle='None', markersize=10, markeredgecolor='black'))
    legend_labels.append("Pedestrian")
    
    legend_handles.append(Line2D([0], [0], marker='^', color='tab:green', 
                         linestyle='None', markersize=10, markeredgecolor='black'))
    legend_labels.append("Cyclist")
    
    # Add the legend to the figure
    ax.legend(
        legend_handles, 
        legend_labels, 
        loc="upper right", 
        fontsize="small"
    )

    # Set reasonable axis limits
    trajectories = np.concatenate([past_traj[..., :2], future_traj[..., :2]], axis=1)
    if trajectories.size > 0:
        # Flatten all trajectories to find bounds
        valid_points = []
        for traj in trajectories:
            valid_points.extend([p for p in traj if p[0]])
        
        if valid_points:
            points = np.array(valid_points)
            x_min, y_min = np.min(points[:, :2], axis=0) - 10
            x_max, y_max = np.max(points[:, :2], axis=0) + 10
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    
    # Add title with scenario info, if available
    if "scenario_id" in batch:
        scenario_id = batch["scenario_id"][draw_index]
        if hasattr(scenario_id, 'item'):
            scenario_id = scenario_id.item()
        ax.set_title(f"Scenario: {scenario_id}")

    ax.axis("off")
    # After retrieving data, add this function:
    def draw_object(position, agent_idx, is_ego=False, is_past=True):
        """Draw a vehicle rectangle at position"""
        # Default dimensions from parameters
        length, width = default_length, default_width
        
        # Try to extract dimensions if available
        try:
            if past_traj.shape[-1] > 4:  # Check if dimensions are available
                # Extract length, width from trajectories
                length = float(past_traj[agent_idx, 0, 3] if is_past else future_traj[agent_idx, 0, 3])
                width = float(past_traj[agent_idx, 0, 4] if is_past else future_traj[agent_idx, 0, 4])
        except (IndexError, ValueError):
            # Use defaults if extraction fails
            pass
            
        # Try to get heading if available
        heading = 0
        try:
            if past_traj.shape[-1] > 34:
                sin_heading = float(past_traj[agent_idx, 0, 33] if is_past else future_traj[agent_idx, 0, 33])
                cos_heading = float(past_traj[agent_idx, 0, 34] if is_past else future_traj[agent_idx, 0, 34])
                heading = np.arctan2(sin_heading, cos_heading)
            else:
                # Estimate heading from trajectory direction
                if is_past:
                    traj = past_traj[agent_idx]
                    mask = past_traj_mask[agent_idx]
                else:
                    traj = future_traj[agent_idx]
                    mask = future_traj_mask[agent_idx]
                
                # Find the first two valid points to estimate direction
                valid_indices = np.where(mask)[0]
                if len(valid_indices) >= 2:
                    p1 = traj[valid_indices[0], :2]
                    p2 = traj[valid_indices[1], :2]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    heading = np.arctan2(dy, dx)
        except (IndexError, ValueError):
            # Use default if extraction fails
            pass
                
        # Create rectangle with determined size
        color = 'red' if is_ego else ('blue' if not is_past else 'green')
        rect = Rectangle(
            (position[0] - length/2, position[1] - width/2),
            length, 
            width,
            angle=np.degrees(heading),
            color=color,
            fill=True,
            alpha=0.3 if is_past else 0.5
        )
        ax.add_patch(rect)
        
        # Add a direction indicator (triangle at front)
        try:
            t = ax.transData
            rot = mtransforms.Affine2D().rotate_around(position[0], position[1], heading)
            triangle = plt.Polygon(
                [[position[0] + length/2, position[1]],
                 [position[0] + length/3, position[1] + width/4], 
                 [position[0] + length/3, position[1] - width/4]],
                color=color,
                alpha=0.7
            )
            triangle.set_transform(rot + t)
            ax.add_patch(triangle)
        except Exception as e:
            # Skip triangle if there's an issue
            print(f"Could not draw direction indicator: {e}")
    
    # ...in the main code after drawing trajectories:
    
    # Draw objects at their positions if requested
    if show_objects:
        # Draw past objects (first position)
        for idx, traj in enumerate(past_traj):
            if idx < len(past_traj_mask) and past_traj_mask[idx, 0]:  # Check if position is valid
                is_ego = idx == batch["track_index_to_predict"][draw_index].item()
                draw_object(traj[0, :2], idx, is_ego=is_ego, is_past=True)

    
        # Draw future objects (last position)
        for idx, traj in enumerate(future_traj):
            if idx < len(future_traj_mask) and future_traj_mask[idx, -1]:  # Check if position is valid
                is_ego = idx == batch["track_index_to_predict"][draw_index].item()
                draw_object(traj[-1, :2], idx, is_ego=is_ego, is_past=False)
    
    # Add object IDs to the visualization
    if show_ids:
        for idx, traj in enumerate(past_traj):
            if past_traj_mask[idx, 0]:  # Check if position is valid
                is_ego = idx == batch["track_index_to_predict"][draw_index].item()
                
                # Add object ID text above the start position
                label = f"{'Ego' if is_ego else 'ID'}: {idx}"
                ax.text(
                    traj[0, 0], traj[0, 1] + 2,
                    label,
                    fontsize=8,
                    ha='center', 
                    va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round')
                )
    
    # Add a mini-map overview in corner
    # Add a mini-map in the corner showing the full scene
    ax_inset = fig.add_axes([0.05, 0.05, 0.2, 0.2])
    ax_inset.set_aspect('equal')
    
    # Draw map in mini-map
    for idx, lane in enumerate(map_xy):
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                ax_inset.plot(
                    [lane[i, 0], lane[i+1, 0]],
                    [lane[i, 1], lane[i+1, 1]],
                    color="gray", 
                    linewidth=0.5,
                    alpha=0.5
                )
    
    # Show all agents in mini-map
    for idx, traj in enumerate(past_traj):
        is_ego = idx == batch["track_index_to_predict"][draw_index].item()
        if is_ego:
            ax_inset.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5)
            ax_inset.scatter(traj[0, 0], traj[0, 1], color='red', s=20)
        else:
            ax_inset.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=0.5, alpha=0.5)
    
    # Mark the current view area in the mini-map
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    rect = Rectangle(
        (xlim[0], ylim[0]), 
        xlim[1]-xlim[0], 
        ylim[1]-ylim[0], 
        fill=False, 
        edgecolor='purple', 
        linewidth=1.5
    )
    ax_inset.add_patch(rect)
    
    # Set mini-map limits wider than main view
    all_points = []
    for traj in past_traj:
        all_points.extend([p for p in traj if p[0]])
    if all_points:
        points = np.array(all_points)
        x_min, y_min = np.min(points[:, :2], axis=0) - 20
        x_max, y_max = np.max(points[:, :2], axis=0) + 20
        ax_inset.set_xlim(x_min, x_max)
        ax_inset.set_ylim(y_min, y_max)
    
    ax_inset.axis('off')

    return fig

def configure_plotting(dpi=300, fig_format='retina'):
    """Configure matplotlib for high-quality output."""
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.figsize'] = (12, 9)
    # plt.rcParams['figure.format'] = fig_format
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.linewidth'] = 1.5
    # Use higher quality rendering for lines
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.0

def save_high_res_figure(fig, filename, dpi=300, format='png'):
    """Save a figure with high resolution."""
    fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
    print(f"Saved high-resolution figure to {filename}")