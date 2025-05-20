from typing import Any, Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
    agent_indices: Optional[List[int]] = None,
    show_legend: bool = True,
    show_map_legend: bool = True,
    vis_range: float = 50,
    include_agents_in_range: bool = False,
    show_timestep_info: bool = False,
    show_bboxes: bool = False,
) -> plt.Figure:
    """
    Improved static visualization for a single DatasetItem.

    Args:
        data: A single DatasetItem to visualize.
        agent_indices: List of agent-indices (0…Nmax-1) whose trajectories we plot.
            If None, defaults to the “ego” agent given by data.track_index_to_predict.
        show_legend: Whether to draw a legend for agent trajectories.
        show_map_legend: Whether to draw a legend for map features.
        vis_range: sets half-width/height of the visible region (in normalized coordinates).
        include_agents_in_range: Whether to automatically plot all agents whose trajectory passes within `vis_range`.
        show_timestep_info: Whether to annotate the number of past, future, and total timesteps on the plot.
        show_bboxes: If True, draw each object's 2D bounding box at the start of its trajectory instead of a start dot.

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

    # 6) Plot map polylines
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
                # dot at start
                ax.scatter(sx, sy, marker="o", color=col, s=30, zorder=5)
                # arrow from penultimate to last point
                prev = pts[end_i - 1]
                curr = pts[end_i]
                ax.annotate(
                    "",
                    xy=(curr[0], curr[1]),
                    xytext=(prev[0], prev[1]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=lw),
                    zorder=5,
                )

    # 8) Finalize
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
    fig.suptitle(title, fontsize="small")
    # Tighten layout within reserved margins
    fig.tight_layout(rect=[0, 0, 0.55, 0.98])
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

    ax.axis("off")
    ax.axis("equal")
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
                start_color = (0, 0, 0.5)
                end_color = (0.53, 0.81, 0.98)
            else:
                start_color = (0, 0.5, 0)
                end_color = (0.56, 0.93, 0.56)
            color = interpolate_color(start_color, end_color, t, total_t)
            if trajectory[t, 0] and trajectory[t + 1, 0]:
                draw_line_with_mask(
                    trajectory[t], trajectory[t + 1], color=color, line_width=line_width
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
    batch: Dict[str, Any], prediction: Dict[str, Any], draw_index: int = 0
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

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                draw_line_with_mask(lane[i], lane[i + 1], color="grey", line_width=1.5)

    for idx, traj in enumerate(past_traj):
        draw_trajectory(traj, line_width=2)

    for idx, traj in enumerate(future_traj):
        draw_trajectory(traj, line_width=2)

    for idx, traj in enumerate(pred_future_traj):
        color = cm.hot(pred_future_prob[idx])
        for i in range(len(traj) - 1):
            draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=2)

    return fig
