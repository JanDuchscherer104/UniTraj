from typing import Any, Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..datasets.types import BatchInputDict, DatasetItem

# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements


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
