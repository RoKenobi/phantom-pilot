"""Matplotlib rendering of the robot, its LiDAR and its path.

Kept deliberately plain: the demo video is a deliverable, but the figures
have to stay readable in a README at small size, so no 3D and no shading.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from env.robot_env import ROBOT_RADIUS

_WALL = "#2b2b2b"
_RAY = "#d94801"
_ROBOT = "#1f77b4"
_GOAL = "#2ca02c"
_TRACE = "#1f77b4"


def draw_env(env, ax, show_rays: bool = True, show_trajectory: bool = True,
             title: str | None = None) -> None:
    fp = env.floorplan
    w, h = fp.extent

    for x1, y1, x2, y2 in fp.segments:
        ax.plot([x1, x2], [y1, y2], color=_WALL, linewidth=1.4, solid_capstyle="round")

    if show_rays:
        angles = env.lidar.ray_angles + env.pose[2]
        ends = env.pose[:2][None, :] + env.scan[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        for (ex, ey) in ends:
            ax.plot([env.pose[0], ex], [env.pose[1], ey],
                    color=_RAY, linewidth=0.5, alpha=0.45, zorder=1)

    if show_trajectory and len(env.trajectory) > 1:
        traj = np.array(env.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color=_TRACE, linewidth=1.2, alpha=0.7, zorder=2)

    ax.add_patch(Circle(env.goal, 0.25, color=_GOAL, alpha=0.35, zorder=2))
    ax.plot(*env.goal, marker="*", color=_GOAL, markersize=14, zorder=4)

    ax.add_patch(Circle(env.pose[:2], ROBOT_RADIUS, color=_ROBOT, zorder=5))
    ax.arrow(env.pose[0], env.pose[1],
             0.3 * np.cos(env.pose[2]), 0.3 * np.sin(env.pose[2]),
             head_width=0.1, color=_ROBOT, zorder=6)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def render_frame(env, figsize=(5, 5), dpi=100, **draw_kwargs) -> np.ndarray:
    """Draw the current state and return it as an RGB array."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_env(env, ax, **draw_kwargs)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def write_video(frames: list[np.ndarray], path: str, fps: int = 20) -> None:
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, fps=fps, loop=0)
    elif path.endswith(".mp4"):
        imageio.mimsave(path, frames, fps=fps, codec="libx264",
                        macro_block_size=None)
    else:
        raise ValueError(f"unsupported video extension: {path!r} (use .gif or .mp4)")


def plot_trajectories(snapshots, labels, path: str, title: str = "") -> None:
    """Overlay several runs. `snapshots` is a list of
    (floorplan, trajectory (T,3), start (2,), goal (2,))."""
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), dpi=120, squeeze=False)
    for ax, (fp, traj, start, goal), label in zip(axes[0], snapshots, labels):
        for x1, y1, x2, y2 in fp.segments:
            ax.plot([x1, x2], [y1, y2], color=_WALL, linewidth=1.2)
        ax.plot(traj[:, 0], traj[:, 1], color=_TRACE, linewidth=1.6)
        ax.plot(*start, marker="o", color="black", markersize=6)
        ax.plot(*goal, marker="*", color=_GOAL, markersize=15)
        w, h = fp.extent
        ax.set_xlim(0, w); ax.set_ylim(0, h)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
