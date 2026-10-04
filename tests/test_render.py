import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from env.lidar import LidarConfig  # noqa: E402
from env.maps import HAND_MAPS  # noqa: E402
from env.robot_env import EnvConfig, RobotNavEnv  # noqa: E402
from env.render import draw_env, plot_trajectories, render_frame, write_video  # noqa: E402


def _stepped_env(n=10):
    env = RobotNavEnv(env_cfg=EnvConfig(), floorplan=HAND_MAPS["u_trap"](),
                      lidar_cfg=LidarConfig(noise_std=0.0, dropout=0.0))
    env.reset(seed=0)
    for _ in range(n):
        env.step(np.array([1.0, 0.8], dtype=np.float32))
    return env


def test_draw_env_adds_artists_to_the_axis():
    env = _stepped_env()
    fig, ax = plt.subplots()
    draw_env(env, ax)
    assert len(ax.lines) + len(ax.patches) + len(ax.collections) > 0
    plt.close(fig)


def test_render_frame_returns_rgb_array():
    frame = render_frame(_stepped_env())
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_render_frame_is_not_blank():
    frame = render_frame(_stepped_env())
    assert frame.std() > 1.0, "a blank canvas means nothing was drawn"


def test_render_frames_have_consistent_shape_across_steps():
    env = _stepped_env(2)
    a = render_frame(env)
    for _ in range(5):
        env.step(np.array([1.0, 1.0], dtype=np.float32))
    b = render_frame(env)
    assert a.shape == b.shape, "video writers require a fixed frame size"


def test_write_gif_creates_a_file(tmp_path):
    frames = [render_frame(_stepped_env(i + 1)) for i in range(3)]
    out = tmp_path / "clip.gif"
    write_video(frames, str(out), fps=10)
    assert out.exists() and out.stat().st_size > 0


def test_write_video_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        write_video([render_frame(_stepped_env())], str(tmp_path / "clip.avi"))


def test_plot_trajectories_writes_a_figure(tmp_path):
    env = _stepped_env(20)
    out = tmp_path / "overlay.png"
    plot_trajectories(
        [(env.floorplan, np.array(env.trajectory), env.start, env.goal)],
        labels=["agent"], path=str(out), title="test",
    )
    assert out.exists() and out.stat().st_size > 0
