import numpy as np
import pytest

from env.lidar import Lidar, LidarConfig, cast_rays


def _box(size=4.0):
    """Axis-aligned square room with corners (0,0)-(size,size)."""
    return np.array([
        [0.0, 0.0, size, 0.0],
        [size, 0.0, size, size],
        [size, size, 0.0, size],
        [0.0, size, 0.0, 0.0],
    ])


def test_single_ray_hits_wall_at_known_distance():
    segs = _box(4.0)
    d = cast_rays(np.array([2.0, 2.0]), np.array([0.0]), segs, max_range=10.0)
    assert d[0] == pytest.approx(2.0, abs=1e-9), "east wall is exactly 2m away"


def test_ray_distances_in_all_four_axis_directions():
    segs = _box(4.0)
    angles = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    d = cast_rays(np.array([1.0, 3.0]), angles, segs, max_range=10.0)
    assert d == pytest.approx([3.0, 1.0, 1.0, 3.0], abs=1e-9)


def test_ray_clipped_to_max_range_when_nothing_hit():
    segs = np.zeros((0, 4))
    d = cast_rays(np.array([0.0, 0.0]), np.array([0.0]), segs, max_range=3.5)
    assert d[0] == pytest.approx(3.5)


def test_ray_takes_nearest_of_multiple_hits():
    segs = np.array([
        [5.0, -1.0, 5.0, 1.0],   # far wall
        [2.0, -1.0, 2.0, 1.0],   # near wall
    ])
    d = cast_rays(np.array([0.0, 0.0]), np.array([0.0]), segs, max_range=10.0)
    assert d[0] == pytest.approx(2.0)


def test_ray_ignores_segments_behind_the_origin():
    segs = np.array([[-2.0, -1.0, -2.0, 1.0]])
    d = cast_rays(np.array([0.0, 0.0]), np.array([0.0]), segs, max_range=5.0)
    assert d[0] == pytest.approx(5.0), "a wall behind the ray must not be hit"


def test_ray_parallel_to_segment_does_not_hit():
    segs = np.array([[0.0, 1.0, 5.0, 1.0]])
    d = cast_rays(np.array([0.0, 0.0]), np.array([0.0]), segs, max_range=5.0)
    assert d[0] == pytest.approx(5.0)


def test_diagonal_ray_distance():
    segs = _box(4.0)
    d = cast_rays(np.array([2.0, 2.0]), np.array([np.pi / 4]), segs, max_range=10.0)
    assert d[0] == pytest.approx(2.0 * np.sqrt(2.0), abs=1e-9)


def test_cast_rays_is_vectorised_over_rays():
    segs = _box(4.0)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    d = cast_rays(np.array([2.0, 2.0]), angles, segs, max_range=10.0)
    assert d.shape == (360,)
    assert (d <= 2.0 * np.sqrt(2.0) + 1e-9).all()


def test_lidar_scan_shape_and_bounds():
    lidar = Lidar(LidarConfig())
    rng = np.random.default_rng(0)
    d = lidar.scan(np.array([2.0, 2.0, 0.0]), _box(4.0), rng)
    assert d.shape == (24,)
    assert (d >= 0.0).all() and (d <= 3.5 + 1e-9).all()


def test_lidar_ray_angles_span_fov_symmetrically():
    lidar = Lidar(LidarConfig(n_rays=5, fov=np.pi))
    a = lidar.ray_angles
    assert a.shape == (5,)
    assert a[0] == pytest.approx(-np.pi / 2)
    assert a[-1] == pytest.approx(np.pi / 2)


def test_lidar_respects_robot_heading():
    """Rotating the robot by 90 deg rotates the scan by one ray-index step."""
    cfg = LidarConfig(n_rays=5, fov=np.deg2rad(270), noise_std=0.0, dropout=0.0)
    lidar = Lidar(cfg)
    rng = np.random.default_rng(0)
    segs = np.array([[3.0, -5.0, 3.0, 5.0]])  # wall to the east
    facing_east = lidar.scan(np.array([0.0, 0.0, 0.0]), segs, rng)
    facing_north = lidar.scan(np.array([0.0, 0.0, np.pi / 2]), segs, rng)
    assert facing_east.min() < facing_north.min()


def test_lidar_noise_is_zero_when_disabled():
    cfg = LidarConfig(noise_std=0.0, dropout=0.0)
    lidar = Lidar(cfg)
    a = lidar.scan(np.array([2.0, 2.0, 0.0]), _box(4.0), np.random.default_rng(1))
    b = lidar.scan(np.array([2.0, 2.0, 0.0]), _box(4.0), np.random.default_rng(2))
    assert np.allclose(a, b), "with noise off, scans must be deterministic"


def test_lidar_noise_perturbs_readings_when_enabled():
    cfg = LidarConfig(noise_std=0.05, dropout=0.0)
    lidar = Lidar(cfg)
    a = lidar.scan(np.array([2.0, 2.0, 0.0]), _box(4.0), np.random.default_rng(1))
    b = lidar.scan(np.array([2.0, 2.0, 0.0]), _box(4.0), np.random.default_rng(2))
    assert not np.allclose(a, b)


def test_lidar_clamps_to_min_range():
    cfg = LidarConfig(noise_std=0.0, dropout=0.0, min_range=0.12)
    lidar = Lidar(cfg)
    segs = np.array([[0.05, -1.0, 0.05, 1.0]])  # wall closer than min_range
    d = lidar.scan(np.array([0.0, 0.0, 0.0]), segs, np.random.default_rng(0))
    assert d.min() >= 0.12 - 1e-9
