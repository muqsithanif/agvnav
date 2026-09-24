"""Unit tests for 2D Planar LiDAR scanner."""
import numpy as np
import pytest
from core.robot import RobotState
from core.lidar import PlanarLidar


def test_lidar_angles_range():
    lidar = PlanarLidar(num_beams=90, fov_rad=np.pi)
    assert len(lidar.beam_angles) == 90
    assert np.isclose(lidar.beam_angles[0], -np.pi / 2.0)
    assert np.isclose(lidar.beam_angles[-1], np.pi / 2.0)


def test_lidar_obstacle_detection():
    lidar = PlanarLidar(num_beams=91, fov_rad=np.pi, noise_std_m=0.0)
    # Robot at origin looking +X (theta = 0)
    st = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, w=0.0)
    # Obstacle 2.0 meters directly ahead at (2.0, 0.0)
    obstacles = [(2.0, 0.0)]

    scan = lidar.scan(st, obstacles)
    # Center beam (beam index 45) should read approximately 2.0 meters
    center_idx = 45
    assert np.isclose(scan.ranges[center_idx], 2.0, atol=0.1)
    assert len(scan.point_cloud_global) > 0
