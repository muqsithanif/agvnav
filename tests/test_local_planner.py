"""Unit tests for Dynamic Window Approach (DWA)."""
import numpy as np
import pytest
from core.robot import RobotState, RobotConfig
from core.local_planner import DynamicWindowPlanner, DWAConfig


def test_clear_path_forward_velocity():
    dwa = DynamicWindowPlanner()
    st = RobotState(x=0.0, y=0.0, theta=0.0, v=0.2, w=0.0)
    target = (3.0, 0.0)
    obstacles = []

    v, w, traj = dwa.plan_velocity_command(st, target, obstacles)
    # Clear path directly ahead: v must be positive, w near 0
    assert v > 0.1
    assert abs(w) < 0.2
    assert len(traj) > 0


def test_obstacle_immediate_brake_or_turn():
    dwa = DynamicWindowPlanner()
    st = RobotState(x=0.0, y=0.0, theta=0.0, v=0.4, w=0.0)
    target = (3.0, 0.0)
    # Impassable obstacle right in front of the robot at 0.45m
    obstacles = [(0.45, 0.0), (0.45, 0.1), (0.45, -0.1)]

    v, w, traj = dwa.plan_velocity_command(st, target, obstacles)
    # Robot must either turn sharply (|w| > 0.4) or brake (v < 0.1)
    assert (abs(w) > 0.4) or (v < 0.15)
