"""Unit tests for differential drive kinematics and acceleration limits."""
import numpy as np
import pytest
from core.robot import DifferentialDriveRobot, RobotState, RobotConfig


def test_velocity_acceleration_clamping():
    cfg = RobotConfig(max_v_m_s=1.0, max_acc_v_m_s2=0.5)
    robot = DifferentialDriveRobot(initial_state=RobotState(0, 0, 0, 0, 0), config=cfg)

    # Command instant jump to 1.0 m/s with dt = 0.1
    # Max dv = 0.5 * 0.1 = 0.05 m/s
    st = robot.step(target_v=1.0, target_w=0.0, dt=0.1)
    assert np.isclose(st.v, 0.05)


def test_forward_displacement_integration():
    robot = DifferentialDriveRobot(initial_state=RobotState(0, 0, 0, 0.5, 0))
    st = robot.step(target_v=0.5, target_w=0.0, dt=1.0)

    # Moving at 0.5 m/s along heading 0 (X-axis) for 1s
    assert np.isclose(st.x, 0.50, atol=1e-3)
    assert np.isclose(st.y, 0.0, atol=1e-3)
    assert np.isclose(st.theta, 0.0, atol=1e-3)


def test_angular_heading_wrapping():
    robot = DifferentialDriveRobot(initial_state=RobotState(0, 0, np.pi - 0.1, 0, 0))
    # Turn left past pi
    st = robot.step(target_v=0.0, target_w=0.5, dt=1.0)
    # Theta must be wrapped to [-pi, pi]
    assert -np.pi <= st.theta <= np.pi
