"""Differential drive mobile robot kinematics and dynamic constraints."""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class RobotState:
    x: float          # Global X in meters
    y: float          # Global Y in meters
    theta: float      # Orientation heading in radians [-pi, pi]
    v: float          # Linear forward velocity (m/s)
    w: float          # Angular yaw velocity (rad/s)


@dataclass
class RobotConfig:
    radius_m: float = 0.30          # Bounding collision circle radius
    max_v_m_s: float = 0.80         # Maximum linear speed
    min_v_m_s: float = 0.0          # Non-holonomic forward-only drive
    max_w_rad_s: float = 1.20       # Maximum rotational speed
    max_acc_v_m_s2: float = 0.80    # Linear acceleration limit
    max_acc_w_rad_s2: float = 1.60  # Angular acceleration limit
    wheel_base_m: float = 0.45      # Distance between drive wheels


class DifferentialDriveRobot:
    """Non-holonomic unicycle / differential-drive automated guided vehicle (AGV)."""

    def __init__(self, initial_state: Optional[RobotState] = None, config: Optional[RobotConfig] = None):
        self.state = initial_state or RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, w=0.0)
        self.config = config or RobotConfig()

    def step(self, target_v: float, target_w: float, dt: float) -> RobotState:
        """Update robot state under dynamic acceleration limits and kinematics."""
        # 1. Enforce acceleration limits (Rate of Change)
        dv = np.clip(target_v - self.state.v, -self.config.max_acc_v_m_s2 * dt, self.config.max_acc_v_m_s2 * dt)
        new_v = np.clip(self.state.v + dv, self.config.min_v_m_s, self.config.max_v_m_s)

        dw = np.clip(target_w - self.state.w, -self.config.max_acc_w_rad_s2 * dt, self.config.max_acc_w_rad_s2 * dt)
        new_w = np.clip(self.state.w + dw, -self.config.max_w_rad_s, self.config.max_w_rad_s)

        # 2. Kinematic Forward Integration (Midpoint Runge-Kutta 2)
        half_theta = self.state.theta + 0.5 * new_w * dt
        new_x = self.state.x + new_v * np.cos(half_theta) * dt
        new_y = self.state.y + new_v * np.sin(half_theta) * dt
        new_theta = (self.state.theta + new_w * dt + np.pi) % (2.0 * np.pi) - np.pi

        self.state = RobotState(x=new_x, y=new_y, theta=new_theta, v=new_v, w=new_w)
        return self.state
