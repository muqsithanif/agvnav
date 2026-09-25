"""Dynamic Window Approach (DWA) local trajectory optimizer for reactive obstacle avoidance."""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from core.robot import RobotState, RobotConfig


@dataclass
class DWAConfig:
    predict_time_s: float = 1.6         # Forward trajectory rollout horizon
    dt: float = 0.10                    # Trajectory simulation step
    v_samples: int = 12                 # Velocity search space resolution
    w_samples: int = 25                 # Angular velocity search space resolution
    alpha_heading: float = 0.40         # Target heading weight
    beta_clearance: float = 0.15        # Obstacle clearance weight
    gamma_velocity: float = 0.30        # Velocity maximization weight
    safety_margin_m: float = 0.10       # Clearance kept beyond the robot radius
    sensing_margin_m: float = 0.05      # Obstacle points sit up to half a grid cell inside the surface


class DynamicWindowPlanner:
    """Dynamic Window Approach (DWA) real-time local motion planner."""

    def __init__(self, robot_config: Optional[RobotConfig] = None, dwa_config: Optional[DWAConfig] = None):
        self.r_cfg = robot_config or RobotConfig()
        self.dwa_cfg = dwa_config or DWAConfig()

    def plan_velocity_command(
        self,
        current_state: RobotState,
        target_waypoint: Tuple[float, float],
        obstacles: List[Tuple[float, float]],  # List of (x, y) obstacle points
        v_limit: Optional[float] = None,       # speed cap, e.g. from a safety warning field
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """Compute optimal (v, w) command and corresponding trajectory rollout.

        Returns:
            (best_v, best_w, predicted_trajectory_points)
        """
        # 1. Compute Dynamic Window Vr = Vs ∩ Vd
        vs = [self.r_cfg.min_v_m_s, self.r_cfg.max_v_m_s, -self.r_cfg.max_w_rad_s, self.r_cfg.max_w_rad_s]
        vd = [
            current_state.v - self.r_cfg.max_acc_v_m_s2 * self.dwa_cfg.dt,
            current_state.v + self.r_cfg.max_acc_v_m_s2 * self.dwa_cfg.dt,
            current_state.w - self.r_cfg.max_acc_w_rad_s2 * self.dwa_cfg.dt,
            current_state.w + self.r_cfg.max_acc_w_rad_s2 * self.dwa_cfg.dt,
        ]

        v_min = max(vs[0], vd[0])
        v_max = min(vs[1], vd[1])
        w_min = max(vs[2], vd[2])
        w_max = min(vs[3], vd[3])
        if v_limit is not None:
            # Brake towards the cap as fast as the dynamic window allows.
            v_max = min(v_max, max(v_limit, v_min))

        v_candidates = np.linspace(v_min, v_max, self.dwa_cfg.v_samples)
        w_candidates = np.linspace(w_min, w_max, self.dwa_cfg.w_samples)

        best_score = -float("inf")
        best_v = 0.0
        best_w = 0.0
        best_traj: List[Tuple[float, float]] = []

        obs_np = np.array(obstacles) if len(obstacles) > 0 else np.empty((0, 2))

        # Admissible trajectories keep the robot radius plus the safety margin.
        # If the robot is already inside the margin, a trajectory is still
        # admissible as long as it does not get any closer; otherwise every
        # option, including turning on the spot, is rejected and it deadlocks.
        hard_limit = self.r_cfg.radius_m + self.dwa_cfg.sensing_margin_m
        current = self._compute_obstacle_clearance([(current_state.x, current_state.y, current_state.theta)], obs_np)
        limit = max(hard_limit, min(self.r_cfg.radius_m + self.dwa_cfg.safety_margin_m, current) - 1e-3)

        for v in v_candidates:
            for w in w_candidates:
                traj = self._rollout_trajectory(current_state, v, w)

                min_obs_dist = self._compute_obstacle_clearance(traj, obs_np)
                if min_obs_dist < limit:
                    continue

                # Score evaluation
                heading_score = self._compute_heading_score(traj[-1], target_waypoint)
                clearance_score = min(min_obs_dist, 2.0)  # Bound clearance utility
                velocity_score = v / (self.r_cfg.max_v_m_s + 1e-6)

                total_score = (
                    self.dwa_cfg.alpha_heading * heading_score
                    + self.dwa_cfg.beta_clearance * clearance_score
                    + self.dwa_cfg.gamma_velocity * velocity_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_v = v
                    best_w = w
                    best_traj = [(p[0], p[1]) for p in traj]

        # If trapped or no viable trajectory found, safely brake
        if best_score == -float("inf"):
            best_v = 0.0
            best_w = 0.0
            best_traj = [(current_state.x, current_state.y)]

        return best_v, best_w, best_traj

    def _rollout_trajectory(self, start_state: RobotState, v: float, w: float) -> List[Tuple[float, float, float]]:
        """Simulate robot kinematics forward in time."""
        steps = int(self.dwa_cfg.predict_time_s / self.dwa_cfg.dt)
        traj = []
        x, y, theta = start_state.x, start_state.y, start_state.theta

        for _ in range(steps):
            x += v * np.cos(theta) * self.dwa_cfg.dt
            y += v * np.sin(theta) * self.dwa_cfg.dt
            theta += w * self.dwa_cfg.dt
            traj.append((x, y, theta))
        return traj

    def _compute_heading_score(self, final_pose: Tuple[float, float, float], target_waypoint: Tuple[float, float]) -> float:
        """Measure orientation alignment with vector toward target."""
        fx, fy, ftheta = final_pose
        dx = target_waypoint[0] - fx
        dy = target_waypoint[1] - fy
        target_angle = np.arctan2(dy, dx)
        diff_angle = np.abs((target_angle - ftheta + np.pi) % (2.0 * np.pi) - np.pi)
        return float(np.pi - diff_angle)

    def _compute_obstacle_clearance(self, traj: List[Tuple[float, float, float]], obs_np: np.ndarray) -> float:
        """Calculate minimum distance from any point along rollout to obstacles."""
        if len(obs_np) == 0:
            return 10.0

        traj_xy = np.array([(p[0], p[1]) for p in traj])
        # Compute pairwise distance matrix: [N_traj, N_obs]
        dists = np.linalg.norm(traj_xy[:, None, :] - obs_np[None, :, :], axis=2)
        return float(np.min(dists))
