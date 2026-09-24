"""2D Planar laser rangefinder (LiDAR) raycasting simulation."""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from core.robot import RobotState


@dataclass
class LaserScan:
    ranges: np.ndarray      # Measured distance per beam in meters
    angles: np.ndarray      # Angle of each beam relative to robot heading in radians
    point_cloud_global: List[Tuple[float, float]]  # (x, y) hit points in world frame


class PlanarLidar:
    """Simulates a 2D industrial safety LiDAR scanner mounted to the AGV chassis."""

    def __init__(
        self,
        num_beams: int = 90,
        fov_rad: float = np.pi,  # 180 degree forward field of view
        range_min_m: float = 0.10,
        range_max_m: float = 6.0,
        noise_std_m: float = 0.015,
        seed: int = 42,
    ):
        self.num_beams = num_beams
        self.fov_rad = fov_rad
        self.range_min = range_min_m
        self.range_max = range_max_m
        self.noise_std = noise_std_m
        self.rng = np.random.RandomState(seed)

        self.beam_angles = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, num_beams)

    def scan(self, state: RobotState, obstacle_points: List[Tuple[float, float]]) -> LaserScan:
        """Cast beams from robot state and return ranges and detected obstacle point cloud."""
        rx, ry, rtheta = state.x, state.y, state.theta
        ranges = np.full(self.num_beams, self.range_max, dtype=np.float32)
        detected_points: List[Tuple[float, float]] = []

        if len(obstacle_points) == 0:
            return LaserScan(ranges=ranges, angles=self.beam_angles, point_cloud_global=[])

        obs_np = np.array(obstacle_points)  # [N, 2]
        dx = obs_np[:, 0] - rx
        dy = obs_np[:, 1] - ry
        obs_dists = np.sqrt(dx ** 2 + dy ** 2)
        obs_angles_global = np.arctan2(dy, dx)
        obs_angles_local = (obs_angles_global - rtheta + np.pi) % (2.0 * np.pi) - np.pi

        for b_idx, b_angle in enumerate(self.beam_angles):
            # Find obstacles within narrow angular cone of this beam
            angle_diff = np.abs(obs_angles_local - b_angle)
            valid_mask = (angle_diff < (self.fov_rad / self.num_beams)) & (obs_dists <= self.range_max) & (obs_dists >= self.range_min)

            if np.any(valid_mask):
                closest_dist = float(np.min(obs_dists[valid_mask]))
                noisy_dist = np.clip(closest_dist + self.rng.normal(0, self.noise_std), self.range_min, self.range_max)
                ranges[b_idx] = noisy_dist

                # Reconstruct hit coordinate in world frame
                hit_angle_global = rtheta + b_angle
                hit_x = rx + noisy_dist * np.cos(hit_angle_global)
                hit_y = ry + noisy_dist * np.sin(hit_angle_global)
                detected_points.append((float(hit_x), float(hit_y)))

        return LaserScan(
            ranges=ranges,
            angles=self.beam_angles,
            point_cloud_global=detected_points,
        )
