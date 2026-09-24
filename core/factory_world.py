"""Factory floor environment with static layout machinery and dynamic workers/obstacles."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from core.grid_map import OccupancyGridMap


@dataclass
class DynamicObstacle:
    name: str
    x: float
    y: float
    vx: float
    vy: float
    radius_m: float = 0.35


class FactoryFloorWorld:
    """Industrial manufacturing floor with machine cells, pallet racks, and moving obstacles."""

    def __init__(self, width_m: float = 10.0, height_m: float = 10.0, resolution_m: float = 0.10):
        self.width_m = width_m
        self.height_m = height_m
        self.grid_map = OccupancyGridMap(width_m=width_m, height_m=height_m, resolution_m=resolution_m)

        # Setup standard factory cell layout
        self._build_static_layout()
        self.grid_map.compute_inflation(robot_radius_m=0.30, safety_margin_m=0.10)

        # Dynamic obstacles (e.g. human worker crossing aisle)
        self.dynamic_obstacles: List[DynamicObstacle] = [
            DynamicObstacle(name="worker", x=4.8, y=6.2, vx=0.0, vy=-0.25, radius_m=0.35),
        ]

    def _build_static_layout(self) -> None:
        """Define machine footprints, storage racks, and boundary walls."""
        # 1. Boundary walls
        self.grid_map.set_obstacle_rect(0.0, 0.0, 10.0, 0.2)
        self.grid_map.set_obstacle_rect(0.0, 9.8, 10.0, 10.0)
        self.grid_map.set_obstacle_rect(0.0, 0.0, 0.2, 10.0)
        self.grid_map.set_obstacle_rect(9.8, 0.0, 10.0, 10.0)

        # 2. CNC Machining Center 1
        self.grid_map.set_obstacle_rect(1.5, 4.5, 3.2, 7.5)

        # 3. High-Density Pallet Racks
        self.grid_map.set_obstacle_rect(6.2, 4.0, 8.8, 5.5)

        # 4. Assembly & Inspection Island
        self.grid_map.set_obstacle_rect(4.0, 1.5, 5.8, 3.2)

    def step_dynamic(self, dt: float) -> None:
        """Advance dynamic obstacles in time."""
        for obs in self.dynamic_obstacles:
            obs.x += obs.vx * dt
            obs.y += obs.vy * dt
            # Bounce back at boundaries
            if obs.y < 3.2 or obs.y > 7.0:
                obs.vy *= -1.0

    def get_all_obstacle_points(self, resolution_m: float = 0.20) -> List[Tuple[float, float]]:
        """Sample all static and dynamic surface points for LiDAR and DWA clearance checks."""
        points: List[Tuple[float, float]] = []

        # Extract static obstacle surface cells
        occupied_indices = np.argwhere(self.grid_map.grid)
        # Subsample for computational speed
        subsampled = occupied_indices[::4]
        for r, c in subsampled:
            x, y = self.grid_map.grid_to_world(r, c)
            points.append((x, y))

        # Add dynamic obstacles (perimeter circle points)
        for obs in self.dynamic_obstacles:
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                ox = obs.x + obs.radius_m * np.cos(angle)
                oy = obs.y + obs.radius_m * np.sin(angle)
                points.append((ox, oy))

        return points
