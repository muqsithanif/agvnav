"""Factory floor: walls, machine cells, a rack row, and a worker crossing an aisle."""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.ndimage import binary_erosion

from core.grid_map import OccupancyGridMap


@dataclass
class DynamicObstacle:
    """A person following a fixed schedule: walk to a point, wait there, walk on.

    The schedule is a list of (x, y, wait_s) stops. The person walks between
    stops at `speed_m_s` and does not react to the robot, so any avoiding has
    to be done by the robot.
    """
    name: str
    x: float
    y: float
    schedule: List[Tuple[float, float, float]]
    speed_m_s: float = 0.45
    radius_m: float = 0.35
    vx: float = 0.0
    vy: float = 0.0
    _stop: int = 0
    _waited: float = 0.0

    def step(self, dt: float) -> None:
        if self._stop >= len(self.schedule):
            self.vx = self.vy = 0.0
            return
        tx, ty, wait_s = self.schedule[self._stop]
        dx, dy = tx - self.x, ty - self.y
        dist = float(np.hypot(dx, dy))
        if dist > 1e-6:
            move = min(self.speed_m_s * dt, dist)
            self.vx, self.vy = dx / dist * self.speed_m_s, dy / dist * self.speed_m_s
            self.x += dx / dist * move
            self.y += dy / dist * move
            return
        self.vx = self.vy = 0.0
        self._waited += dt
        if self._waited >= wait_s:
            self._stop += 1
            self._waited = 0.0


@dataclass(frozen=True)
class StaticObstacle:
    name: str
    kind: str      # "wall", "machine" or "rack"
    x0: float
    y0: float
    x1: float
    y1: float


class FactoryFloorWorld:
    """A 10 m x 10 m floor laid out so the straight line from dock to goal is blocked.

    Two CNC cells leave a 1.4 m doorway between them, and a rack row closes
    the direct route above it, so the robot has to go through the doorway and
    round the end of the racks. A worker walks back and forth across the aisle
    beside the racks, which the robot has to pass. The worker waits at the
    east end, walks to the rack face as the robot arrives, works there for
    three seconds, and walks back.
    """

    def __init__(self, width_m: float = 10.0, height_m: float = 10.0, resolution_m: float = 0.10):
        self.width_m = width_m
        self.height_m = height_m
        self.grid_map = OccupancyGridMap(width_m=width_m, height_m=height_m, resolution_m=resolution_m)

        self.static_obstacles: List[StaticObstacle] = [
            StaticObstacle("wall", "wall", 0.0, 0.0, 10.0, 0.2),
            StaticObstacle("wall", "wall", 0.0, 9.8, 10.0, 10.0),
            StaticObstacle("wall", "wall", 0.0, 0.0, 0.2, 10.0),
            StaticObstacle("wall", "wall", 9.8, 0.0, 10.0, 10.0),
            StaticObstacle("CNC cell A", "machine", 1.5, 3.0, 4.3, 4.2),
            StaticObstacle("CNC cell B", "machine", 5.7, 3.0, 8.5, 4.2),
            StaticObstacle("Pallet racks", "rack", 3.0, 6.0, 7.5, 7.0),
        ]
        for obs in self.static_obstacles:
            self.grid_map.set_obstacle_rect(obs.x0, obs.y0, obs.x1, obs.y1)
        # The plan keeps 0.2 m beyond the robot radius, more than the 0.1 m the
        # local planner insists on, so the local planner can always follow it.
        self.grid_map.compute_inflation(robot_radius_m=0.30, safety_margin_m=0.20)

        # Surface cells only: occupied cells with a free neighbour. These are
        # what the LiDAR can see, dense enough that no beam slips through.
        surface = self.grid_map.grid & ~binary_erosion(self.grid_map.grid, border_value=1)
        self._surface_points = [self.grid_map.grid_to_world(r, c) for r, c in np.argwhere(surface)]

        self.dynamic_obstacles: List[DynamicObstacle] = [
            DynamicObstacle(
                name="worker", x=9.4, y=6.5,
                schedule=[(9.4, 6.5, 17.0), (7.95, 6.5, 3.0), (9.4, 6.5, float("inf"))],
            ),
        ]

    def step_dynamic(self, dt: float) -> None:
        """Advance every person along their schedule."""
        for obs in self.dynamic_obstacles:
            obs.step(dt)

    def get_all_obstacle_points(self) -> List[Tuple[float, float]]:
        """Surface points of the static layout plus the worker's outline, for the LiDAR."""
        points = list(self._surface_points)
        for obs in self.dynamic_obstacles:
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                points.append((obs.x + obs.radius_m * np.cos(angle), obs.y + obs.radius_m * np.sin(angle)))
        return points

    def static_clearance_m(self, x: float, y: float) -> float:
        """Exact distance from a point to the nearest wall, machine or rack."""
        gaps = []
        for o in self.static_obstacles:
            dx = max(o.x0 - x, 0.0, x - o.x1)
            dy = max(o.y0 - y, 0.0, y - o.y1)
            gaps.append(np.hypot(dx, dy))
        return float(min(gaps))

    def worker_clearance_m(self, x: float, y: float) -> float:
        """Distance from a point to the edge of the nearest worker."""
        return min(np.hypot(x - o.x, y - o.y) - o.radius_m for o in self.dynamic_obstacles)
