"""A* global path planner on configuration-space grid maps with path smoothing."""
import heapq
from typing import List, Tuple, Optional
import numpy as np

from core.grid_map import OccupancyGridMap


class AStarPlanner:
    """8-connected grid A* path planner finding optimal shortest collision-free routes."""

    # 8-connected motion deltas: (d_row, d_col, cost)
    MOTIONS = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    def __init__(self, grid_map: OccupancyGridMap):
        self.grid_map = grid_map

    def plan(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan shortest path from start (x, y) to goal (x, y) in meters."""
        start_r, start_c = self.grid_map.world_to_grid(*start_pos)
        goal_r, goal_c = self.grid_map.world_to_grid(*goal_pos)

        if not self.grid_map.is_in_bounds(start_r, start_c) or not self.grid_map.is_in_bounds(goal_r, goal_c):
            raise ValueError("Start or Goal position is outside grid boundaries.")

        if self.grid_map.inflated_grid[start_r, start_c]:
            raise ValueError("Start position is inside an inflated obstacle collision zone.")
        if self.grid_map.inflated_grid[goal_r, goal_c]:
            raise ValueError("Goal position is inside an inflated obstacle collision zone.")

        # Priority Queue: (f_score, g_score, (row, col))
        open_set = []
        h0 = np.hypot(goal_r - start_r, goal_c - start_c)
        heapq.heappush(open_set, (h0, 0.0, (start_r, start_c)))

        came_from = {}
        g_scores = {(start_r, start_c): 0.0}

        found = False
        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == (goal_r, goal_c):
                found = True
                break

            for dr, dc, step_cost in self.MOTIONS:
                nr, nc = current[0] + dr, current[1] + dc

                if not self.grid_map.is_in_bounds(nr, nc):
                    continue
                if self.grid_map.inflated_grid[nr, nc]:
                    continue

                tentative_g = current_g + step_cost
                neighbor = (nr, nc)

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    h = np.hypot(goal_r - nr, goal_c - nc)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        if not found:
            raise RuntimeError(f"No collision-free path found from {start_pos} to {goal_pos}")

        # Reconstruct path
        path_grid = [(goal_r, goal_c)]
        curr = (goal_r, goal_c)
        while curr in came_from:
            curr = came_from[curr]
            path_grid.append(curr)
        path_grid.reverse()

        # Convert to world coordinates
        path_world = [self.grid_map.grid_to_world(r, c) for r, c in path_grid]
        smoothed = self.smooth_path(path_world)
        return self.densify_path(smoothed, max_spacing_m=0.35)

    def densify_path(self, path: List[Tuple[float, float]], max_spacing_m: float = 0.35) -> List[Tuple[float, float]]:
        """Interpolate intermediate waypoints so waypoint spacing does not exceed max_spacing_m."""
        if len(path) <= 1:
            return path
        dense = []
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            num_pts = max(2, int(np.ceil(dist / max_spacing_m)))
            xs = np.linspace(p1[0], p2[0], num_pts)
            ys = np.linspace(p1[1], p2[1], num_pts)
            for x, y in zip(xs[:-1], ys[:-1]):
                dense.append((float(x), float(y)))
        dense.append(path[-1])
        return dense

    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Path pruning / string pulling: bypass collinear and redundant intermediate waypoints."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Look ahead as far as possible with line-of-sight check
            farthest = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if self._line_of_sight(path[i], path[j]):
                    farthest = j
                    break
            smoothed.append(path[farthest])
            i = farthest

        return smoothed

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float], num_samples: int = 20) -> bool:
        """Check if straight line segment between p1 and p2 is collision-free."""
        xs = np.linspace(p1[0], p2[0], num_samples)
        ys = np.linspace(p1[1], p2[1], num_samples)
        for x, y in zip(xs, ys):
            if not self.grid_map.is_free(x, y, use_inflated=True):
                return False
        return True
