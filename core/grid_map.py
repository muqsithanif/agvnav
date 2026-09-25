"""2D Factory Occupancy Grid representation and configuration-space obstacle inflation."""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.ndimage import binary_dilation


class OccupancyGridMap:
    """Discretized 2D grid map with C-space obstacle inflation layer."""

    def __init__(
        self,
        width_m: float = 10.0,
        height_m: float = 10.0,
        resolution_m: float = 0.10,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        self.width_m = width_m
        self.height_m = height_m
        self.res = resolution_m
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.cols = int(np.round(width_m / self.res))
        self.rows = int(np.round(height_m / self.res))

        # Binary occupancy grid: False (free), True (occupied)
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)
        self.inflated_grid = np.zeros((self.rows, self.cols), dtype=bool)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous world coordinates (meters) to discrete grid indices (row, col)."""
        col = int(np.floor((x - self.origin_x) / self.res))
        row = int(np.floor((y - self.origin_y) / self.res))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid cell (row, col) to continuous world coordinates at cell center."""
        x = self.origin_x + (col + 0.5) * self.res
        y = self.origin_y + (row + 0.5) * self.res
        return x, y

    def is_in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def set_obstacle_rect(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        """Mark the cells whose centres fall inside the rectangle.

        Marking every cell the rectangle touches, as an earlier version did,
        makes each obstacle up to one cell larger than drawn.
        """
        c_start = int(np.ceil((x_min - self.origin_x) / self.res - 0.5))
        c_end = int(np.floor((x_max - self.origin_x) / self.res - 0.5))
        r_start = int(np.ceil((y_min - self.origin_y) / self.res - 0.5))
        r_end = int(np.floor((y_max - self.origin_y) / self.res - 0.5))
        self.grid[max(0, r_start):min(self.rows, r_end + 1), max(0, c_start):min(self.cols, c_end + 1)] = True

    def compute_inflation(self, robot_radius_m: float, safety_margin_m: float = 0.10) -> None:
        """Dilate obstacles by robot physical footprint to construct Configuration Space (C-space)."""
        total_radius = robot_radius_m + safety_margin_m
        cell_radius = int(np.ceil(total_radius / self.res))

        # Circular structuring element
        y, x = np.ogrid[-cell_radius:cell_radius + 1, -cell_radius:cell_radius + 1]
        structuring_element = (x ** 2 + y ** 2) <= (cell_radius ** 2)

        self.inflated_grid = binary_dilation(self.grid, structure=structuring_element)

    def is_free(self, x: float, y: float, use_inflated: bool = True) -> bool:
        """Check if world coordinate is collision-free."""
        row, col = self.world_to_grid(x, y)
        if not self.is_in_bounds(row, col):
            return False
        active_grid = self.inflated_grid if use_inflated else self.grid
        return not active_grid[row, col]
