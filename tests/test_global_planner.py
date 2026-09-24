"""Unit tests for A* global path planner."""
import numpy as np
import pytest
from core.grid_map import OccupancyGridMap
from core.global_planner import AStarPlanner


def test_free_space_path():
    grid = OccupancyGridMap(width_m=5.0, height_m=5.0, resolution_m=0.1)
    grid.compute_inflation(robot_radius_m=0.2)
    planner = AStarPlanner(grid)

    path = planner.plan((0.5, 0.5), (4.0, 4.0))
    assert len(path) >= 2
    # Start and end match
    np.testing.assert_allclose(path[0], (0.5, 0.5), atol=0.15)
    np.testing.assert_allclose(path[-1], (4.0, 4.0), atol=0.15)


def test_obstacle_avoidance():
    grid = OccupancyGridMap(width_m=6.0, height_m=6.0, resolution_m=0.1)
    # Add wall in middle with passage at top
    grid.set_obstacle_rect(2.8, 0.0, 3.2, 4.0)
    grid.compute_inflation(robot_radius_m=0.2, safety_margin_m=0.1)
    planner = AStarPlanner(grid)

    path = planner.plan((1.0, 2.0), (5.0, 2.0))
    assert len(path) >= 3

    # Path must detour around the obstacle (Y > 4.0)
    max_y = max(p[1] for p in path)
    assert max_y > 4.0
