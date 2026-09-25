"""Drive the AGV from its dock to the pallet station and record how it went.

The global path comes from A* on the inflated map. The local controller (DWA)
only sees obstacles through the simulated LiDAR, including the worker, whose
position it is never told. Writes samples/demo.gif, samples/path_planning.png
and results/summary.json.
"""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from core.factory_world import FactoryFloorWorld
from core.robot import DifferentialDriveRobot, RobotState, RobotConfig
from core.global_planner import AStarPlanner
from core.local_planner import DynamicWindowPlanner, DWAConfig
from core.lidar import PlanarLidar
from core.safety import SafetyFields
from core.visualizer import AGVVisualizer

START = (1.2, 1.5)
GOAL = (8.5, 8.2)
GOAL_TOLERANCE_M = 0.40
DT = 0.05
MAX_STEPS = 900          # 45 s of simulated time


def main():
    world = FactoryFloorWorld(width_m=10.0, height_m=10.0, resolution_m=0.10)
    robot_cfg = RobotConfig(radius_m=0.30, max_v_m_s=0.75, max_w_rad_s=1.20)
    robot = DifferentialDriveRobot(
        initial_state=RobotState(x=START[0], y=START[1], theta=0.0, v=0.0, w=0.0),
        config=robot_cfg,
    )
    lidar = PlanarLidar(num_beams=90, range_max_m=6.0)
    local_planner = DynamicWindowPlanner(robot_config=robot_cfg, dwa_config=DWAConfig(predict_time_s=1.6, dt=0.10))
    fields = SafetyFields()
    visualizer = AGVVisualizer(world)

    t0 = time.perf_counter()
    global_path = AStarPlanner(world.grid_map).plan(START, GOAL)
    plan_ms = (time.perf_counter() - t0) * 1000.0
    print(f"A* path: {len(global_path)} waypoints in {plan_ms:.1f} ms")

    trail = [(robot.state.x, robot.state.y)]
    frames = []
    wp_idx = 0
    goal_reached = False
    min_static, min_worker = np.inf, np.inf
    closest = None
    tightest_static = None
    stop_steps = 0
    warning_steps = 0
    contact_while_moving = False
    steps = 0
    worker_xs = []

    for steps in range(1, MAX_STEPS + 1):
        world.step_dynamic(DT)
        worker = world.dynamic_obstacles[0]
        worker_xs.append(worker.x)

        scan = lidar.scan(robot.state, world.get_all_obstacle_points())
        x, y = robot.state.x, robot.state.y

        # Clearances between the robot's body and anything it could hit.
        static_gap = world.static_clearance_m(x, y) - robot_cfg.radius_m
        worker_gap = world.worker_clearance_m(x, y) - robot_cfg.radius_m
        if static_gap < min_static:
            min_static, tightest_static = static_gap, (x, y)
        if worker_gap < min_worker:
            min_worker = worker_gap
            closest = ((x, y), (worker.x, worker.y))
        if worker_gap < 0 and robot.state.v > 0.01:
            contact_while_moving = True

        if np.hypot(GOAL[0] - x, GOAL[1] - y) < GOAL_TOLERANCE_M:
            goal_reached = True
            break

        while wp_idx < len(global_path) - 1 and np.hypot(global_path[wp_idx][0] - x, global_path[wp_idx][1] - y) < 0.65:
            wp_idx += 1

        # Safety fields first: an unmapped object close by caps or stops the robot.
        unmapped = fields.unmapped_points(scan.point_cloud_global, world.static_clearance_m)
        v_limit, _ = fields.speed_limit(x, y, robot_cfg.radius_m, unmapped)
        if v_limit == 0.0:
            v_cmd, w_cmd, rollout = 0.0, 0.0, [(x, y)]
            stop_steps += 1
        else:
            if v_limit is not None:
                warning_steps += 1
            v_cmd, w_cmd, rollout = local_planner.plan_velocity_command(
                current_state=robot.state,
                target_waypoint=global_path[wp_idx],
                obstacles=scan.point_cloud_global,
                v_limit=v_limit,
            )
        robot.step(v_cmd, w_cmd, DT)
        trail.append((robot.state.x, robot.state.y))

        if steps % 3 == 0:
            frames.append(visualizer.render_frame_pil(
                robot_state=robot.state,
                global_path=global_path,
                scan=scan,
                target_goal=GOAL,
                trail_pts=trail,
                dwa_rollout=rollout,
            ))

    samples = PROJECT_ROOT / "samples"
    samples.mkdir(exist_ok=True)
    frames[0].save(str(samples / "demo.gif"), save_all=True, append_images=frames[1:], duration=120, loop=0)
    visualizer.plot_mission_summary(
        global_path=global_path,
        executed_trail=trail,
        target_goal=GOAL,
        save_path=str(samples / "path_planning.png"),
        worker_walk=((min(worker_xs), worker.y), (max(worker_xs), worker.y)),
        closest_approach=closest,
    )

    driven = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trail, trail[1:])))
    straight = float(np.hypot(GOAL[0] - START[0], GOAL[1] - START[1]))
    summary = {
        "goal_reached": goal_reached,
        "time_to_goal_s": round(steps * DT, 2) if goal_reached else None,
        "astar_planning_ms": round(plan_ms, 1),
        "path_length_m": {
            "driven": round(driven, 2),
            "straight_line": round(straight, 2),
            "ratio": round(driven / straight, 2),
        },
        "min_clearance_m": {
            "static_obstacles": round(float(min_static), 3),
            "static_obstacles_at_xy": [round(float(c), 2) for c in tightest_static],
            "worker": round(float(min_worker), 3),
        },
        "collision": bool(min_static < 0 or min_worker < 0),
        "contact_with_worker_while_moving": contact_while_moving,
        "time_in_protective_stop_s": round(stop_steps * DT, 2),
        "time_at_warning_speed_s": round(warning_steps * DT, 2),
        "control_period_s": DT,
    }
    out = PROJECT_ROOT / "results" / "summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
