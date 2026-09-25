"""Factory layout visualizer and animated GIF exporter."""
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
from PIL import Image

from core.factory_world import FactoryFloorWorld
from core.robot import RobotState
from core.lidar import LaserScan


KIND_COLORS = {"machine": ("#4a5568", "#2d3748"), "rack": ("#5a67d8", "#434190"), "wall": ("#2d3748", "#2d3748")}


class AGVVisualizer:
    """Renders 2D top-down factory shop floor simulations and mission trajectories."""

    def _draw_layout(self, ax, labels: bool = True) -> None:
        """Draw the world's static obstacles, so the picture always matches the map."""
        for obs in self.world.static_obstacles:
            face, edge = KIND_COLORS[obs.kind]
            ax.add_patch(Rectangle((obs.x0, obs.y0), obs.x1 - obs.x0, obs.y1 - obs.y0,
                                   facecolor=face, edgecolor=edge, lw=1.2))
            if labels and obs.kind != "wall":
                ax.text((obs.x0 + obs.x1) / 2, (obs.y0 + obs.y1) / 2, obs.name,
                        color="white", fontsize=8, ha="center", va="center", weight="bold")

    def __init__(self, world: FactoryFloorWorld):
        self.world = world

    def render_frame_pil(
        self,
        robot_state: RobotState,
        global_path: List[Tuple[float, float]],
        scan: LaserScan,
        target_goal: Tuple[float, float],
        trail_pts: List[Tuple[float, float]],
        dwa_rollout: Optional[List[Tuple[float, float]]] = None,
    ) -> Image.Image:
        """Render a single frame as PIL Image."""
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.set_xlim([0, self.world.width_m])
        ax.set_ylim([0, self.world.height_m])
        ax.set_aspect("equal")
        ax.set_facecolor("#f2f4f7")

        # 1. Draw Inflated Safety Boundaries
        inflated_mask = self.world.grid_map.inflated_grid
        ax.imshow(
            inflated_mask,
            origin="lower",
            extent=[0, self.world.width_m, 0, self.world.height_m],
            cmap="Blues",
            alpha=0.18,
        )

        # 2. Walls, machine cells and racks
        self._draw_layout(ax)

        # 3. Global A* Path
        if len(global_path) > 1:
            g_xs, g_ys = zip(*global_path)
            ax.plot(g_xs, g_ys, color="#00b4d8", linestyle="--", lw=2, label="Global A* Path")

        # 4. Trajectory History Trail
        if len(trail_pts) > 1:
            t_xs, t_ys = zip(*trail_pts)
            ax.plot(t_xs, t_ys, color="#f77f00", lw=2.5, alpha=0.8, label="AGV Trajectory")

        # 5. DWA Local Rollout
        if dwa_rollout and len(dwa_rollout) > 1:
            d_xs, d_ys = zip(*dwa_rollout)
            ax.plot(d_xs, d_ys, color="#06d6a0", lw=2, label="DWA Rollout")

        # 6. LiDAR Laser Fan
        rx, ry, rtheta = robot_state.x, robot_state.y, robot_state.theta
        for hit_x, hit_y in scan.point_cloud_global[::3]:
            ax.plot([rx, hit_x], [ry, hit_y], color="#38b000", lw=0.6, alpha=0.35)
            ax.plot(hit_x, hit_y, "o", color="#d00000", markersize=2)

        # 7. Dynamic Obstacle (Worker)
        for obs in self.world.dynamic_obstacles:
            ax.add_patch(Circle((obs.x, obs.y), obs.radius_m, facecolor="#e63946", edgecolor="#9b2226", lw=1.5))
            ax.text(obs.x, obs.y + 0.45, "Worker", color="#9b2226", fontsize=7, ha="center", weight="bold")

        # 8. Goal Pallet Station
        gx, gy = target_goal
        ax.plot(gx, gy, "P", color="#2a9d8f", markersize=14, label="Goal Station")
        ax.text(gx, gy + 0.45, "Pallet Drop", color="#2a9d8f", fontsize=8, ha="center", weight="bold")

        # 9. AGV Chassis & Orientation Arrow
        ax.add_patch(Circle((rx, ry), 0.30, facecolor="#1d3557", edgecolor="#457b9d", lw=2))
        arrow_len = 0.45
        ax.arrow(
            rx, ry,
            arrow_len * np.cos(rtheta), arrow_len * np.sin(rtheta),
            head_width=0.18, head_length=0.15, fc="#e63946", ec="#e63946"
        )

        ax.set_title(
            f"AGV navigation | speed {robot_state.v:.2f} m/s",
            fontsize=9, pad=8, weight="bold"
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        pil_img = Image.fromarray(rgba)
        plt.close(fig)
        return pil_img

    def plot_mission_summary(
        self,
        global_path: List[Tuple[float, float]],
        executed_trail: List[Tuple[float, float]],
        target_goal: Tuple[float, float],
        save_path: str,
        worker_walk: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        closest_approach: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:
        """Plot the A* plan against the path the robot actually drove."""
        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        ax.set_xlim([0, self.world.width_m])
        ax.set_ylim([0, self.world.height_m])
        ax.set_aspect("equal")
        ax.set_facecolor("#f8f9fa")

        self._draw_layout(ax)
        if worker_walk is not None:
            (wx0, wy0), (wx1, wy1) = worker_walk
            ax.plot([wx0, wx1], [wy0, wy1], color="#e63946", lw=6, alpha=0.25, solid_capstyle="round",
                    label="Worker's walk")
        if closest_approach is not None:
            (rx, ry), (wx, wy) = closest_approach
            ax.add_patch(Circle((wx, wy), 0.35, facecolor="#e63946", edgecolor="#9b2226", alpha=0.8))
            ax.plot([rx, wx], [ry, wy], color="#9b2226", lw=1.2, linestyle=":", label="Closest approach")

        g_xs, g_ys = zip(*global_path)
        ax.plot(g_xs, g_ys, color="#00b4d8", linestyle="--", lw=2.5, label="A* plan")

        t_xs, t_ys = zip(*executed_trail)
        ax.plot(t_xs, t_ys, color="#f77f00", lw=3.0, label="Driven path (DWA)")

        ax.plot(global_path[0][0], global_path[0][1], "s", color="green", markersize=10, label="Start")
        ax.plot(target_goal[0], target_goal[1], "P", color="red", markersize=12, label="Goal")

        ax.set_title("A* plan and the path driven by the DWA controller", fontsize=10, weight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
