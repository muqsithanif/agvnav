"""Speed and separation monitoring, configured the way AGV safety scanners are.

The local planner treats every obstacle as static over its 1.6 s horizon, so
on its own it will drive into the path of a person walking towards it. Safety
scanners handle this with fields around the vehicle: something inside the
warning field slows it down, something inside the protective field stops it.

Only objects that are not on the map count. A LiDAR hit within a few
centimetres of a mapped wall, machine or rack is the map itself; anything
else, such as a person, triggers the fields. Otherwise the fields would fire
on every doorway.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass
class SafetyFields:
    protective_m: float = 0.50        # stop if an unmapped object is this close to the robot's body
    warning_m: float = 1.20           # slow down inside this
    warning_speed_m_s: float = 0.25
    map_tolerance_m: float = 0.12     # grid resolution plus LiDAR noise

    def unmapped_points(
        self,
        points: List[Tuple[float, float]],
        static_clearance: Callable[[float, float], float],
    ) -> List[Tuple[float, float]]:
        return [p for p in points if static_clearance(p[0], p[1]) > self.map_tolerance_m]

    def speed_limit(
        self, x: float, y: float, radius_m: float, unmapped: List[Tuple[float, float]]
    ) -> Tuple[Optional[float], float]:
        """Speed cap from the nearest unmapped object, and the gap to it.

        0.0 inside the protective field, the warning speed inside the warning
        field, and None (no cap) beyond it.
        """
        if not unmapped:
            return None, float("inf")
        gap = min(float(np.hypot(px - x, py - y)) for px, py in unmapped) - radius_m
        if gap < self.protective_m:
            return 0.0, gap
        if gap < self.warning_m:
            return self.warning_speed_m_s, gap
        return None, gap
