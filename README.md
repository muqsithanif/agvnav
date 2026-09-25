# agvnav

A 2D simulation of a factory AGV driving from its dock to a pallet station. An A* planner computes a global path around the machines, and a Dynamic Window Approach (DWA) controller follows it using only what a simulated LiDAR sees. That includes a worker who walks into the robot's aisle, whose position the robot is never told.

![Navigation run](samples/demo.gif)

*The robot goes through the doorway between the two CNC cells, rounds the end of the pallet racks, slows as the worker walks towards its path, stops while the worker is at the rack face, and continues once the worker walks away.*

---

## The floor

The floor is 10 m × 10 m, with walls, two CNC cells with a 1.4 m doorway between them, and a row of pallet racks. The straight line from the dock at (1.2, 1.5) to the goal at (8.5, 8.2) runs through CNC cell A and the racks, so the robot has to take the doorway and the aisle at the end of the racks. The worker waits at the east end of that aisle, walks to the rack face as the robot arrives, works there for three seconds, and walks back. The worker follows this schedule whatever the robot does.

## How it drives

**Global plan.** A* runs on an 8-connected occupancy grid with 0.1 m cells. Obstacles are inflated by the robot radius plus 0.2 m, and the path is shortened by line-of-sight smoothing. The inflation is deliberately larger than the margin the local planner keeps. With the same margin, the smoothed path cut corners closer than the local planner would accept, and the robot stopped in front of the doorway.

**Local control.** DWA samples forward and turning speeds reachable within one control step, rolls each pair forward 1.6 s, and scores it on heading to the next waypoint, clearance and speed. A trajectory is admissible only if it keeps 0.1 m beyond the robot radius. Once the robot is already inside that margin, a trajectory is still allowed as long as it does not get any closer. Without that exception every option is rejected, including turning on the spot, and the robot deadlocks.

**Perception.** The simulated LiDAR has 90 beams over 180°, a 6 m range and 1.5 cm noise. DWA plans against the LiDAR hits only, not the map.

**Safety fields.** DWA treats obstacles as static over its horizon, so on its own it will drive into the path of someone walking towards it. An earlier run did exactly that. The robot therefore slows to 0.25 m/s when an object that is not on the map comes within 1.2 m of its body, and stops when one comes within 0.5 m, the way AGV safety-scanner fields work. A LiDAR hit counts as the map if it lies within 0.12 m of a mapped wall, machine or rack. Without that check the fields would fire in every doorway.

---

## Results

All figures below are from `results/summary.json`.

| | |
|---|---|
| Goal reached | yes, in 33.2 s |
| Path driven / straight line | 10.57 m / 9.91 m (1.07×) |
| Closest approach to a wall, machine or rack | 0.064 m |
| Closest approach to the worker | 0.156 m |
| Collision | none |
| Time stopped by the protective field | 5.9 s |
| Time slowed by the warning field | 7.5 s |
| A* planning time | 30 ms |

Clearances are measured between the robot's body and the exact obstacle outlines, not grid cells. The tightest point, 6.4 cm, is at the corner of the pallet racks, where the robot turns into the aisle while the worker is approaching.

## Limits

- **2D kinematics only.** There is no wheel slip, no localization error (the robot knows its pose exactly), and no SLAM.
- **The LiDAR is modelled** as the nearest obstacle point inside each beam's angular cone, not a true ray-cast against geometry.
- **The worker's behaviour is scripted.** A person who changes direction unpredictably, or walks straight at the robot from behind the LiDAR's field of view, is not covered.
- **One scenario.** Performance across many layouts and worker timings has not been measured.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_navigation.py
pytest -q
```

The script writes `samples/demo.gif`, `samples/path_planning.png` and `results/summary.json`. It takes about a minute, most of it rendering the GIF.

## Tests

There are fourteen tests. These are the ones worth naming:

- **Inside the margin, the robot can still turn away.** This is the deadlock described above.
- **Mapped surfaces do not trigger the safety fields, and unmapped objects do.**
- **An obstacle rectangle marks only the cells whose centres fall inside it.** The earlier rasterisation made every obstacle up to one cell larger than drawn.
- **A* detours round a wall, and the DWA brakes or turns for an obstacle directly ahead.**
