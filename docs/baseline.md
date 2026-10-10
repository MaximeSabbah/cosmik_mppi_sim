# Baseline to preserve

All values below come from the exported files. `provenance/baseline-configs.json`
contains their original SHA-256 hashes; `scripts/doctor.sh` reports edits.
Source edits are deliberate handoff fixes; the seven tuning files are identical.

| Setting | Value | Authoritative file |
| --- | --- | --- |
| Native acceleration MPPI; CaT paper mode | `control_space: acc`, `cat_equation_mode: paper` | `storm/content/configs/mpc/franka_reacher_cat_real.yml` |
| Controller period / horizon / particles | 0.02 s / 50 / 1000 | same |
| Exploration covariance / beta | 1.0 / 5.0 | same |
| Acceleration limit / velocity scale | 50.0 / 1.0 | same |
| Goal weights, rotation / position | 1.0 / 10.0 | same |
| CaT human clearance threshold | 0.1 m | same, `mppi.cat_primitive_collision` |
| Sampling | multiple, 100% halton-knot, fixed samples, seed 0, knot scale 4 | same |
| Exact robot latency prediction | enabled | same |
| Target A / B positions, metres | (-0.5,-0.5,0.3) / (-0.5,+0.5,0.3) | `ros2_ws/src/storm_pick_place_demo/config/pick_place_task_manager_params.yaml` |
| Target orientations | quaternion XYZW (1,0,0,0), both | same |
| Arrival tolerance | position 0.06 m, rotation 2.0 rad | same |
| Maximum time on one physical target | 10 s; 0 disables | same, `target_timeout_sec` |
| Phase abort | disabled | same, `phase_timeout_sec` |
| Joint feedback gains, feedforward | supervisor's kp/kd, inverse dynamics | `ros2_ws/src/storm_agimus_bridge/config/storm_mppi_planner_params.yaml` |
| Low-level LFC parameters | supervisor's existing values | `ros2_ws/src/storm_pick_place_demo/config/linear_feedback_controller_params.yaml` |
| Static world | table and two poles | `storm/content/configs/gym/collision_primitives_3d.yml` |

Some historical YAML comments describe earlier tuning. Read the actual numeric
values. The `antithetic` flag is preserved verbatim; do not assume an unused flag
changes the selected legacy `multiple` sampler.

The simulator entry point is
`ros2_ws/src/storm_pick_place_demo/launch/pick_place_sim.launch.py`. Its explicit
overrides select the CaT profile, the `/rtcosmik/collision_markers` input, the
0.1-rad joint-reference guard used by the corresponding real launch, and simulation
time. The bridge YAML's legacy task default is overridden here.

The existing pick/place state machine is retained. All offsets and gripper target
widths are zero, giving alternating reaching with short gripper phases. The
arrival check can advance before 10 seconds. The timeout is per physical target,
not reset by a phase change at that same target. Some visits may still time out.
The first move starts from Gazebo's existing home configuration and may take a
timeout before the first arrival. We have not retuned this away.

In Gazebo, the robot URDF already defines `world -> fer_link0` as identity.
Do not add the real launch's inverse alias; that would form a TF cycle.
The environment markers represent controller obstacles. The synthetic human is
a ROS collision/visualization input, not a physical Gazebo human model.

## Changes made only in this handoff

- A clearly named, relocatable Gazebo launch selects the native CaT profile and
  explicit human input topic. It forces simulation even if hardware arguments are supplied.
- Gripper commands are remapped to Gazebo's `/gripper_action_controller/gripper_cmd`.
  The legacy grasp action is absent in Gazebo; its existing short timeout falls
  back to that command action. An associated warning is expected.
- Task deadlines use ROS time when `use_sim_time` is true; hardware clock behavior
  is retained otherwise. Restart the launch for a complete scene reset.
- A deterministic capsule input and isolated environment/build/check scripts were added.
- The Gazebo plugin's robot-description response wait now has bounded retries;
  RViz defaults were simplified for this reaching workspace.
- Root documentation and standalone prediction examples were added. Real launch
  files and historical validation data were omitted from the student demo package.

No MPPI algorithm or baseline tuning was changed to make the export run.
Restart the demo after editing YAML. Python/STORM files are loaded directly;
C++ or launch/install changes should be followed by `./scripts/build.sh`.
