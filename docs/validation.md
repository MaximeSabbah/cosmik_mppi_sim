# Validation record — 2026-10-09

The independent export was built against `/opt/ros/humble`, with the original
`/workspace/ros2_ws/install` overlay removed from the environment. The doctor check
verified that STORM, the ROS bridge, task manager, controller and vendored robot
packages resolve inside the handoff. CUDA execution succeeded on the RTX 5000 Ada.
All seven baseline configuration hashes matched.

## Completed checks

| Check | Evidence |
| --- | --- |
| Fresh build of 13 ROS packages | `validation/build.log` |
| Rebuild after simulator startup fix | `validation/build-after-startup-fix.log` |
| 5 prediction-contract tests and 12 task/clock tests | `validation/unit-tests.log` |
| Package isolation, CUDA and baseline config hashes | `validation/environment.json` |
| ROS/Gazebo reaching without human input | `validation/no-human/result.json` and `launch.log` |
| ROS/Gazebo reaching with synthetic capsule input | `validation/moving-capsule/result.json` and `launch.log` |
| Apt dependency resolution only (no installation performed) | `validation/apt-resolution.log` |
| Fresh build from archive extracted under `/tmp/cosmik-mppi-relocation/cosmik-mppi-sim` | `validation/relocated-build.log` |
| Relocated imports/CUDA/config checks and 17 tests | `validation/relocated-environment.json`, `validation/relocated-tests.log` |
| Relocated moving-capsule run after startup fix | `validation/relocated-moving-capsule/result.json` and `launch.log` |

The first run covered 45 simulated seconds after the first goal; the capsule run
covered 50 seconds. The default period, sampling, costs, gains and targets were
the supervisor's values.

| Result | No human | Moving capsule |
| --- | ---: | ---: |
| Target visits | 5 | 7 |
| Spatial arrivals at A / B | 1 / 2 | 1 / 3 |
| Timeout events | 3 | 3 |
| Minimum measured target position error A / B | 15.7 / 1.5 mm | 44.7 / 2.1 mm |
| Control messages received | 1,604 | 1,836 |
| Human frames received | 0 | 1,545 |
| Throttled controller geometry-update log entries | 0 | 20 |
| Nonfinite state or command detected | no | no |

An arrival requires both the 6-cm position threshold and 2-rad orientation
threshold. Arrival count is once per physical-target visit, not once per task
phase. Timeout counts therefore cannot be interpreted as successes. Minimum
position error alone does not imply the orientation criterion passed at that instant.

The controller logged the selected native CaT profile and initialized one dynamic
capsule slot containing five spheres. This verifies geometry ingestion and command
generation; it does not quantify minimum human clearance or establish predictive
avoidance. Those are explicit research evaluation tasks in `prediction.md`.

## GUI startup finding

A GUI run exposed a lost robot-description service response in the vendored Gazebo
plugin. The original unconditional future wait then stalled startup. The handoff
uses a two-second response timeout with at most 15 requests, an interrupt check,
and an explicit failure instead of an infinite wait. This changes startup only,
not physics or control. One retry was exercised successfully in the retest.

RViz uses a compact window, a dark background, a 30-Hz display rate, and a view
centred on the reaching workspace. Its default displays show robot geometry,
target markers, the current human capsule, static collision geometry and the MPC
reference. Additional debug displays remain available but disabled.
The initialized view was inspected directly: the complete robot, target markers
and moving capsule are visible with TF status OK. Evidence is in
`validation/rviz/startup-result.json`, `validation/rviz/launch.log` and
`validation/rviz/window.png`. The retained `startup-before-fix.log` records the
earlier service stall.

## Reproducibility boundary

The source revisions, local edits, assets, input scenario and tuning are captured.
The synthetic motion is an explicit function of Gazebo time. ROS scheduling,
asynchronous optimization and GPU execution can change exact trajectories and
the number of timeout visits; do not claim bit-identical robot behavior.

The relocated copy built all 13 packages in 2 min 10 s. Its 50-s capsule run
passed: 6 target visits, 1 spatial arrival at A and 2 at B, 4 timeout events,
1,539 capsule frames, 1,939 control messages and no nonfinite states/commands.
The selected CaT profile and 21 throttled dynamic-geometry update messages were
verified. Every checked project import resolved under the new extraction path.
This is a second independent source build on the same computer, not a second-machine test.

These runs use the author's installed ROS/OS/GPU stack. On 2026-10-10 a fresh Docker image was also built and tested through SSH on
the kurai host. A different physical computer has not been tested. The first
image exposed a missing Hydra import; the corrected recipe and deeper preflight
passed. Evidence: `validation/docker-20261010/` and `provenance/docker-image.json`.
The source snapshot was checksum-verified after extraction before building;
`scripts/verify-release.py` performs the same check on the final release.

No real robot was connected or commanded during this packaging work. The original
controller workspace and its tuning files were not edited.

## Fresh Docker validation — 2026-10-10

All 13 ROS packages built in the image, all 17 tests passed, and CUDA 12.4 executed
on the host RTX 5000 Ada. The image was run with only a results directory mounted;
no existing workspace or installed ROS overlay was mounted.

- No-human run, 50 simulated seconds: 3 measured arrivals at A and 3 at B;
  2 timeout switches; 1,359 control messages.
- Moving-capsule run, 50 simulated seconds: 2 measured arrivals at A and 3 at B;
  3 timeout switches; 1,540 capsule frames; 1,728 control messages.
- All seven baseline config hashes still match. No nonfinite state/command was detected.

The host-side GUI helper recorded two supervisor sessions on 2026-10-10. The
supervisor subsequently reported "ok it seems quite ok". This is qualitative GUI
feedback, recorded separately from the automated arrival checks; it is not a
measurement of human clearance. Legacy shutdown traces are retained in the test
logs separately from running-period evidence.
