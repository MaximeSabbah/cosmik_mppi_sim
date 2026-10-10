# COSMIK + CaT MPPI simulation handoff

This is the supervisor's 2026-10-09 reaching baseline, exported into an independent
ROS 2 Humble / Gazebo Fortress workspace. The robot alternates between two targets
while CaT MPPI receives moving human obstacle geometry. Start here before adding
human-motion prediction.

The seven controller/task/environment configuration files are unchanged from the
supervisor's workspace. The entry point explicitly selects
`franka_reacher_cat_real.yml` (the name is retained to identify the tuning).
This launch always uses Gazebo.

Repository: [MaximeSabbah/cosmik_mppi_sim](https://github.com/MaximeSabbah/cosmik_mppi_sim).
Start from the `v1.0.0` release for the complete source and GitHub image installer.
The earlier `baseline-2026-10-10` tag records the same controller tuning.
The complete [student setup and sharing guide](docs/sharing.md) covers getting the
Docker image, first acceptance checks and editing the source.

**What runs the rollouts?** STORM computes candidate trajectories with its own
PyTorch/CUDA joint-motion and URDF kinematics model. Gazebo simulates the robot
executing the chosen commands; RViz displays the scene. Isaac Sim is not used in
this demo. See [the architecture explanation](docs/architecture.md).

## First run

For a new workstation, use the tested container: Linux x86-64, an NVIDIA GPU,
Docker with Compose, and NVIDIA Container Toolkit. ROS/Gazebo/Python dependencies
are already inside the supplied image. Follow [setup.md](docs/setup.md).

After cloning the repository or extracting the source, check it with
`python3 scripts/verify-release.py` before making edits.
Download and load the tested image directly from the
[GitHub Release](https://github.com/MaximeSabbah/cosmik_mppi_sim/releases/tag/v1.0.0):

```bash
./scripts/host_fetch_image.sh
DEMO_SKIP_BUILD=1 ./scripts/host_acceptance.sh
./scripts/host_rviz.sh
```

Run these in the host terminal. The fetch helper downloads the release parts,
assembles them, checks SHA-256 and verifies the loaded image ID. No GitHub login
is needed for this public release. To build from source instead, run
`./scripts/host_acceptance.sh`. For development, use
`./scripts/host_dev.sh` (or append `with_rviz:=false` without a display). It mounts
your source and builds it in separate container volumes. The simulation inside
the fixed `sim` image does not pick up host source edits; the `dev` service does.

For an existing native Ubuntu 22.04 / ROS Humble environment, the original route
remains `./scripts/build.sh`, `./scripts/doctor.sh`, `./scripts/test.sh`, then
`./scripts/run.sh with_rviz:=true`.

Gazebo runs without its own window; RViz displays the robot, goals, collision
geometry and moving capsule. Set `gz_headless:=false` for the Gazebo window too.
Without a display, use `docker compose run --rm sim` (or `./scripts/run.sh` in a
native ROS installation). Stop with Ctrl-C. Restart the command
to reset the robot, task, and synthetic motion together.

The native/dev default ROS domain is **73**, limited to localhost. The host
automated runner uses **83**, the host RViz helper uses **84**, and the development
helper uses **85**. It does not inherit a
hardware demo's ROS domain. All scripts locate this folder themselves; extraction
at another path is supported. Rebuild after moving a previously built workspace.

## Check the baseline before changing it

The host acceptance helper already runs both checks below. For a native ROS
installation, or from a shell inside the configured image:

```bash
./scripts/smoke.sh --human false --output .runtime/check-no-human
./scripts/smoke.sh --human true  --output .runtime/check-human
```

These bounded checks start their own simulation and write `result.json` and
`launch.log`. Run them one at a time with no other demo running on domain 73.
They require measured arrivals at both targets, finite robot states/commands,
and an active command stream. A timeout-only alternation does not pass.

[Recorded validation](docs/validation.md): all 13 ROS packages built independently;
17 unit tests passed; headless runs with and without the capsule reached both
targets. RViz was inspected, and a second build/run from an archive extracted at
a different path also passed. This verifies the runnable baseline, not collision-free behavior for
all human trajectories. The Docker image is built from a clean base and was
validated on kurai through host SSH access on 2026-10-10. The student's own machine still needs its first smoke test.

## Where to work

| Area | File or directory |
| --- | --- |
| Student task and milestones | [prediction.md](docs/prediction.md) |
| ROS-independent hold/CV examples | [research/capsule_prediction.py](research/capsule_prediction.py) |
| Human input, transforms, geometry updates | `ros2_ws/src/storm_agimus_bridge/storm_agimus_bridge/storm_mppi_planner_node.py` |
| Collision geometry over rollouts | `storm/storm_kit/mpc/cost/primitive_collision_cost.py` |
| CaT weighting | `storm/storm_kit/mpc/control/mppi.py` |
| Frozen tuning and parameter map | [baseline.md](docs/baseline.md) |
| Supervisor's suggested handover | [handover.md](docs/handover.md) |

The research prediction examples are **not connected to MPPI**. The executable
baseline reacts to the latest observed human geometry. Incorporating a future
human trajectory is the student's next task.

Keep the baseline tag and develop on a separate branch.
Build/install/log directories are ignored. The copied dependency repositories are
vendored source with their revisions and local changes recorded in `provenance/`.
Keep their licenses when sharing this folder; see [third-party.md](docs/third-party.md).
