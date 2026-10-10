# Environment setup

The source archive includes the required project repositories, robot meshes,
collision spheres and learned self-collision weights. It excludes installed
workspaces, rosbags, IsaacLab, Isaac Gym and historical experiments.

Both the native source build and a fresh Docker image have been tested on kurai.
The image ran the GPU checks, all 17 unit tests, and both Gazebo reaching tests
without mounting the existing source or ROS install. See
`validation/docker-20261010/result.json` and `provenance/docker-image.json`.
The ROS base image is pinned by digest; the exported tested image also freezes the
installed apt/Python environment. Rebuilding later may resolve newer apt packages.

The initial clean-image test exposed a missing Hydra dependency. It is now pinned
in the requirements file, and the build/preflight checks import the planner's
actual ReacherTask module so this is caught before launching Gazebo.

## Existing Ubuntu 22.04 / ROS Humble machine

Use Python 3.10 and an NVIDIA GPU. The validated system had an RTX 5000 Ada,
PyTorch 2.5.1+cu124, NumPy 1.26.4, Pinocchio 3.8.0 and Gazebo 6.16.0.
The exact observed ROS package versions are recorded in
`provenance/author-ros-debian-versions.txt`.

With ROS/OSRF apt repositories configured, install the package list:

```bash
sudo apt-get update
xargs -r sudo apt-get install -y < docker/apt-packages.txt
python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r docker/requirements.txt
./scripts/build.sh
./scripts/doctor.sh
./scripts/test.sh
./scripts/run.sh with_rviz:=true
```

Prefer a dedicated development machine or the container route if you already
have a conflicting Python/ROS environment. Do not install the `pin` PyPI package
on top of ROS Pinocchio. No `ghalton` package is needed: the captured environment
uses the existing SciPy fallback. Adding another sampling backend could change
the fixed samples. `BUILD_JOBS=2` is the default; increase it if build resources permit.

The environment script discards unrelated ROS overlays and sets `STORM_ROOT`
relative to the handoff directory. For manual ROS commands:

```bash
source scripts/env.sh
ros2 topic list
ros2 param get /storm_mppi_planner storm_task_file
```

Do not source the original supervisor workspace. Do not run STORM's historical
README installation instructions for this packaged demo.

## Container recipe for a new workstation

Host prerequisites: Docker with Compose and NVIDIA Container Toolkit, a working
NVIDIA driver, and X11/Xwayland if using RViz. The headless path avoids display setup.
The recipe uses Docker's documented [GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/)
and [service inheritance](https://docs.docker.com/compose/how-tos/multiple-compose-files/extends/).

The easiest route is to load the supplied tested image, then check your GPU:

```bash
./scripts/host_load_image.sh /path/to/cosmik-mppi-sim-2026-10-10.image.tar.gz
DEMO_SKIP_BUILD=1 ./scripts/host_acceptance.sh
./scripts/host_rviz.sh
```

The loader verifies the archive checksum against the Git record, then checks the
loaded image ID. For a manual check, compare the adjacent `.sha256` and
`docker image inspect cosmik-mppi-sim:baseline --format '{{.Id}}'` with
`provenance/docker-image.json`. The source guide is the current handoff guide;
the exported runtime image contains the source/docs that existed when it was tested.

Without the image archive, `./scripts/host_acceptance.sh` builds the recipe and
runs the same acceptance sequence. For the complete user test and live log locations,
see [host-testing.md](host-testing.md).

For editing the source on the host, the `dev` service mounts this directory and
keeps container build/install/log files in separate named volumes:

```bash
./scripts/host_dev.sh  # build mounted source, then run with RViz
./scripts/host_dev.sh with_rviz:=false  # headless
```

For RViz, `DISPLAY` and `XAUTHORITY` must refer to your desktop's X11 display and
cookie file. The helper uses a readable `$HOME/.Xauthority` when `XAUTHORITY` is
unset. Their socket/cookie are mounted by the recipe. If the window fails,
establish the headless smoke test first and inspect
the display authentication error; avoid broad `xhost +` access.

To export a tested image again, run on the Docker host:

```bash
./scripts/host_export_image.sh /path/to/cosmik-mppi-sim.image.tar.gz
```

The helper writes the image archive, checksum and image ID. Keep the source archive
alongside it; the `dev` workflow needs editable source. No registry publication is required.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| Import/package resolves elsewhere | Run `doctor.sh`; use only `scripts/env.sh` for this terminal |
| CUDA unavailable | Confirm GPU visibility in the host/container; the supplied profile uses `cuda:0` |
| Robot/human absent in RViz | Fixed frame `fer_link0`, `/robot_description`, `/rtcosmik/collision_markers`, matching domain |
| No target alternation | Inspect `launch.log`, `/goal_pose`, `/control`, TF and `/clock`; smoke results separate arrivals/timeouts |
| Human moves visually but avoidance seems wrong | Check dynamic-slot initialization/update messages and the collision representation, not just RViz |
| A stale obstacle persists after stopping input | Restart the baseline; explicit stale-data handling belongs in the prediction work |
| Moving a built workspace breaks imports | Rebuild at the new location; build/install outputs are intentionally absent from the source archive |

Ctrl-C can produce shutdown tracebacks from legacy ROS/multiprocessing code. An
error during startup or while running requires investigation; a traceback only
after the requested shutdown is not evidence that the preceding smoke run failed.
