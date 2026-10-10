# Share the baseline and start development

Repository: <https://github.com/MaximeSabbah/cosmik_mppi_sim>.
The initial reproducible snapshot is tagged `baseline-2026-10-10`.

## Supervisor: send two links

1. Invite the student to the repository and send its URL.
2. Upload these three existing files to laboratory shared storage, keeping them
   together, and send the student the download link:

   - `cosmik-mppi-sim-2026-10-10.image.tar.gz` (approximately 4.4 GiB)
   - `cosmik-mppi-sim-2026-10-10.image.tar.gz.sha256`
   - `cosmik-mppi-sim-2026-10-10.image.tar.gz.image-id.txt`

On kurai they are already in:

```text
/home/msabbah/cosmik_ral/cosmik_ral/workspace/handoff/docker_images/
```

Inside the original development container that directory is
`/workspace/handoff/docker_images/`. The image has not been uploaded to a
registry or file-sharing service. Git contains the expected checksum and image ID
in `provenance/docker-image.json`, so the loader checks the download independently
of the adjacent checksum file. There is no need to re-export or rebuild this image.

Git stores source, tuning, assets, weights, dependency recipes, tests, documentation
and selected validation reports. Keep Docker exports, bags and bulk experiment
outputs outside Git. The source dependencies are already vendored: no recursive
submodule checkout or access to the supervisor's original workspace is needed.

## Student: host prerequisites, once

Use a Linux x86-64 workstation with an NVIDIA GPU. The tested native environment
is Ubuntu 22.04; the image supplies ROS 2 Humble, Gazebo Fortress, PyTorch and the
other application dependencies. Isaac Sim is not required.

Install Git and Python 3 on the host, a working NVIDIA driver, Docker Engine with
the Compose plugin, and NVIDIA Container Toolkit. Use the official instructions:

- [Docker Engine for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Docker access for your user](https://docs.docker.com/engine/install/linux-postinstall/)
- [NVIDIA Container Toolkit installation and Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Configure GPU support using the toolkit's Docker instructions. Before proceeding,
`nvidia-smi`, `docker info` and `docker compose version` must work in your normal
host terminal. RViz additionally needs an X11/Xwayland desktop. Budget roughly
30 GB of free disk space for the download, unpacked image and development build;
this is a planning allowance, not a measured minimum.

## Student: first run

After accepting the repository invitation and configuring your GitHub SSH access:

```bash
git clone git@github.com:MaximeSabbah/cosmik_mppi_sim.git
cd cosmik_mppi_sim
git switch -c human-prediction baseline-2026-10-10
python3 scripts/verify-release.py
./scripts/host_load_image.sh /path/to/cosmik-mppi-sim-2026-10-10.image.tar.gz
DEMO_SKIP_BUILD=1 ./scripts/host_acceptance.sh
./scripts/host_rviz.sh
```

Replace `/path/to/` with the directory where you downloaded the image. The loader
checks the archive and loaded image ID. The acceptance script checks CUDA, runs
17 unit tests, and runs reaching checks with and without the moving capsule.
Both targets must be reached spatially; timeout-only alternation does not pass.
Results stay in `.runtime/host-validation/`. Ctrl-C stops the interactive demo.

The first-run commands execute the frozen code inside the tested image. The Git
snapshot also contains later documentation and host helper improvements; those do
not change the tested controller. Do not expect arbitrary future source edits to
be visible in the frozen image.

## Student: edit and run your code

Stop the baseline demo, edit the cloned files in your normal editor, then run:

```bash
./scripts/host_dev.sh
```

This builds the mounted Git source and launches it with RViz, using dependencies
from the tested image. It handles the same display-cookie fallback as the baseline
helper. For a headless development run:

```bash
./scripts/host_dev.sh with_rviz:=false
```

To validate source changes (with no other demo on the same domain):

```bash
docker compose run --rm dev bash -c \
  './scripts/build.sh && ./scripts/test.sh && ./scripts/smoke.sh --human false --output .runtime/dev-no-human && ./scripts/smoke.sh --human true --output .runtime/dev-human'
```

The image is an environment and reference baseline; the `dev` service mounts your
current source. Python/C++ source changes need no image rebuild; `host_dev.sh`
rebuilds the ROS workspace. Dependency changes require an updated image and
revalidation. Run the helpers as your normal Docker-enabled user, not with sudo.

Domains are localhost-only by default: acceptance 83, frozen GUI 84, development
helper 85, direct native/Compose commands 73. `DEMO_DOMAIN_ID` overrides these.

Keep the supervisor's tuning intact for the initial comparison. Use separate
experiment configurations and record the Git commit, configuration, input
trajectory and seed alongside results. Open pull requests for reviewed changes.
Start the research work with [prediction.md](prediction.md); use
[architecture.md](architecture.md) to understand the rollout/simulator boundary.

## Message to send the student

> Clone the repository and follow docs/sharing.md. I will send the tested Docker
> image separately. First reproduce the baseline and keep its CaT tuning intact.
> Use host_dev.sh when running your edits. Your first research milestone is to
> connect time-indexed human geometry to the MPPI collision evaluation, prove that
> a stationary forecast matches the baseline, and then compare constant velocity.
> The current moving capsule is a repeatable input fixture; the prediction
> examples are not yet connected to the controller.
