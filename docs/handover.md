# What to give the student

Send the [Git repository](https://github.com/MaximeSabbah/cosmik_mppi_sim) and follow
[sharing.md](sharing.md) for the image transfer and student commands. The source
archive remains a standalone alternative. She needs the whole directory, because vendored ROS dependencies, robot
assets and STORM weights are part of the executable baseline. No bags, live COSMIK
connection or access to the supervisor's `/workspace` is needed for the first demo.

Suggested accompanying message:

> Start with README.md. First build and reproduce the baseline with and without
> the synthetic capsule. Keep my CaT tuning unchanged for that check. Save the
> doctor output and the two smoke-test results in your initial experiment record.
> Then follow docs/prediction.md: add stable-ID history and a time-indexed collision
> interface, prove zero-order hold reproduces the existing controller, and compare
> constant velocity before integrating your prediction model. The standalone
> predictor examples are scaffolding; they are not connected to MPPI yet.

The tested Docker image is distributed through the
[v1.0.0 GitHub Release](https://github.com/MaximeSabbah/cosmik_mppi_sim/releases/tag/v1.0.0).
She runs `./scripts/host_fetch_image.sh` to download, verify and load it, avoiding
a fresh dependency installation or a separate file transfer from the supervisor.
A clean Docker image was built and passed GPU, unit and both reaching checks on
kurai on 2026-10-10. The supervisor subsequently reported the GUI looked acceptable;
this feedback is recorded separately in `provenance/docker-image.json` and does
not establish quantitative human-clearance performance.
Her own workstation still needs one acceptance run after loading the image.

For collaboration, keep the initial baseline tag and use branches/PRs for
prediction work. Do not commit
build/install/log directories or experiments containing bulky bags. Record the
Git revision, active config hashes, environment, input scenario and random seed
with each result. Use separate experiment configurations instead of overwriting
the supervisor's baseline in place.

The original hardware workspace has not been modified by this export. Later
robot deployment should port the reviewed prediction changes and their tests
back into that existing demo, preserving its established launch and wiring.
