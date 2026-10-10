# Testing the student handoff on a Docker host

Run these commands outside the existing development container. On kurai, the
shared source directory is:

```bash
cd /home/msabbah/cosmik_ral/cosmik_ral/workspace/handoff/cosmik-mppi-sim
./scripts/host_acceptance.sh
```

This builds the Docker image, checks CUDA/imports, runs the unit tests, and runs
two bounded Gazebo reaching tests (without and with a moving capsule). It uses
ROS domain 83 by default and mounts only a results directory into the test image.
It does not use the original workspace's installed ROS packages.

Each run writes `.runtime/host-validation/<UTC timestamp>/` containing `phase.txt`,
`host.log`, `build.log`, `image-id.txt`, dependency versions, and a final `result.json`.
The directory is shared with the assistant's workspace, so logs can be reviewed
without copying them into the conversation. After correcting a runtime issue
that does not require rebuilding, use `DEMO_SKIP_BUILD=1 ./scripts/host_acceptance.sh`.
Changes to files baked into the image require a normal rebuild.

After the automated checks pass, run in a terminal on the host desktop:

```bash
./scripts/host_rviz.sh
```

This uses ROS domain 84, starts the same packaged simulation with RViz, and keeps
Gazebo's own window closed. Check that the complete robot, both targets and the
moving capsule appear, and that reaching alternates for at least two minutes.
Distinguish reaching a target from a switch caused by the ten-second timeout.
Use Ctrl-C in that terminal to stop the demo, then launch it again to check restart.
The live terminal log is saved in `.runtime/host-validation/gui-<timestamp>/gui.log`
so the assistant can review the same run while you inspect the window.
The supplied predictor examples are still research scaffolding, not an enabled
prediction feature.

For a headless host, the automatic tests are sufficient for software bring-up;
GUI/display authentication must be checked in an actual desktop session.

Keep the test report and image ID with the release. A successful test on kurai
validates this Docker image on kurai's GPU/driver. It does not substitute for the
student's first smoke test on her own workstation.
