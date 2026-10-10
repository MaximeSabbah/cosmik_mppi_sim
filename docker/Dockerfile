# Runtime validated in a fresh image on kurai, 2026-10-10.
# See provenance/docker-image.json and validation/docker-20261010/.
ARG ROS_BASE=osrf/ros:humble-desktop@sha256:fb07245b32187d74350be25323d8ad2f8ca5c25c325759911a1eff2267a49c1e
FROM ${ROS_BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/apt-packages.txt /tmp/demo-apt-packages.txt
RUN apt-get update && xargs -r apt-get install -y --no-install-recommends < /tmp/demo-apt-packages.txt \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
COPY docker/requirements.txt /tmp/demo-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/demo-requirements.txt
COPY . /opt/cosmik-mppi-sim
WORKDIR /opt/cosmik-mppi-sim
RUN ./scripts/build.sh && ./scripts/test.sh \
    && source scripts/env.sh \
    && python3 -c 'from storm_kit.mpc.task.reacher_task import ReacherTask'
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENTRYPOINT []
CMD ["./scripts/run.sh"]
