#!/usr/bin/env bash
# Run on the Docker host. Results remain in the shared source directory.
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e /.dockerenv ]]; then
  echo 'Run this script on the host, outside the existing development container.' >&2
  exit 2
fi
command -v docker >/dev/null
docker compose version >/dev/null
docker info >/dev/null
out="${DEMO_TEST_OUTPUT:-$root/.runtime/host-validation/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
exec > >(tee -a "$out/host.log") 2>&1
finish() {
  code=$?
  if [[ "$code" != 0 ]]; then printf 'FAIL host runner exit=%s\n' "$code" > "$out/phase.txt"; fi
  printf 'Results: %s\n' "$out"
}
trap finish EXIT
printf 'RUNNING preflight\n' > "$out/phase.txt"
printf 'Results: %s\n' "$out"
docker version --format '{{.Server.Version}}' > "$out/docker-version.txt"
docker compose version >> "$out/docker-version.txt"
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv > "$out/gpu.txt"
cd "$root"
docker compose config --quiet
if [[ "${DEMO_SKIP_BUILD:-0}" != 1 ]]; then
  printf 'RUNNING image-build\n' > "$out/phase.txt"
  docker compose --progress plain build sim 2>&1 | tee "$out/build.log"
fi
image_id="$(docker image inspect cosmik-mppi-sim:baseline --format '{{.Id}}')"
printf '%s\n' "$image_id" > "$out/image-id.txt"
printf 'RUNNING container-checks\n' > "$out/phase.txt"
# Use the exact newly built image, with no source overlay or old installed workspace.
# Only the results directory is mounted. The ROS domain is isolated from the robot.
docker run --rm --init --gpus all --shm-size 2g --network host \
  -e DEMO_DOMAIN_ID="${DEMO_DOMAIN_ID:-83}" \
  --mount "type=bind,src=$out,dst=/results" \
  "$image_id" bash ./scripts/container_checks.sh /results "$(id -u):$(id -g)"
