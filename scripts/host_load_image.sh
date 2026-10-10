#!/usr/bin/env bash
# Verify the supervisor's exported baseline before loading it into host Docker.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
archive="${1:?Usage: host_load_image.sh /path/to/cosmik-mppi-sim-2026-10-10.image.tar.gz}"
command -v python3 >/dev/null
command -v docker >/dev/null
docker info >/dev/null
python3 - "$root/provenance/docker-image.json" "$archive" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text())
archive = Path(sys.argv[2])
if archive.stat().st_size != record['export_bytes']:
    raise SystemExit('Image archive size differs from the baseline. Check that the download finished.')
print('Verifying Docker archive SHA-256...', flush=True)
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != record['export_sha256']:
    raise SystemExit('Image archive checksum mismatch. Download the baseline archive again.')
print('Archive matches the tested baseline.')
PY
expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$root/provenance/docker-image.json")"
tag="cosmik-mppi-sim:baseline"
current="$(docker image inspect "$tag" --format '{{.Id}}' 2>/dev/null || true)"
if [[ "$current" != "$expected" ]]; then
  docker image load --input "$archive"
fi
actual="$(docker image inspect "$tag" --format '{{.Id}}')"
if [[ "$actual" != "$expected" ]]; then
  echo "Loaded image ID differs from the tested baseline: $actual" >&2
  exit 1
fi
printf 'Ready: %s (%s)\nNext, from the repository: DEMO_SKIP_BUILD=1 ./scripts/host_acceptance.sh\n' "$tag" "$actual"
