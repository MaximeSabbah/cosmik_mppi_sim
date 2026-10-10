#!/usr/bin/env bash
# One command to download, verify and load the released Docker environment.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cache="${1:-$root/.runtime/images}"
command -v python3 >/dev/null
command -v curl >/dev/null
command -v docker >/dev/null
docker info >/dev/null
python3 "$root/scripts/fetch_image.py" "$cache"
archive="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["export_filename"])' "$root/provenance/docker-image.json")"
exec "$root/scripts/host_load_image.sh" "$cache/$archive"
