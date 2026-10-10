#!/usr/bin/env bash
# Build and run the editable source using the tested container dependencies.
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e /.dockerenv ]]; then
  echo 'Run this script in a terminal on the host desktop.' >&2; exit 2
fi
with_rviz=true
for arg in "$@"; do
  case "$arg" in
    with_rviz:=true) with_rviz=true ;;
    with_rviz:=false) with_rviz=false ;;
  esac
done
if [[ "$with_rviz" == true && -z "${DISPLAY:-}" ]]; then
  echo 'DISPLAY is unset. Use a desktop terminal, or pass with_rviz:=false.' >&2; exit 2
fi
if [[ -z "${XAUTHORITY:-}" && -r "$HOME/.Xauthority" ]]; then
  export XAUTHORITY="$HOME/.Xauthority"
fi
cd "$root"
export DEMO_DOMAIN_ID="${DEMO_DOMAIN_ID:-85}"
exec docker compose run --rm dev bash -c \
  './scripts/build.sh && ./scripts/run.sh with_rviz:=true "$@"' bash "$@"
