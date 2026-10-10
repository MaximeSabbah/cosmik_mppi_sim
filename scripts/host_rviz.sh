#!/usr/bin/env bash
# Run in the host desktop terminal after host_acceptance.sh passes.
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e /.dockerenv ]]; then
  echo 'Run this script in a terminal on the host desktop.' >&2; exit 2
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo 'DISPLAY is unset. Run this in the host desktop terminal.' >&2; exit 2
fi
if [[ -z "${XAUTHORITY:-}" && -r "$HOME/.Xauthority" ]]; then
  export XAUTHORITY="$HOME/.Xauthority"
fi
cd "$root"
export DEMO_DOMAIN_ID="${DEMO_DOMAIN_ID:-84}"
out="$root/.runtime/host-validation/gui-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
exec > >(tee -a "$out/gui.log") 2>&1
printf 'Live GUI log: %s/gui.log\n' "$out"
printf 'RUNNING rviz\n' > "$out/phase.txt"
trap 'printf "STOPPED rviz exit=%s\n" "$?" > "$out/phase.txt"' EXIT
docker compose run --rm sim ./scripts/run.sh with_rviz:=true "$@"
