#!/usr/bin/env bash
# Invoked by host_acceptance.sh inside the freshly built image.
set -uo pipefail
out="${1:?results directory required}"
owner="${2:?host uid:gid required}"
mkdir -p "$out"
trap 'chown -R "$owner" "$out"' EXIT
printf '' > "$out/checks.tsv"
run_check() {
  local name="$1" code
  shift
  printf 'RUNNING %s\n' "$name" | tee "$out/phase.txt"
  "$@" > "$out/$name.log" 2>&1
  code=$?
  printf '%s\t%s\n' "$name" "$code" >> "$out/checks.tsv"
  printf 'FINISHED %s exit=%s\n' "$name" "$code"
  return "$code"
}
run_check environment ./scripts/doctor.sh
environment_code=$?
run_check unit-tests ./scripts/test.sh
if [[ "$environment_code" == 0 ]]; then
  run_check no-human ./scripts/smoke.sh --human false --seconds 50 --wall-limit 240 --output "$out/no-human"
  run_check moving-capsule ./scripts/smoke.sh --human true --seconds 50 --wall-limit 240 --output "$out/moving-capsule"
else
  printf 'no-human\tSKIPPED\nmoving-capsule\tSKIPPED\n' >> "$out/checks.tsv"
fi
dpkg-query -W -f='${binary:Package}=${Version}\n' 'ros-humble-*' 'libignition-*' > "$out/ros-versions.txt" 2>/dev/null
python3 -m pip freeze > "$out/python-versions.txt"
python3 - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
checks = dict(line.split('\t') for line in (out/'checks.tsv').read_text().splitlines())
result = {'pass': all(value == '0' for value in checks.values()), 'checks': checks,
          'scope': 'Fresh Docker image, CUDA, unit tests, Gazebo transport and reaching. Human prediction is not connected.'}
for name in ['no-human', 'moving-capsule']:
    path = out/name/'result.json'
    if path.exists():
        result[name] = json.loads(path.read_text())
(out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
(out/'phase.txt').write_text('PASS\n' if result['pass'] else 'FAIL\n')
print(json.dumps({'pass': result['pass'], 'checks': checks}, indent=2))
raise SystemExit(0 if result['pass'] else 1)
PY
