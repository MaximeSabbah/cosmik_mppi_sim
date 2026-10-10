#!/usr/bin/env python3
"""Check the unmodified extracted release before making research edits."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'provenance/release-sha256.json').read_text())
bad = []
for name, expected in manifest.items():
    path = root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        bad.append(name)
print(json.dumps({'pass': not bad, 'checked_files': len(manifest), 'changed_or_missing': bad}, indent=2))
raise SystemExit(bool(bad))
