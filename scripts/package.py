#!/usr/bin/env python3
"""Create a source handoff archive and checksum outside this directory."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
args = parser.parse_args()
target = args.archive.resolve()
if target.is_relative_to(root):
    parser.error('Write the archive outside the source directory')
ignored = {'.git', '__pycache__', '.pytest_cache', '.runtime'}
files = []
for path in sorted(root.rglob('*')):
    rel = path.relative_to(root)
    if any(part in ignored for part in rel.parts) or path.suffix == '.pyc':
        continue
    if rel.parts[:2] in [('ros2_ws', 'build'), ('ros2_ws', 'install'), ('ros2_ws', 'log')]:
        continue
    if path.is_symlink():
        raise RuntimeError(f'Source archive must not depend on external links: {rel}')
    if path.is_file() and str(rel) != 'provenance/release-sha256.json':
        files.append(path)
manifest = root/'provenance/release-sha256.json'
manifest.write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files}, indent=2)+'\n')
files.append(manifest)
with tarfile.open(target, 'w:gz') as tar:
    for path in files:
        tar.add(path, arcname=str(Path(root.name)/path.relative_to(root)), recursive=False)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix(target.suffix+'.sha256').write_text(f'{digest}  {target.name}\n')
print(f'{target}: {target.stat().st_size / 1024**2:.1f} MiB, {len(files)} files, SHA256 {digest}')
