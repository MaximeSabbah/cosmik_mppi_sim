#!/usr/bin/env python3
"""Download and verify the exact tested image from public GitHub Release assets."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def matches(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest() == digest


def download_part(part, directory):
    name = part['name']
    if Path(name).name != name:
        raise ValueError('Release asset must be a filename.')
    path = directory / name
    if matches(path, part['bytes'], part['sha256']):
        print(f'Already verified: {name}', flush=True)
        return path
    temporary = directory / (name + '.partial')
    if temporary.exists() and temporary.stat().st_size >= part['bytes']:
        if matches(temporary, part['bytes'], part['sha256']):
            temporary.replace(path)
            return path
        temporary.unlink()
    print(f'Downloading {name} ({part["bytes"] / 1024**3:.2f} GiB)...', flush=True)
    command = ['curl', '--fail', '--location', '--retry', '3', '--retry-delay', '2',
               '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
               '--progress-bar', '--output', str(temporary)]
    result = subprocess.run(command + ['--continue-at', '-', part['url']], check=False)
    # Some proxies do not support byte ranges. Retry from zero in that case.
    if result.returncode == 33:
        temporary.unlink(missing_ok=True)
        subprocess.run(command + [part['url']], check=True)
    elif result.returncode:
        raise SystemExit(f'Download interrupted (curl exit {result.returncode}); rerun to resume.')
    if not matches(temporary, part['bytes'], part['sha256']):
        temporary.unlink(missing_ok=True)
        raise SystemExit(f'Checksum mismatch for {name}; rerun to download it again.')
    temporary.replace(path)
    print(f'Verified: {name}', flush=True)
    return path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/github-image.json').read_text())
    baseline = json.loads((root / 'provenance/docker-image.json').read_text())
    if (manifest['sha256'], manifest['bytes'], manifest['image_id'], manifest['archive']) != (
            baseline['export_sha256'], baseline['export_bytes'], baseline['image_id'],
            baseline['export_filename']):
        raise SystemExit('GitHub image metadata differs from the tested baseline.')
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else root / '.runtime/images'
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / manifest['archive']
    if matches(archive, manifest['bytes'], manifest['sha256']):
        print(f'Already verified: {archive}', flush=True)
        return
    parts = [download_part(part, directory) for part in manifest['parts']]
    temporary = directory / (manifest['archive'] + '.assembling')
    print('Assembling and verifying the Docker archive...', flush=True)
    with temporary.open('wb') as output:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if not matches(temporary, manifest['bytes'], manifest['sha256']):
        temporary.unlink(missing_ok=True)
        raise SystemExit('Assembled archive checksum mismatch.')
    temporary.replace(archive)
    # These are this helper's verified download chunks; the complete archive remains.
    for part in parts:
        part.unlink()
    print(f'Verified Docker archive: {archive}', flush=True)


if __name__ == '__main__':
    main()
