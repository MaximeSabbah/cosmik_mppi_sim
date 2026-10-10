#!/usr/bin/env bash
# Export a tested image so a colleague can load it without repeating dependency installation.
set -eo pipefail
target="${1:?Usage: host_export_image.sh /path/to/image.tar.gz [image-tag]}"
image="${2:-cosmik-mppi-sim:baseline}"
if [[ -e "$target" || -e "$target.part" ]]; then
  echo "Refusing to overwrite $target or its partial export." >&2; exit 2
fi
mkdir -p "$(dirname -- "$target")"
image_id="$(docker image inspect "$image" --format '{{.Id}}')"
docker image save "$image" | gzip -1 > "$target.part"
mv -- "$target.part" "$target"
printf '%s\n' "$image_id" > "$target.image-id.txt"
(
  cd -- "$(dirname -- "$target")"
  sha256sum -- "$(basename -- "$target")" > "$(basename -- "$target").sha256"
)
printf 'Exported %s (%s)\n' "$target" "$image_id"
