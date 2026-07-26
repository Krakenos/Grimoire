#!/usr/bin/env bash
#
# Re-download and verify the vendored frontend libraries.
# Run from anywhere; files are written next to this script.
#
# Usage:
#   ./fetch.sh          # download (if missing) and verify checksums
#   ./fetch.sh --force  # always re-download, then verify
#
set -euo pipefail

cd "$(dirname "$0")"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# name  version  sha256  source-url
libs=(
  "vis-network.min.js|10.1.0|fd730e304a5b877a937a896be9536e7974dc473d8ac87fa66644bce52cb5f8e4|https://unpkg.com/vis-network@10.1.0/standalone/umd/vis-network.min.js"
)

for entry in "${libs[@]}"; do
  IFS='|' read -r file version sha url <<<"$entry"

  if [[ $FORCE -eq 1 || ! -f "$file" ]]; then
    echo "Downloading $file ($version) ..."
    curl -sSL -o "$file" "$url"
  fi

  echo "Verifying $file ..."
  echo "${sha}  ${file}" | sha256sum -c -
done

echo "All vendored files verified."
