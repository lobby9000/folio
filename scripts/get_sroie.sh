#!/usr/bin/env bash
# Downloads the SROIE 2019 receipts (626 training receipts with line boxes and key fields)
# into data/sroie. The receipts come from the ICDAR 2019 Robust Reading Challenge and are
# meant for research use. This mirror's code is MIT licensed.
set -euo pipefail

DEST="${1:-data/sroie}"
if [ -d "$DEST/data/img" ]; then
  echo "Already downloaded: $DEST"
  exit 0
fi
git clone --depth 1 https://github.com/zzzDavid/ICDAR-2019-SROIE "$DEST"
echo "Receipts are in $DEST/data (img, box, key)."
