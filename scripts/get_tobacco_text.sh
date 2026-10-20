#!/usr/bin/env bash
# Downloads the Tesseract text of the Tobacco3482 documents into data/qs-ocr-small:
# QS-OCR-Small, published by Quicksign in 2019 (github.com/Quicksign/ocrized-text-dataset).
# 3,482 text files in ten folders, one per class (ADVE, Email, Form, Letter, Memo, News,
# Note, Report, Resume, Scientific). The documents come from the Truth Tobacco Industry
# Documents archive; the dataset has no licence of its own, so it is used for research
# and not stored in this repository. Cite Kumar et al. (2013) and Harley et al. (2015).
set -euo pipefail

DEST="${1:-data/qs-ocr-small}"
URL="https://github.com/QuickSign/ocrized-text-dataset/releases/download/v1.0/QS-OCR-small.tar.gz"
SHA256="9fb802036a2d95159a2fcb7c7bcd35ccbcc6d590f1cf39e7229a4fa9ee608b76"

if [ -d "$DEST/Memo" ]; then
  echo "Already downloaded: $DEST"
  exit 0
fi
mkdir -p "$DEST"
curl -L --fail --silent --show-error -o "$DEST/QS-OCR-small.tar.gz" "$URL"
echo "$SHA256  $DEST/QS-OCR-small.tar.gz" | sha256sum --check --quiet
tar --no-same-owner --no-same-permissions -xzf "$DEST/QS-OCR-small.tar.gz" -C "$DEST"
rm "$DEST/QS-OCR-small.tar.gz"
echo "Documents are in $DEST, one folder per class."
