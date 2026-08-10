#!/usr/bin/env bash
# Download UR Fall Detection Dataset (fenix.ur.edu.pl) — cam0 RGB videos + label CSVs only.
# Skips depth and accelerometer data.
set -euo pipefail

BASE_URL="http://fenix.ur.edu.pl/~mkepski/ds/data"
DEST_DIR="${1:-$HOME/esw/raw_data}"

FALL_DIR="$DEST_DIR/fall/cam0-rgb"
ADL_DIR="$DEST_DIR/adl/cam0-rgb"
LABEL_DIR="$DEST_DIR/labels"

mkdir -p "$FALL_DIR" "$ADL_DIR" "$LABEL_DIR"

download() {
  local url="$1" out="$2"
  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "skip (exists): $out"
    return
  fi
  echo "downloading: $url"
  curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
}

# Fall sequences: fall-01 .. fall-30, cam0 RGB only
for i in $(seq -w 1 30); do
  download "$BASE_URL/fall-${i}-cam0-rgb.zip" "$FALL_DIR/fall-${i}-cam0-rgb.zip"
done

# ADL sequences: adl-01 .. adl-40, cam0 RGB only
for i in $(seq -w 1 40); do
  download "$BASE_URL/adl-${i}-cam0-rgb.zip" "$ADL_DIR/adl-${i}-cam0-rgb.zip"
done

# Label CSVs
download "$BASE_URL/urfall-cam0-falls.csv" "$LABEL_DIR/urfall-cam0-falls.csv"
download "$BASE_URL/urfall-cam0-adls.csv" "$LABEL_DIR/urfall-cam0-adls.csv"

echo "Done. Files saved under: $DEST_DIR"
