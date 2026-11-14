#!/usr/bin/env bash
# Download the NLB MC_Maze_Large train file (DANDI 000138, version 0.220113.0407, 149 MB).
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/nlb_mc_maze_large"
FILE="$DEST/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"
if [ -f "$FILE" ]; then echo "Already present: $FILE"; exit 0; fi
mkdir -p "$DEST"
curl -L --fail -o "$FILE" "https://api.dandiarchive.org/api/assets/e67b57b2-e9ad-4d95-b9e3-1262997360dc/download/"
echo "Downloaded $FILE"
