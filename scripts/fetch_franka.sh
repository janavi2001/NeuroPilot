#!/usr/bin/env bash
# Fetch only the Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0) at a pinned commit.
set -euo pipefail
COMMIT=370c43965323c0944b433b202aac5058b5f907a9
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/mujoco_menagerie"
if [ -f "$DEST/franka_emika_panda/panda.xml" ]; then
  echo "Franka model already present: $DEST/franka_emika_panda"
  exit 0
fi
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set franka_emika_panda
git -C "$DEST" checkout "$COMMIT"
echo "Fetched Franka model at $COMMIT into $DEST/franka_emika_panda"
