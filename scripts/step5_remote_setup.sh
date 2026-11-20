#!/usr/bin/env bash
# One-time GPU host setup for Step 5 (official Isaac-GR00T + LIBERO, libero_goal checkpoint).
# Run under nohup on the instance; writes SETUP_DONE as the last line on success.
# Expects the HF token already at /root/.cache/huggingface/token (mode 600).
# Resumable: each finished stage leaves /root/.np_stage_<name>, and a re-run skips it.
set -euxo pipefail

GR00T_COMMIT=${GR00T_COMMIT:-708617f5716c63fc0a327391492ea2b91977ae90}  # inspected locally 2026-11-19
REPO=/root/Isaac-GR00T
LIBERO_VENV=$REPO/gr00t/eval/sim/LIBERO/libero_uv/.venv
export PATH="$HOME/.local/bin:$PATH"

done_stage() { [ -f "/root/.np_stage_$1" ]; }
mark_stage() { touch "/root/.np_stage_$1"; echo "STAGE_DONE $1"; }

if ! done_stage apt; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git git-lfs ffmpeg libegl1-mesa-dev libglu1-mesa libgl1 libosmesa6-dev build-essential cmake
  command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  mark_stage apt
fi

if ! done_stage gr00t; then
  [ -d $REPO/.git ] || git clone https://github.com/NVIDIA/Isaac-GR00T.git $REPO
  cd $REPO
  git fetch --depth 1 origin "$GR00T_COMMIT" || true
  git checkout "$GR00T_COMMIT"
  git rev-parse HEAD > /root/gr00t_commit.txt
  uv sync --python 3.12
  mark_stage gr00t
fi
cd $REPO

if ! done_stage libero; then
  # The upstream script installs mujoco 4.0.1 then downgrades to 3.3.1 in place;
  # 4.0.1's libsdf_plugin.so survives the downgrade and fails to load
  # (libmujoco.so.4.0.1: cannot open shared object file). If its final import
  # check fails, reinstall mujoco 3.3.1 from a clean directory and repeat the
  # upstream checks verbatim.
  if ! bash gr00t/eval/sim/LIBERO/setup_libero.sh; then
    echo "setup_libero.sh failed; clean-reinstalling mujoco==3.3.1"
    source $LIBERO_VENV/bin/activate
    uv pip uninstall mujoco
    rm -rf "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/mujoco"
    uv pip install numpy==1.26.4 mujoco==3.3.1
    rm -rf $HOME/.libero
    printf 'n\n' | python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs"
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'PY'
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY
    deactivate
  fi
  mark_stage libero
fi

if ! done_stage checkpoint; then
  # Inference files only: "libero_goal/*" also pulls ~18 GB of training state (global_step*, rng_state*).
  uv run hf download nvidia/GR00T-N1.7-LIBERO --local-dir checkpoints/GR00T-N1.7-LIBERO \
    --include "libero_goal/*.json" "libero_goal/model-*.safetensors" "libero_goal/experiment_cfg/*"
  rm -rf checkpoints/GR00T-N1.7-LIBERO/.cache
  ls -la checkpoints/GR00T-N1.7-LIBERO/libero_goal
  mark_stage checkpoint
fi

mkdir -p /root/neuropilot_results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /root/neuropilot_results/gpu.txt
cp /root/gr00t_commit.txt /root/neuropilot_results/
echo SETUP_DONE
