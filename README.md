# NeuroPilot

NeuroPilot is being implemented in five gated milestones. Steps 1–4 are verified. Step 5 has not started.

## Step 1 setup

```bash
git submodule update --init  # third_party/LEAF (upstream LEAF, pinned)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . --group dev
```

The implementation uses the released [LEAF](https://github.com/Jiang-Muyun/LEAF) instruction-tuned checkpoint for direct EEG-to-text-prototype inference. The originally proposed Hrishirt and Roeiyanku repositories were audited but did not contain distributable pretrained checkpoints. The checkpoint is intentionally stored under ignored `artifacts/checkpoints/`.

Download the real BCI Competition IV 2a subject file from the BNCI source:

```bash
mkdir -p data/raw/MI_BCIC_IV2a_mat
wget -O data/raw/MI_BCIC_IV2a_mat/A01T.mat \
  https://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
```

Place `leaf-v1.0-instruct-mpnet-base.ckpt` in `artifacts/checkpoints/` using the release URL documented by LEAF. Do not commit model or EEG artifacts.

## Run Step 1

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/step1_bci_demo.py
```

The loader follows LEAF’s BCIC-IV-2a preprocessing: 22 channels at 250 Hz, 0.3–40 Hz filtering, resampling to 200 Hz, the cue-relative four-second window, robust percentile/IQR normalization, and mapping to the 65-channel LEAF montage. Scores are raw cosine similarities, not calibrated probabilities.

Step 1 was verified before continuing.

## Step 2 brain-controlled selector

After Step 1 approval, run the visual replay UI:

```bash
.venv/bin/python scripts/step2_ui_demo.py
```

The default replay uses real trials 39–42 because their decoded sequence is `LEFT, RIGHT, RIGHT, LEFT`: it selects `APPLE`, highlights `CUP`, highlights `PHONE`, and selects `PHONE`. The UI shows the signal, decoded command, raw score, highlighted object, selected object, and recent command history. `RIGHT` advances through `APPLE`, `CUP`, and `PHONE`; `LEFT` selects the highlighted object. Use `--start-trial N --limit M` to inspect another recorded window. For deterministic terminal verification without opening a window:

```bash
.venv/bin/python scripts/step2_ui_demo.py --headless
```

Step 2 was verified before continuing.

## Step 3 scripted robot

Step 3 connects the real EEG replay and selector to a deterministic local MuJoCo simulation. MuJoCo was selected for this Mac because it installs natively on Apple Silicon; NVIDIA Isaac Sim remains intentionally out of scope until a later GPU-backed milestone.

```bash
.venv/bin/python -m pytest tests/test_step3_robot.py -q
.venv/bin/python scripts/step3_robot_demo.py
```

The demo decodes trials 1–2 (`RIGHT`, then `LEFT`) to select `CUP`, moves the scripted gripper to the target, attaches it, and verifies the object’s measured height increase. On macOS, MuJoCo’s passive viewer must be launched with `mjpython`:

```bash
DYLD_FALLBACK_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
  .venv/bin/mjpython scripts/step3_robot_demo.py --gui
```

Press Enter in the terminal after inspecting the robot window to close it. The ordinary `.venv/bin/python` command with `--gui` is not supported by MuJoCo on macOS.

GR00T is not used in Step 3. Do not begin Step 4 until the user verifies the scripted robot milestone explicitly.

## Step 4 GR00T integration status

The approved official route uses NVIDIA `GR00T-N1.7-LIBERO` with the `LIBERO_PANDA` embodiment on a Vast RTX 4090. `scripts/step4_gr00t_demo.py` replays real EEG, selects `CUP`, emits `Pick up the cup.`, and sends an explicit compatibility instruction to the official LIBERO runner. Because LIBERO has no cup asset, the current documented bridge uses its alphabet-soup-in-basket task; it does not silently claim that alphabet soup is a cup.

The end-to-end run produced real GR00T actions and programmatic LIBERO outcomes: 3/5 successful episodes (0.600), with videos saved under `artifacts/videos/step4/`. The policy is stochastic, so this is reported as an observed evaluation result rather than a guaranteed success. Run the GPU-side runner from the official Isaac-GR00T checkout with `scripts/step4_libero_runner.py`; no scripted robot fallback is used.
