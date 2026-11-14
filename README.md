# NeuroPilot

NeuroPilot is being implemented in five gated milestones. Steps 1–4 are verified. Step 4b (SSVEP frontend) is verified. Step 4c piece 1 (Franka Panda physical grasp) is verified. Step 4c pieces 2–3 (three-panel demo, two-look confirm, cancel mid-reach) are verified. Step 4d (SSVEP early stopping) is verified. Step 4e (intracortical decoder) is implemented and awaits the user's verification. Step 5 has not started.

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

## Step 4b SSVEP frontend

Step 4b replaces the LEFT/RIGHT menu with direct four-target SSVEP selection: each target flickers at its own frequency, and a training-free CCA decoder identifies which one the subject was looking at. The LEAF motor-imagery path remains unchanged.

```bash
uv pip install --python .venv/bin/python -e . --group dev --group ssvep
.venv/bin/python -m pytest -q
.venv/bin/python scripts/step4b_ssvep_eval.py           # calibrate on S11–S20, test on S1–S10
.venv/bin/python scripts/step4b_ssvep_demo.py           # BETA S1: PHONE -> CANCEL -> CUP -> MuJoCo pick
.venv/bin/python scripts/step4b_ssvep_demo.py --subject 7
```

- Data: real BETA SSVEP EEG (Liu et al. 2020) through MOABB, downloaded to MOABB's cache and never committed. The EEG is replayed; nobody looked at the MuJoCo objects.
- Targets (`interface/ssvep_targets.yaml`): APPLE 9.8 Hz, CUP 11.8 Hz, PHONE 13.8 Hz, CANCEL 15.8 Hz. These are spaced so that no target's 1st–3rd harmonic falls within 1.8 Hz of another target's. The evenly spaced 8.0/10.6/13.2/15.8 Hz set was rejected because 2 × 8.0 Hz lies 0.2 Hz from 15.8 Hz.
- Decoder: exact CCA against sine/cosine references (3 harmonics), 8 posterior channels, 2.0 s window from stimulus onset. If the raw score margin is below 0.025, the result is `UNSURE` and the selection is not changed.
- Calibration: the frequency set and the margin threshold were chosen on S11–S20 only. On held-out S1–S10 (160 trials), raw accuracy was 90.6%, 5.0% of trials were `UNSURE`, and 7.5% produced a wrong action. Per-subject accuracy ranged from 62% (S7) to 100%.
- Robot: the existing Step 3 scripted MuJoCo controller with programmatic lift verification. GR00T is not used in Step 4b.

## Step 4c piece 1 — Franka Panda physical grasp

This piece replaces the floating Step 3 gripper with MuJoCo Menagerie's Franka Emika Panda, loaded unmodified at a pinned commit (Apache-2.0, fetched by script and not committed). The decoded SSVEP selection drives a scripted pick: inverse-kinematics waypoints for straight-line Cartesian motion, joint position control, and a force-closed gripper. The object is never attached or teleported. Whether it is lifted is decided by contact and friction, then measured.

```bash
./scripts/fetch_franka.sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/step4c_franka_demo.py                               # S1: PHONE -> CANCEL -> CUP, records MP4
.venv/bin/python scripts/step4c_franka_demo.py --subject 7 --sequence PHONE  # real decoder error: FAIL, exit code 1
DYLD_FALLBACK_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
  .venv/bin/mjpython scripts/step4c_franka_demo.py --gui --no-video
```

- Verification: both fingers are in contact at grasp and after the lift, the object rises more than 10 cm and stays within 4 cm of the tool centre point, and other objects move less than 1 cm. A negative-control test runs the identical motion with the gripper open and asserts that nothing is lifted.
- Robustness check: 24 of 24 picks succeeded with each object displaced by up to ±1.5 cm.
- Objects are sized to fit the Panda's 8 cm gripper. The apple uses sliding friction 2.0 because a sphere slipped at 1.5.
- Limitations: the controller reads object positions from the simulator rather than from a camera. Motion is scripted (IK waypoints), not a learned policy. Videos are written to `artifacts/videos/step4c/` and require `ffmpeg`.

## Step 4c pieces 2–3 — three-panel demo, two-look confirm, cancel mid-reach

```bash
.venv/bin/python scripts/step4c_confirm_eval.py     # single look vs two-look confirm on held-out S1–S10
.venv/bin/python scripts/step4c_hero_demo.py        # renders artifacts/videos/step4c/hero_S1.mp4 (~2 min)
.venv/bin/python scripts/step4c_hero_demo.py --subject 7
```

The video shows three panels:

- **What the subject sees:** the camera view with a flicker tag on each object and a CANCEL tile. Pulses are drawn 8× slower than the real stimulus for photosensitivity safety.
- **Brain:** occipital EEG traces, the spectrum, and a head map of the response at the leading frequency. The decoder bars show the CCA scores computed on a window that grows with the look; nothing is animated by hand.
- **Robot:** the Franka with a translucent ghost arm at the planned grasp pose.

Story: the subject looks at CUP twice (preview, then confirm), and the robot starts reaching. Mid-reach the subject looks at CANCEL, so the robot aborts and returns home. The subject then looks at PHONE twice, and the robot grasps and lifts it. Every look uses the next recorded trial of that target, chosen by ground truth. Decoder errors play out as they really happen, and the script exits with code 1 if the final pick is not what the subject meant.

- Two-look confirm on held-out S1–S10: wrong robot actions fall from 7.5% (single look) to 1.2% (1 of 80 pairs). The price is 16.2% of pairs ending with no action and twice the looking time.
- Information transfer rate (Wolpaw, single look, 2.5 s per look including the 0.5 s cue): 1.40 bits per selection, about 34 bits/min.
- CANCEL is honoured only before the gripper closes. Putting an object back down is not implemented.
- Limitations: replayed EEG, not a live headset; scripted IK motion, not a learned policy; robot perception reads object poses from the simulator. CCA scores on short windows are naturally higher than at 2 s and are not probabilities.

## Step 4d — early stopping

The decoder checks the look every 0.1 s and commits at the first checkpoint t ≥ 0.3 s where the gap between the top two CCA scores is at least 0.10 × (2 s / t)^1.5. Short windows therefore need a larger lead. If the lead never qualifies, the full 2 s window is decoded as before, including `UNSURE`. The rule was chosen on S11–S20 as the highest information rate whose wrong-action rate did not exceed the fixed 2 s decoder's on those subjects.

```bash
.venv/bin/python scripts/step4d_early_stop_eval.py
.venv/bin/python scripts/step4c_hero_demo.py                  # early stopping on (default)
.venv/bin/python scripts/step4c_hero_demo.py --fixed-window   # previous behaviour
```

Held-out S1–S10, 160 trials:

| | Accuracy | Wrong action | Mean decision time | Bits/min (2.5 s/look incl. cue) |
|---|---:|---:|---:|---:|
| Fixed 2 s | 90.6% | 7.5% | 2.00 s | 33.7 |
| Early stop | 90.0% | 8.1% | 1.46 s | 42.1 |
| Two-look confirm, fixed | — | 1.2% (1/80) | 4.00 s for both looks | — |
| Two-look confirm, early | — | 2.5% (2/80) | 2.91 s for both looks | — |

Early stopping is about 27% faster per decision, at the cost of one more wrong decision in 160. With two-look confirm it adds one wrong action in 80 pairs. The demo's statistics panel shows the rates for whichever mode is running.

## Step 4e — intracortical reach-intention decoder (awaiting verification)

Step 4e plugs a second, implant-style decoder into the same robot pipeline. The data is recorded spiking from 162 neurons in macaque M1 and PMd during delayed reaching, from the Neural Latents Benchmark MC_Maze_Large dataset (DANDI 000138; Churchland lab, monkey J). It is read with `h5py` and is not committed. The nine reach targets are grouped by direction and mapped onto the table layout: left → APPLE, up → CUP, upper-right → PHONE, lower-right → CANCEL.

```bash
./scripts/fetch_nlb_mc_maze.sh
uv pip install --python .venv/bin/python -e . --group intracortical
.venv/bin/python scripts/step4e_intracortical_eval.py
.venv/bin/python scripts/step4e_intracortical_demo.py      # artifacts/videos/step4e/intracortical_demo.mp4
```

- Decoder: shrinkage LDA on square-root spike counts in 100 ms bins, using only activity before movement onset. No pretrained decoder exists for this data, so this is a small fitted adapter: fitted on the official NLB train split (375 trials) and tested on the held-out val split (125 trials). The window length (400 ms) was chosen by 5-fold cross-validation on the train split only.
- Held-out results: 97.6% accuracy (95.5% balanced) against a majority-class chance of 34.4%. A shuffled-label control reaches 23.2%. With a 200 ms window, accuracy is 74% at 600 ms before movement and rises to 96% at movement onset.
- Demo: CUP is decoded and the robot reaches; a CANCEL reach is decoded mid-reach and the robot aborts; APPLE is decoded and picked. Trials are the first held-out val trials of each direction, chosen by ground truth. The robot code is unchanged from the SSVEP demo.
- Limitations: this is recorded monkey data, replayed; it is not a human or a live implant. Decoding a four-way direction from a well-trained delayed-reach task is easier than free, everyday intent. LDA posteriors are overconfident (often 1.00) and are not calibrated probabilities.
