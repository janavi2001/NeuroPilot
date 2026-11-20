# NeuroPilot

**Brain signals choose. A robot acts.** Recorded EEG is decoded into a goal ("put the bowl on the stove"). NVIDIA GR00T carries the goal out, and NVIDIA Cosmos shows the same run in realistic homes.

### ▶ [Watch the demo video](https://drive.google.com/file/d/1rwE-JbZ1jnKiZRX-zqZIBG9UrTbcjt6N/view?usp=sharing)

NeuroPilot was built in gated milestones, and all of them are verified:
- Steps 1–4
- Step 4b: SSVEP frontend
- Step 4c: Franka Panda grasp, three-panel demo, two-look confirm, cancel mid-reach
- Step 4d: SSVEP early stopping
- Step 4e: intracortical decoder
- Step 4f: six targets
- Step 5a: real EEG → GR00T in the LIBERO kitchen
- Step 5b: NVIDIA Cosmos 3 worlds
- Steps 5c–5d: the end-to-end demo videos

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

## Step 4e — intracortical reach-intention decoder

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

## Step 4f — six targets, five graspable objects

The table now holds a banana, an apple, a mug, a bottle and a phone, with CANCEL as the sixth target. The three-object scene and the four-target configuration remain the defaults for the verified earlier steps. Both demos now default to six.

```bash
.venv/bin/python scripts/step4f_select_frequencies.py                                       # chosen on S11–S20
.venv/bin/python scripts/step4b_ssvep_eval.py --config interface/ssvep_targets_6.yaml --output artifacts/results/step4f_ssvep_eval.json
.venv/bin/python scripts/step4d_early_stop_eval.py --config interface/ssvep_targets_6.yaml --output artifacts/results/step4f_early_stop_eval.json
.venv/bin/python scripts/step4e_intracortical_eval.py --scheme six
.venv/bin/python scripts/step4c_hero_demo.py               # six targets: CUP -> CANCEL mid-reach -> BOTTLE
.venv/bin/python scripts/step4e_intracortical_demo.py      # six directions: CUP -> CANCEL mid-reach -> BOTTLE
```

| Decoder | 4 targets (held-out) | 6 targets (held-out) |
|---|---|---|
| SSVEP CCA, single look | 90.6% accuracy, 7.5% wrong actions | 87.1% accuracy, 3.8% wrong actions, 20% UNSURE |
| SSVEP, two-look confirm | 1.2% wrong actions | 0.8% wrong actions (1 of 120), 37.5% of pairs need another look |
| SSVEP with early stopping | 42 bits/min | 48 bits/min, 1.68 s mean decision |
| Intracortical LDA, pre-movement | 97.6% (95.5% balanced) | 89.6% (84.6% balanced) |

- SSVEP frequencies (8.6, 9.8, 11.0, 12.2, 14.0 and 15.8 Hz) are the best of the 56 six-frequency sets whose 1st–3rd harmonics stay at least 1.2 Hz apart, scored on S11–S20 only. The UNSURE threshold (0.075) and early-stopping rule were chosen on the same subjects. The stricter threshold trades more "look again" for fewer wrong actions.
- The six intracortical directions split the two left-hand targets (−174° and 173°) rather than the upper-right pair (30° and 39°). This was chosen by train-split CV (84.7% versus 76.9%). Most held-out errors are among the three left-side directions.
- Robot: the banana is three capsules and is gripped across its middle. The bottle is gripped by its 2.4 cm neck. The six-object scene uses elliptic friction cones with impratio 10, which stopped the bottle creeping out of a sustained grip. It also re-aims the gripper by the measured 1.5–4.8 mm gravity-sag error before closing. Result: 40 of 40 picks had both fingers in contact at closing and lifted the object, with each object displaced up to ±1.5 cm.

## Step 5 (GPU) — real EEG chooses a kitchen task, pretrained GR00T performs it

Real BETA SSVEP EEG (held-out subject S1) → early-stopped CCA → two-look confirm → LIBERO-Goal instruction → NVIDIA `GR00T-N1.7-LIBERO` (`libero_goal`, `LIBERO_PANDA`, **inference only, no training**) → LIBERO kitchen simulation, whose own `check_success()` decides success. The six-target decoder from Step 4f is reused unchanged. Only the object names change (`interface/ssvep_targets_kitchen.yaml`: BOWL, PLATE, CHEESE, WINE, STOVE, CANCEL), and each name maps to an exact LIBERO-Goal task.

| Run | Result |
|---|---|
| GR00T alone, 5 tasks × 10 episodes | 50/50: bowl→stove 10/10, plate→front 10/10, cream cheese→bowl 10/10, wine→rack 10/10, turn on stove 10/10 |
| EEG-chosen, 5 commands | 5/5 confirmed the intended object; 5/5 GR00T successes. Brain time 2.7–7.5 s per command (CHEESE needed two extra looks after two UNSURE results) |

- Host: Vast RTX 4090 24 GB, driver 580.65.06, `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, 80 GB disk. Isaac-GR00T commit `708617f5716c63fc0a327391492ea2b91977ae90`. About 20 s per episode. Total run cost $0.79.
- Build jobs locally: `python scripts/step5_gr00t_jobs.py baseline --episodes 10` and `python scripts/step5_gr00t_jobs.py bci --subject 1`.
- On the GPU host: `scripts/step5_remote_setup.sh`, then the official `run_gr00t_server.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper`, then `scripts/step5_libero_goal_runner.py --jobs <file>` with the LIBERO venv python.
- Safety: `scripts/step5_vast_guard.py` (local hard cap plus credit floor; destroys the instance and verifies it is gone), `scripts/step5_self_destruct.sh` (on-instance timer using Vast's per-instance key), `scripts/step5_wait_remote.py` (fails loudly on a dead or unreachable job).
- Two upstream setup issues are handled in the setup script:
  - The LIBERO setup downgrades mujoco 4.0.1 to 3.3.1 in place, leaving a stale plugin that fails to load. The script does a clean reinstall.
  - `--include "libero_goal/*"` also downloads about 18 GB of training state. The script downloads only the inference files.
- Limitations:
  - These are NVIDIA's own LIBERO-Goal tasks, the ones the checkpoint was post-trained on. The 50/50 shows the policy works in its own benchmark, not that it generalises.
  - The EEG run is 5 commands from one held-out subject. The decoder's error rate comes from Step 4f's held-out statistics (0.8% wrong confirmed actions), not from these 5.
  - The brain → robot hand-off is offline: decisions are made locally first, then the jobs run on the GPU.
- Outputs: `artifacts/step5_gr00t/summary.json`, `bci_jobs.json` and `remote_results/` (56 videos, per-job `result.json`, logs).

## Step 5b — NVIDIA Cosmos 3: the brain-chosen GR00T episodes in new worlds

NVIDIA `Cosmos3-Nano` (16B, bf16, inference only) re-renders the Step 5a brain-chosen GR00T episodes as photoreal video through NVIDIA's Diffusers transfer recipe (`Cosmos3OmniModularPipeline`, edge control, the recipe's `edge.json` sampling values: 50 steps, guidance 3.0, control guidance 1.5, shift 10). The control is the real episode's third-person view, upscaled to 640×640, with NVIDIA's "medium" Canny preset (100/200). The prompt (`cosmos/scenarios.py`, in NVIDIA's structured prompt schema) restyles only the world around the robot.

| Clip | World | Frames | Generation | Edge F1 vs control (original sim) |
|---|---|---|---|---|
| BOWL → stove | modern kitchen | 41 | 251 s | 0.949 (0.987) |
| PLATE → front of stove | modern kitchen | 73 | 479 s | 0.960 (0.989) |
| CHEESE → bowl | modern kitchen | 41 | 262 s | 0.942 (0.988) |
| WINE → rack | modern kitchen | 61 | 400 s | 0.896 (0.987) |
| STOVE on | modern kitchen | 41 | 262 s | 0.947 (0.985) |
| PLATE | warehouse | 73 | 489 s | 0.946 (0.989) |
| PLATE | office | 73 | 490 s | 0.963 (0.989) |
| PLATE | night kitchen | 73 | 489 s | 0.965 (0.989) |
| CHEESE | office | 41 | 572 s* | 0.960 (0.988) |
| WINE | night kitchen | 61 | 907 s* | 0.950 (0.987) |

- **Edge F1:** Canny edges of each generated video are compared with the control edges, with a 3 px tolerance. The original simulator frames give the upper bound. The mean over all 10 clips is 0.948 against 0.988 for the simulator, so the robot and object outlines of the real episode are kept.
- \* The last two clips came from a second RTX PRO 6000 host (driver 580.178) that was about 2.2× slower. Its run cost $1.39 and reached its $1.50 credit floor while STOVE/warehouse was still generating. The watchdog copied the results back and destroyed the instance, so that clip was not made. Total Cosmos cost: $3.32.
- **Host:** Vast RTX PRO 6000 Blackwell, 96 GB, driver 595.84, `nvidia/cuda:13.0.1-base-ubuntu22.04`. torch 2.14.1+cu130, diffusers `4f788320` (0.44.0.dev0). Peak GPU memory was 35.5 GiB. Setup took about 4 minutes. Total cost $1.93.
- **Commands:**
  - Locally: `python scripts/step5b_cosmos_prepare.py` builds the controls, prompts and the priority-ordered job list.
  - On the GPU host: `scripts/step5b_cosmos_setup.sh`, then `scripts/step5b_cosmos_runner.py --negative-prompt /root/cosmos_inputs/negative_prompt.json`.
  - Locally: `python scripts/step5b_cosmos_report.py`.
- **Limitations:**
  - These are generated videos. The success labels come from the original LIBERO runs, and GR00T was not re-run in the generated worlds.
  - Small objects are sometimes reinterpreted: the stove burner becomes a glowing disc, the plate becomes a metal ring, the night kitchen's cabinet becomes a sink, and the office clip has a blurred patch at the bottom edge. WINE has the lowest F1 (0.896).
  - The sources are 256 px simulator frames at 20 fps, about 2–4 s long.
  - The other 10 episode/world combinations were not generated, to stay within budget.
- **Outputs:** `artifacts/step5_cosmos/summary.json`, `compare/*.mp4` (simulator next to Cosmos, plus the `bci_02_PLATE__4_worlds.mp4` grid) and `remote_results/` (videos, per-clip `result.json`, logs, including the first failed attempt).

## Step 5c — end-to-end demo video

```bash
.venv/bin/python scripts/step5c_mission_demo.py   # -> artifacts/videos/step5/neuropilot_mission.mp4 (76 s, 1920x1080)
```

The video (84 s) runs in five parts:
1. **Command 1 (BOWL) in the plain simulator.** Real EEG traces and the CCA score race replay the recorded BETA S1 trial, the two-look confirm produces an instruction, GR00T acts, and LIBERO's success check is shown.
2. **"Why Cosmos".** The same run appears as simulator → edge control → photoreal kitchen, side by side.
3. **Commands 2–5.** When GR00T acts, a wipe turns the simulator into Cosmos, and each command is in a different world (warehouse, office, night kitchen, kitchen). The simulator inset stays as ground truth.
4. **The four-world grid** for one decision.
5. **An end card** with the measured numbers and caveats.

Nothing is animated by hand:
- Every look is re-decoded from its recorded trial with the same early-stopping CCA. The script stops if the result differs from the decision logged in `artifacts/step5_gr00t/bci_jobs.json`.
- Every number is read from `artifacts/results/step4f_*` and the Step 5 summaries.
- `step4f_confirm_eval.json` comes from `scripts/step4c_confirm_eval.py --eval artifacts/results/step4f_ssvep_eval.json`.

## Step 5d — narrated demo

```bash
.venv/bin/python scripts/step5d_voiceover.py        # ElevenLabs voiceover, one file per line, with word timings
.venv/bin/python scripts/step5d_narrated_demo.py    # -> artifacts/videos/step5/neuropilot_narrated.mp4 (2:45)
```

The script is `scripts/step5d_narration.yaml`, voiced by ElevenLabs "Rachel" (`eleven_multilingual_v2`). The narration drives the pacing:
- Every scene lasts at least as long as its line.
- Captions light up word by word from ElevenLabs' character timestamps.
- The panel being described is spotlighted while the rest is dimmed.
- The first command is slowed down so it can be explained, and the on-screen tags say so.

Data and checks are the same as in Step 5c. The API key is read from `ELEVENLABS_API_KEY` or `~/.config/elevenlabs_api_key` and is never printed or committed. Unchanged lines are cached, so editing one line only re-bills that line.
