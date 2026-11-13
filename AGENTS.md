# Repository Guidelines

## Project: NeuroPilot

NeuroPilot is a demo-first brain-to-robot physical-AI system. Its intended path is:

```text
real EEG -> existing BCI decoder -> neural/semantic intent -> NVIDIA GR00T -> robot simulation
                                                       |
                                                       +-> NVIDIA Cosmos scenario variation
```

The engineering goal is reliable integration of existing pretrained components, not a new EEG architecture or a training-heavy research project. Prefer inference and small adapters. Never expose credentials or claim success for mocked, fabricated, or silently substituted behavior.

## Non-Negotiable Milestone Process

Implement exactly one milestone at a time. Do not begin the next milestone until the user personally verifies the current one and explicitly approves continuation. At each milestone completion:

1. Run milestone tests and regression tests for all prior milestones.
2. Show exact commands and important output.
3. Explain what passed, limitations, and any real failures.
4. Provide a short manual test.
5. Stop and wait for approval.

Do not modify a working earlier milestone to hide later failures. If something fails, report what failed, why, evidence, possible fixes, and the recommended next action.

## Five-Milestone Plan

### Step 1 — Existing BCI decoder (current)

Prove that a real pretrained motor-imagery checkpoint can load real recorded EEG and predict `LEFT`/`RIGHT` without retraining. Investigate [Hrishirt/eeg-motor-imagery-bci](https://github.com/Hrishirt/eeg-motor-imagery-bci) first; verify its checkpoint, training preprocessing, dataset expectations, class mapping, tensor shapes, normalization, filtering, sampling rate, and trial segmentation from source rather than assumptions. Use BCI Competition IV 2a or another real public dataset. Synthetic arrays are permitted only for unit tests of validation/error handling.

Implement only the smallest inference wrapper, such as `BCIDecoder.predict(eeg_window)`, plus `scripts/step1_bci_demo.py`. The demo must load real EEG, print dataset/trial information, load the checkpoint, infer several trials, report actual outputs/confidence, compare labels where possible, and state limitations. Test preprocessing shape, model input shape, checkpoint loading, deterministic inference, class mapping, several labeled trials, and invalid inputs. Do not build UI, robotics, Isaac Sim, GR00T, Cosmos, Fireworks, or a new trained model.

### Step 2 — Brain-controlled selector

After Step 1 approval, replay recorded EEG as deterministic windows into a visual selector containing `APPLE`, `CUP`, and `PHONE`. `RIGHT` advances; `LEFT` selects. Show signal/prediction/confidence, highlighted and selected objects, and command history. Test `RIGHT, RIGHT, LEFT` from Apple and preserve Step 1 regression coverage. Stop after verification.

### Step 3 — Scripted robot in simulation

After Step 2 approval, connect selection to a deterministic controller: `execute(action="pick", target="cup")`. Use a practical NVIDIA-compatible simulator only after checking current requirements. Prefer local work for lightweight tasks and Vast.ai only when GPU/Linux is required; determine GPU, VRAM, CUDA, driver, storage, RAM, container, and headless/GUI requirements before renting. Programmatically verify approach, grasp, and lift (not just visual plausibility). Do not use GR00T yet. Stop after verification.

### Step 4 — NVIDIA GR00T

After Step 3 approval, inspect current official Isaac-GR00T documentation and isolate its environment if useful. Determine supported embodiment, checkpoint, observations, language/action formats, server/client API, simulator integration, Python/CUDA/VRAM requirements, and GPU architecture. Replace the scripted controller with pretrained zero-shot GR00T for the same `SELECT CUP` task through a deterministic adapter (`SELECT(cup)` -> `Pick up the cup.`). Prove actual policy actions and programmatic manipulation with no secret scripted fallback. Stop after verification.

### Step 5 — NVIDIA Cosmos

After Step 4 approval, add Cosmos only as an enhancement to the working BCI -> GR00T -> robot pipeline. Required scope is scenario/environment generation or augmentation across variations such as kitchen, warehouse, office, lighting, clutter, and object appearance. Record scenario metadata, target, task, success/failure, latency where relevant, and screenshots/video. Synthetic training data or GR00T fine-tuning is optional only after 5A works; prefer official NVIDIA workflows over custom infrastructure.

## Architecture and Interfaces

Create modules only when needed for the current milestone. The eventual organization is:

```text
bci/          decoder.py, preprocessing.py, replay.py, models/
interface/    selector.py, intent.py, visualization/
robot/        controller.py, scripted_controller.py, groot_controller.py
simulation/   environment.py, objects.py
cosmos/       generator.py, scenarios.py
tests/        unit and integration tests
scripts/      step1_bci_demo.py through step5_cosmos_demo.py
artifacts/    logs/, results/, screenshots/
```

Keep components decoupled. The BCI consumer must not depend on whether the decoder is a CNN, BiLSTM, CSP classifier, or foundation model. Robot controllers should eventually share an interface such as `execute(instruction, target=None)`, allowing scripted and GR00T implementations to be swapped without rewriting BCI code.

## Compute and Dependencies

Steps 1–2 should run locally on the Mac whenever possible. Steps 3–5 may use Vast.ai GPU instances after current official documentation establishes actual requirements. Prefer inference over training and document Python, OS, architecture, PyTorch, checkpoint, dataset, CUDA, driver, simulator, GR00T, Cosmos, GPU, image/container, ports, environment variables, and startup commands.

Fireworks is optional and forbidden in Steps 1–3. If later used for semantic reasoning, use `FIREWORKS_API_KEY` from the environment, structured outputs, measured latency, and a deterministic fallback; it must never produce low-level robot control. Never commit `FIREWORKS_API_KEY`, `VAST_API_KEY`, `HUGGINGFACE_TOKEN`, `WANDB_API_KEY`, SSH keys, or `.env` files. Keep only variable names in `.env.example`.

Use current official documentation for MNE, MNE-LSL, PyTorch, BCI datasets, Isaac Sim, Isaac-GR00T, and Cosmos. When dependencies fail: read the exact error, inspect installed versions, check official docs and upstream issues, determine the root cause, then change code. Do not guess APIs or blindly patch errors.

## Testing, Logging, and Reproducibility

Every milestone needs unit tests, integration tests, and a real manual demo. Mocked tests do not establish milestone completion. Log timestamp, model/checkpoint, dataset/trial, prediction, confidence/score, selected object, semantic command, robot action, success, latency, and errors; avoid dumping raw EEG unnecessarily. Store outputs under `artifacts/`. Pin dependencies where practical and keep README documentation current, distinguishing NeuroPilot code from upstream models/projects.

## Git and Handoff

Use focused milestone commits or tags such as `step-1-bci-working`, `step-2-brain-ui-working`, `step-3-brain-robot-working`, `step-4-bci-gr00t-working`, and `step-5-cosmos-working`. Do not tag or claim a milestone until its acceptance tests pass and the user approves it. Pull requests should explain the problem, implementation, validation commands/results, limitations, linked issues, and screenshots or sample output for visible changes.
