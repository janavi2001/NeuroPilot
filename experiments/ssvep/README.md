# NeuroPilot SSVEP Experiment

This is an isolated CPU-only experiment. It does not modify the production BCI, selector, MuJoCo, LIBERO, GR00T, or Cosmos pipeline.

## Dataset

The default source is the real BETA SSVEP benchmark through MOABB (`Liu2020BETA`), starting with subject S1. It contains 64 channels, 250 Hz data, 40 target frequencies from 8.0–15.8 Hz, four blocks, and stored three-second epochs. S1–S15 have two seconds of visual stimulation after a 0.5-second cue; the experiment therefore evaluates S1 windows from 0.5 to 2.0 seconds only.

Install experiment dependencies from the repository root:

```bash
uv pip install --python .venv/bin/python -r experiments/ssvep/requirements.txt
```

## Evaluation

Download only BETA subject S1 on first use through MOABB and run:

```bash
.venv/bin/python -m experiments.ssvep.evaluate --subject 1 --targets 4 8 12 20 40
.venv/bin/python -m experiments.ssvep.replay --subject 1 --trial 0 --window 2.0
```

CCA and FBCCA use posterior channels present in the dataset (`O1`, `Oz`, `O2`, `PO3`, `PO4`, `PO7`, `PO8`, `POz`), three harmonics, and the official target frequencies. Scores and margins are not calibrated probabilities.

Results are written under `experiments/ssvep/artifacts/`. The experiment must report real accuracy, confusion matrices, runtime, target-count scaling, window tradeoffs, and a five-subject evaluation before any discussion of production integration.
