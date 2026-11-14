# SSVEP Experiment Results

These results use real BETA SSVEP EEG loaded through MOABB. No synthetic EEG or relabeled motor-imagery data was used for evaluation.

## Dataset and preprocessing

- Dataset: Liu et al. 2020 BETA SSVEP benchmark
- Subjects: S1 for scaling; S1–S5 for subject comparison; S16 for the valid 3-second-window check
- BETA metadata: 64 EEG channels plus MOABB's stimulus channel, 250 Hz, 40 frequencies from 8.0–15.8 Hz in 0.2 Hz increments, four blocks, 40 trials per subject block set
- S1–S15 protocol: 0.5-second cue followed by 2 seconds of stimulation; 0.5/1/1.5/2-second windows were evaluated
- S16 protocol: 3-second stimulation; a 3-second window was evaluated separately
- Decoder channels: `O1`, `Oz`, `O2`, `PO3`, `PO4`, `PO7`, `PO8`, `POz`
- Preprocessing: posterior-channel selection and per-channel mean removal; no learned model

## S1 target scaling, 2-second window

| Targets | Trials | CCA | FBCCA |
|---:|---:|---:|---:|
| 4 | 16 | 93.8% | 87.5% |
| 8 | 32 | 90.6% | 84.4% |
| 12 | 48 | 91.7% | 91.7% |
| 20 | 80 | 90.0% | 83.8% |
| 40 | 160 | 88.1% | 86.3% |

## S1 window scaling, four targets

| Window | CCA | FBCCA |
|---:|---:|---:|
| 0.5 s | 43.8% | 31.2% |
| 1.0 s | 87.5% | 81.2% |
| 1.5 s | 100.0% | 93.8% |
| 2.0 s | 93.8% | 87.5% |
| 3.0 s | Not valid for S1 protocol | Not valid for S1 protocol |

For subject S16, the valid 3-second four-target result was CCA 100.0% and FBCCA 100.0% on 16 trials.

## Five-subject comparison, four targets, 2 seconds

| Subject | CCA | FBCCA |
|---:|---:|---:|
| S1 | 93.8% | 87.5% |
| S2 | 93.8% | 93.8% |
| S3 | 100.0% | 100.0% |
| S4 | 93.8% | 100.0% |
| S5 | 100.0% | 100.0% |
| Mean | 96.3% | 96.3% |

## Runtime and recommendation

On the Mac CPU, S1 four-target mean decode runtime was approximately 2.6 ms for CCA and 13.5 ms for FBCCA. The first correct replay identified the real 8.6 Hz target with a raw FBCCA score margin of 0.2211; scores are not calibrated probabilities.

The initial evidence supports four-target SSVEP selection, with CCA as the simplest and fastest baseline. Forty-target accuracy remains meaningful but lower, and short 0.5-second windows are unreliable. A practical NeuroPilot follow-up should begin with four targets and a window near 1.5–2.0 seconds. This experiment does not integrate SSVEP into the production pipeline.

## Ten-subject CCA sweep (supersedes the five-subject recommendation)

Command: `.venv/bin/python -m experiments.ssvep.sweep` (S1–S10, CCA, three harmonics, targets spread evenly over 8.0–15.8 Hz). Trials are pooled across subjects; "worst" is the lowest single-subject accuracy. Rejection uses a raw CCA score-margin threshold of 0.05, chosen after looking at the data, so the rejection columns are indicative, not a held-out estimate. Per-trial predictions and margins are saved in `artifacts/results/sweep.json`.

| Targets | Window | Channels | Trials | Accuracy | Worst subject | Rejected at margin ≥ 0.05 | Accuracy of accepted |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 2.0 s | posterior8 | 160 | 90.0% | 68.8% | 12.5% | 95.7% |
| 6 | 2.0 s | posterior8 | 240 | 90.0% | 70.8% | 12.9% | 94.7% |
| 10 | 2.0 s | posterior8 | 400 | 84.5% | 70.0% | 20.5% | 92.8% |
| 20 | 2.0 s | posterior8 | 800 | 78.1% | 47.5% | 27.5% | 90.7% |
| 4 | 2.0 s | occipital3 | 160 | 85.6% | 68.8% | 19.4% | 91.5% |
| 10 | 2.0 s | occipital3 | 400 | 78.0% | 57.5% | 26.7% | 90.4% |
| 4 | 1.5 s | posterior8 | 160 | 82.5% | 68.8% | 21.9% | 91.2% |
| 10 | 1.5 s | posterior8 | 400 | 75.5% | 55.0% | 26.0% | 86.1% |

Findings:

- S1–S5 were easier subjects than S6–S10, so the earlier 96% five-subject mean was optimistic. Over ten subjects, four targets give 90.0%, and S8 and S10 drop to 68.8%.
- The eight posterior channels beat O1/Oz/O2 in every configuration. The earlier three-channel advantage was noise from a 16-trial sample.
- A 2.0 s window beats 1.5 s in every configuration. Two seconds is the longest valid window for S1–S15.
- Only 4 and 6 targets reach about 90% without rejection. Ten targets reach 84.5%, or 92.8% if about one trial in five is rejected as uncertain.
