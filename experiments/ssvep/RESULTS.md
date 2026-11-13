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
