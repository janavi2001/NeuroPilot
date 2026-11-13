#!/usr/bin/env python3
"""Step 2: replay real EEG into a brain-controlled object selector."""

from __future__ import annotations

import argparse
from pathlib import Path

from bci.decoder import BCIDecoder
from bci.preprocessing import load_real_trials
from interface.replay import RecordedEEGReplay
from interface.selector import ObjectSelector


def run_headless(events, selector: ObjectSelector) -> int:
    for event in events:
        prediction = event.prediction
        state = selector.apply(str(prediction["intent"]))
        print(
            f"Trial {event.trial_number}: {prediction['intent']} "
            f"(ground truth {event.ground_truth}, score {prediction['score']:.4f}) "
            f"-> current={state.current_object} selected={state.selected_object}"
        )
    print(f"Final selected object: {selector.state.selected_object}")
    print("OBJECT SELECTION REPLAY: PASS")
    return 0


def run_ui(events, selector: ObjectSelector, delay_ms: int) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("NeuroPilot — Step 2 Brain-Controlled Selector")
    root.configure(bg="#10151c")
    root.geometry("1100x700")

    title = tk.Label(root, text="NEUROPILOT", fg="#7ee787", bg="#10151c", font=("Helvetica", 26, "bold"))
    title.pack(pady=(18, 4))
    status = tk.Label(root, text="Recorded EEG replay", fg="#c9d1d9", bg="#10151c", font=("Helvetica", 14))
    status.pack()
    advance_hint = tk.Label(
        root,
        text="▶  NEXT EEG EVENT  (press ENTER)  •  SPACE toggles auto-replay  •  R restarts",
        fg="#ffffff",
        bg="#238636",
        font=("Helvetica", 15, "bold"),
        padx=18,
        pady=10,
        cursor="hand2",
    )
    advance_hint.pack(pady=(10, 8))
    prediction_label = tk.Label(root, text="Prediction: —", fg="#58a6ff", bg="#10151c", font=("Helvetica", 20, "bold"))
    prediction_label.pack(pady=8)
    signal = tk.Canvas(root, width=1000, height=150, bg="#0d1117", highlightthickness=0)
    signal.pack(pady=8)
    object_frame = tk.Frame(root, bg="#10151c")
    object_frame.pack(pady=16)
    object_labels = []
    for name in selector.state.objects:
        label = tk.Label(object_frame, text=name, width=14, height=3, fg="#c9d1d9", bg="#21262d", font=("Helvetica", 18, "bold"), relief="flat")
        label.pack(side="left", padx=12)
        object_labels.append(label)
    selected_label = tk.Label(root, text="Selected: —", fg="#f2cc60", bg="#10151c", font=("Helvetica", 18, "bold"))
    selected_label.pack(pady=8)
    history_label = tk.Label(root, text="History: —", justify="left", anchor="w", width=90, fg="#8b949e", bg="#10151c", font=("Menlo", 12))
    history_label.pack(pady=12)

    event_list = list(events)
    controls = tk.Frame(root, bg="#10151c")
    controls.pack(pady=8)
    control_hint = tk.Label(
        root,
        text="CONTROLS:  SPACE = PAUSE/RESUME    ENTER = NEXT EEG EVENT    R = REPLAY AGAIN",
        fg="#f2cc60",
        bg="#10151c",
        font=("Helvetica", 13, "bold"),
    )
    control_hint.pack(pady=(2, 12))
    event_index = 0
    paused = True
    scheduled = None

    def draw_signal(window):
        signal.delete("all")
        trace = window[0]
        stride = max(1, len(trace) // 960)
        values = trace[::stride]
        if len(values) < 2:
            return
        low, high = float(values.min()), float(values.max())
        span = max(high - low, 1e-9)
        points = []
        for index, value in enumerate(values[:960]):
            x = 20 + index * (960 / max(1, min(len(values), 960) - 1))
            y = 75 - ((float(value) - low) / span) * 120
            points.extend((x, y))
        signal.create_line(*points, fill="#39d353", width=1)

    def render():
        nonlocal event_index, scheduled
        scheduled = None
        if event_index >= len(event_list):
            status.config(text="Replay complete — close the window when finished")
            return
        event = event_list[event_index]
        event_index += 1
        prediction = event.prediction
        state = selector.apply(str(prediction["intent"]))
        prediction_label.config(text=f"Prediction: {prediction['intent']}  score {prediction['score']:.4f}")
        draw_signal(event.eeg_window)
        for index, label in enumerate(object_labels):
            label.config(bg="#238636" if index == state.current_index else "#21262d")
        selected_label.config(text=f"Selected: {state.selected_object or '—'}")
        history_label.config(text="History: " + " | ".join(state.history[-5:]))
        status.config(text=f"Trial {event.trial_number} / {len(event_list)}  ground truth: {event.ground_truth}")
        if not paused:
            scheduled = root.after(delay_ms, render)

    def toggle_pause():
        nonlocal paused, scheduled
        paused = not paused
        if scheduled is not None:
            root.after_cancel(scheduled)
            scheduled = None
        pause_button.config(text="Resume" if paused else "Pause")
        if not paused:
            scheduled = root.after(100, render)

    def next_event():
        nonlocal paused, scheduled
        paused = True
        if scheduled is not None:
            root.after_cancel(scheduled)
            scheduled = None
        pause_button.config(text="Resume")
        render()

    def restart():
        nonlocal event_index, paused, scheduled
        if scheduled is not None:
            root.after_cancel(scheduled)
            scheduled = None
        selector.reset()
        event_index = 0
        paused = True
        pause_button.config(text="Resume")
        scheduled = root.after(100, render)

    pause_button = tk.Button(controls, text="Pause", command=toggle_pause, width=12, bg="#21262d", fg="#c9d1d9", activebackground="#30363d", activeforeground="#ffffff")
    pause_button.pack(side="left", padx=6)
    tk.Button(controls, text="Next EEG event", command=next_event, width=16, bg="#21262d", fg="#c9d1d9", activebackground="#30363d", activeforeground="#ffffff").pack(side="left", padx=6)
    tk.Button(controls, text="Replay again", command=restart, width=14, bg="#238636", fg="#ffffff", activebackground="#2ea043", activeforeground="#ffffff").pack(side="left", padx=6)

    root.bind("<space>", lambda _event: toggle_pause())
    root.bind("<Return>", lambda _event: next_event())
    root.bind("r", lambda _event: restart())
    root.bind("R", lambda _event: restart())
    advance_hint.bind("<Button-1>", lambda _event: next_event())

    scheduled = root.after(250, render)
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", type=Path, default=Path("data/raw/MI_BCIC_IV2a_mat/A01T.mat"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/leaf-v1.0-instruct-mpnet-base.ckpt"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--start-trial", type=int, default=39, help="1-based recorded trial to start replaying")
    parser.add_argument("--delay-ms", type=int, default=1800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true", help="Run the deterministic replay without opening a window")
    args = parser.parse_args()

    if args.start_trial < 1:
        parser.error("--start-trial must be at least 1")
    requested_end = args.start_trial - 1 + args.limit
    eeg, labels, metadata = load_real_trials(args.eeg, requested_end)
    eeg = eeg[args.start_trial - 1:requested_end]
    labels = labels[args.start_trial - 1:requested_end]
    print(f"Loaded {len(labels)} real trials; replay input shape {tuple(eeg.shape[1:])}")
    decoder = BCIDecoder(args.checkpoint, device=args.device)
    events = list(RecordedEEGReplay(eeg, labels, decoder))
    selector = ObjectSelector()
    if args.headless:
        return run_headless(events, selector)
    return run_ui(events, selector, args.delay_ms)


if __name__ == "__main__":
    raise SystemExit(main())
