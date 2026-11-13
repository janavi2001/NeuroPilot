"""Inference-only adapter for the released LEAF EEG-language checkpoint."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


LEAF_LABELS = ("LEFT", "RIGHT")
LEAF_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
    "P2", "POz",
)


@dataclass(frozen=True)
class Prediction:
    """A decoder result with the actual prototype similarities."""

    intent: str
    score: float
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BCIDecoder:
    """Run frozen LEAF direct inference on preprocessed EEG windows.

    Input windows must already be LEAF-preprocessed and shaped ``(22, 800)``
    before channel unification, or ``(65, 800)`` after it. The decoder never
    trains, adapts, or changes checkpoint parameters.
    """

    def __init__(self, checkpoint: str | Path, leaf_root: str | Path = "third_party/LEAF", device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.leaf_root = Path(leaf_root).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        if not (self.leaf_root / "LEAF.py").is_file():
            raise FileNotFoundError(f"LEAF source not found under: {self.leaf_root}")

        if str(self.leaf_root) not in sys.path:
            sys.path.insert(0, str(self.leaf_root))
        import torch
        from LEAF import LEAF
        from load_config import build_model_config, load_yaml

        self._torch = torch
        self.device = torch.device(device)
        config = build_model_config(load_yaml(self.leaf_root / "configs" / "LEAF_mpnet.yaml"))
        self.model = LEAF(config)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self._text_embeddings = self._load_text_embeddings()
        self._instruction = self._text_embeddings["This is a motor imagery decoding task"]
        self._prototypes = np.stack([self._text_embeddings["Left"], self._text_embeddings["Right"]]).astype(np.float32)

    def _load_text_embeddings(self) -> dict[str, np.ndarray]:
        texts_path = self.leaf_root / "text_embeddings" / "mpnet-base.txt"
        vectors_path = self.leaf_root / "text_embeddings" / "mpnet-base.npy"
        texts = texts_path.read_text().splitlines()
        vectors = np.load(vectors_path, allow_pickle=False).astype(np.float32)
        if vectors.ndim != 2 or len(texts) != len(vectors) or len(set(texts)) != len(texts):
            raise ValueError("Invalid LEAF text-embedding cache")
        return dict(zip(texts, vectors))

    def predict(self, eeg_window: np.ndarray) -> dict[str, Any]:
        """Return ``intent``, raw cosine ``score``, and both class scores."""

        if not isinstance(eeg_window, np.ndarray):
            raise TypeError(f"eeg_window must be a numpy array, got {type(eeg_window).__name__}")
        if eeg_window.shape != (65, 800):
            raise ValueError(f"Expected LEAF input shape (65, 800), got {eeg_window.shape}")
        if eeg_window.dtype.kind not in "f":
            raise TypeError(f"eeg_window must contain floating-point values, got {eeg_window.dtype}")
        if not np.isfinite(eeg_window).all():
            raise ValueError("eeg_window contains non-finite values")

        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(eeg_window, dtype=np.float32)).unsqueeze(0).to(self.device)
        instruction = torch.from_numpy(self._instruction).to(self.device).unsqueeze(0)
        prototypes = torch.from_numpy(self._prototypes).to(self.device)
        with torch.inference_mode():
            embedding, _ = self.model(x, instruction)
            similarities = embedding @ prototypes.T
        scores = similarities[0].detach().cpu().numpy().astype(float)
        index = int(scores.argmax())
        result = Prediction(LEAF_LABELS[index], float(scores[index]), {
            label: float(score) for label, score in zip(LEAF_LABELS, scores)
        })
        return result.as_dict()
