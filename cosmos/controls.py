"""Control videos for Cosmos 3 edge transfer, built from GR00T/LIBERO rollout videos.

GR00T's LIBERO videos put the 256x256 third-person camera on the left, the
wrist camera on the right and a caption strip underneath. We keep the
third-person view, upscale it to Cosmos's 480-tier square (640x640), and take
NVIDIA's "medium" Canny preset (100, 200), as in cosmos-framework's
AddControlInputEdge. Frame counts are trimmed to 4k+1 (the VAE's temporal
compression).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

SIZE = 640
EDGE_MEDIUM = (100, 200)


def probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den)


def read_frames(path: Path) -> tuple[np.ndarray, float]:
    """All frames as uint8 [T, H, W, 3] RGB, and the fps."""
    width, height, fps = probe(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3), fps


def write_frames(frames: np.ndarray, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(path)],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


def third_person_view(frames: np.ndarray) -> np.ndarray:
    """Left square of the side-by-side LIBERO video (above the caption strip)."""
    _, height, width, _ = frames.shape
    side = width // 2
    if height < side:
        raise ValueError(f"expected two square views side by side, got {width}x{height}")
    return frames[:, :side, :side]


def usable_frame_count(n: int) -> int:
    """Largest 4k+1 not above n."""
    if n < 1:
        raise ValueError("no frames")
    return (n - 1) // 4 * 4 + 1


def upscale(frames: np.ndarray, size: int = SIZE) -> np.ndarray:
    import cv2

    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_CUBIC) for f in frames])


def canny_edges(frames: np.ndarray, thresholds: tuple[int, int] = EDGE_MEDIUM) -> np.ndarray:
    """uint8 [T, H, W, 3] RGB -> 3-channel Canny edge maps (0 or 255)."""
    import cv2

    low, high = thresholds
    return np.stack([np.repeat(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), low, high)[..., None], 3, axis=2)
                     for f in frames])


def edge_agreement(control: np.ndarray, generated: np.ndarray, tolerance: int = 3) -> dict[str, float]:
    """How closely a generated video follows the control edges.

    Edges are recomputed on the generated frames with the same Canny preset.
    recall: share of control edge pixels with a generated edge within
    ``tolerance`` px; precision: the reverse. Computed over all frames.
    """
    import cv2

    if control.shape[:3] != generated.shape[:3]:
        raise ValueError(f"shape mismatch {control.shape} vs {generated.shape}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    gen_edges = canny_edges(generated)[..., 0] > 0
    ctl_edges = control[..., 0] > 127
    hit_recall = hit_precision = 0
    for c, g in zip(ctl_edges, gen_edges):
        hit_recall += np.count_nonzero(c & (cv2.dilate(g.astype(np.uint8), kernel) > 0))
        hit_precision += np.count_nonzero(g & (cv2.dilate(c.astype(np.uint8), kernel) > 0))
    recall = hit_recall / max(1, np.count_nonzero(ctl_edges))
    precision = hit_precision / max(1, np.count_nonzero(gen_edges))
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"recall": round(float(recall), 4), "precision": round(float(precision), 4), "f1": round(float(f1), 4)}


def build_control(source: Path, out_dir: Path) -> dict:
    """Write rgb.mp4 (cropped, upscaled source) and control_edge.mp4; return their metadata."""
    frames, fps = read_frames(source)
    count = usable_frame_count(len(frames))
    rgb = upscale(third_person_view(frames[:count]))
    edges = canny_edges(rgb)
    write_frames(rgb, out_dir / "rgb.mp4", fps)
    write_frames(edges, out_dir / "control_edge.mp4", fps)
    return {"source": str(source), "source_frames": int(len(frames)), "num_frames": int(count), "fps": fps,
            "size": SIZE, "edge_thresholds": list(EDGE_MEDIUM),
            "edge_pixel_fraction": round(float((edges[..., 0] > 0).mean()), 4)}
