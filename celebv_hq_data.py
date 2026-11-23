from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch


class DatasetClipError(RuntimeError):
    """Raised when a processed CelebV-HQ clip cannot be decoded."""


@dataclass(frozen=True)
class FaceCentricVideoConfig:
    width: int = 512
    height: int = 512
    frames: int = 49
    target_fps: float = 25.0


@dataclass
class FaceCentricWindow:
    pixel_values: torch.Tensor
    conditioning_image: torch.Tensor
    face_mask: torch.Tensor


@lru_cache(maxsize=1)
def _face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")
    return detector


def _video_metadata(path: str) -> Tuple[float, int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise DatasetClipError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if (
        frame_count <= 0
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise DatasetClipError(
            f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}"
        )
    return fps, frame_count


def temporal_sample_positions(
    total_frames: int,
    source_fps: float,
    target_fps: float,
    num_frames: int,
    training: bool,
    rng: random.Random,
) -> np.ndarray:
    if (
        total_frames <= 0
        or num_frames <= 0
        or not math.isfinite(source_fps)
        or source_fps <= 0
        or not math.isfinite(target_fps)
        or target_fps <= 0
    ):
        raise DatasetClipError("Invalid temporal sampling parameters")

    # Interpret 23.976/24 FPS clips at 25 FPS when enough source frames exist.
    # This small speed-up avoids repeated frames.
    if (
        target_fps * 0.92 <= source_fps < target_fps
        and total_frames >= num_frames
    ):
        offsets = np.arange(num_frames, dtype=np.float64)
    else:
        offsets = np.arange(num_frames, dtype=np.float64) * source_fps / target_fps
    required_span = float(offsets[-1]) if len(offsets) else 0.0
    if required_span > total_frames - 1:
        raise DatasetClipError(
            f"Clip has {total_frames} frames but needs span {required_span:.2f}"
        )
    max_start = max(0.0, total_frames - 1 - math.ceil(required_span))
    start = rng.uniform(0.0, max_start) if training and max_start > 0 else max_start / 2.0
    return start + offsets


def _decode_interpolated_frames(path: str, positions: np.ndarray) -> List[np.ndarray]:
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    required = sorted(set(lower.tolist()) | set(upper.tolist()))
    if not required:
        raise DatasetClipError(f"No frames requested from {path}")

    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, required[0])
    decoded: Dict[int, np.ndarray] = {}
    required_set = set(required)
    for frame_index in range(required[0], required[-1] + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in required_set:
            decoded[frame_index] = frame
    capture.release()

    missing = required_set.difference(decoded)
    if missing:
        raise DatasetClipError(f"Could not decode frames {sorted(missing)} from {path}")

    result: List[np.ndarray] = []
    for position, low, high in zip(positions, lower, upper):
        if low == high:
            result.append(decoded[int(low)])
        else:
            alpha = float(position - low)
            result.append(
                cv2.addWeighted(
                    decoded[int(low)],
                    1.0 - alpha,
                    decoded[int(high)],
                    alpha,
                    0.0,
                )
            )
    return result


def _largest_face(frame: np.ndarray) -> Tuple[int, int, int, int] | None:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    minimum_side = max(32, int(min(height, width) * 0.08))
    faces = _face_detector().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(minimum_side, minimum_side),
    )
    if len(faces) == 0:
        return None
    return max(
        ((int(x), int(y), int(w), int(h)) for x, y, w, h in faces),
        key=lambda item: item[2] * item[3],
    )


def _soft_face_mask(frame: np.ndarray, width: int, height: int) -> torch.Tensor:
    source_height, source_width = frame.shape[:2]
    face = _largest_face(frame)
    mask = np.zeros((height, width), dtype=np.float32)
    if face is None:
        center = (width // 2, int(round(height * 0.43)))
        axes = (int(round(width * 0.28)), int(round(height * 0.38)))
    else:
        x, y, face_width, face_height = face
        scale_x = width / source_width
        scale_y = height / source_height
        center = (
            int(round((x + face_width / 2.0) * scale_x)),
            int(round((y + face_height / 2.0) * scale_y)),
        )
        # Include face, hair, and the immediate head boundary.
        axes = (
            max(1, int(round(face_width * scale_x * 0.85))),
            max(1, int(round(face_height * scale_y * 1.05))),
        )
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, thickness=-1)
    sigma = max(3.0, min(width, height) * 0.02)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return torch.from_numpy(np.clip(mask, 0.0, 1.0)).unsqueeze(0)


def _frames_to_tensor(
    frames: Sequence[np.ndarray],
    width: int,
    height: int,
) -> torch.Tensor:
    resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.shape[1] != width or frame.shape[0] != height
        else frame
        for frame in frames
    ]
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized]
    array = np.stack(rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor.mul(2.0).sub(1.0)


def read_celebv_hq_video_clip(
    path: str,
    cfg: FaceCentricVideoConfig,
    training: bool,
    rng: random.Random,
) -> FaceCentricWindow:
    source_fps, total_frames = _video_metadata(path)
    positions = temporal_sample_positions(
        total_frames,
        source_fps,
        cfg.target_fps,
        cfg.frames,
        training,
        rng,
    )
    frames = _decode_interpolated_frames(path, positions)
    pixels = _frames_to_tensor(frames, cfg.width, cfg.height)
    face_mask = _soft_face_mask(frames[0], cfg.width, cfg.height)
    return FaceCentricWindow(
        pixel_values=pixels,
        conditioning_image=pixels[0].clone(),
        face_mask=face_mask,
    )


def read_celebv_hq_conditioning_frame(
    path: str,
    cfg: FaceCentricVideoConfig,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, total_frames = _video_metadata(path)
    frame_index = rng.randrange(total_frames)
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise DatasetClipError(f"Could not decode frame {frame_index} from {path}")
    pixels = _frames_to_tensor([frame], cfg.width, cfg.height)
    face_mask = _soft_face_mask(frame, cfg.width, cfg.height)
    return pixels[0], face_mask


def _square_face_crop(frame: np.ndarray, scale: float = 2.45) -> np.ndarray:
    height, width = frame.shape[:2]
    face = _largest_face(frame)
    if face is None:
        side = min(height, width)
        left = (width - side) // 2
        top = (height - side) // 2
        return frame[top : top + side, left : left + side]

    x, y, face_width, face_height = face
    center_x = x + face_width / 2.0
    center_y = y + face_height * 0.45
    side = min(float(min(width, height)), max(face_width, face_height) * scale)
    left = int(round(min(max(0.0, center_x - side / 2.0), width - side)))
    top = int(round(min(max(0.0, center_y - side / 2.0), height - side)))
    side_int = int(round(min(side, width - left, height - top)))
    return frame[top : top + side_int, left : left + side_int]


def read_portrait_conditioning_image(
    path: str,
    cfg: FaceCentricVideoConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise DatasetClipError(f"Could not read portrait image: {path}")
    cropped = _square_face_crop(frame)
    pixels = _frames_to_tensor([cropped], cfg.width, cfg.height)
    face_mask = _soft_face_mask(cropped, cfg.width, cfg.height)
    return pixels[0], face_mask
