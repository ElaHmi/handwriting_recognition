"""
EMNIST-style handwritten word recognition — local Flask web UI.
"""

from __future__ import annotations

import json
import os
import socket
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import base64
import io
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spellchecker import SpellChecker

# ---------------------------------------------------------------------------
# SpellChecker Initialization
# ---------------------------------------------------------------------------
spell = SpellChecker()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# If set, must exist. Otherwise we use the first file found in MODEL_CANDIDATES.
MODEL_PATH_ENV = os.environ.get("MODEL_PATH")
WRITER_MODEL_PATH_ENV = os.environ.get("WRITER_MODEL_PATH")

# Default writer labels if checkpoint has no writer_to_idx (part2.ipynb order).
DEFAULT_WRITER_NAMES = ("aisha", "ela", "mariam", "noorah", "shaikha")

# Writer CNN input size (part2 WriterWordDataset)
WRITER_TARGET_H = 64
WRITER_TARGET_W = 256

# Heuristic "unknown / not in training set" (softmax-only; no prototype file)
WRITER_UNK_MAX_PROB = float(os.environ.get("WRITER_UNK_MAX_PROB", "0.38"))
WRITER_SOFT_MAX_PROB = float(os.environ.get("WRITER_SOFT_MAX_PROB", "0.55"))
WRITER_SOFT_MARGIN = float(os.environ.get("WRITER_SOFT_MARGIN", "0.12"))

# Optional: path to prototype file; else we look for writer_prototypes.pt next to app.py
WRITER_PROTOTYPES_PATH_ENV = os.environ.get("WRITER_PROTOTYPES_PATH")
# Distance cutoff: same rule as part2 — inside group if best prototype distance < threshold.
# Set from notebook output "Chosen threshold (75th percentile): …" or bundle threshold in the .pt file.
WRITER_PROTO_DISTANCE_THRESHOLD_ENV = os.environ.get("WRITER_PROTO_DISTANCE_THRESHOLD")
WRITER_THRESHOLD_JSON_PATH_ENV = os.environ.get("WRITER_THRESHOLD_JSON_PATH")

# Bumped when weight-finding logic changes (see Status / terminal if debugging).
APP_LOADER_VERSION = "v3-shaikhas-charcnn-1152-notebook-segment"
APP_FILE_PATH = str(Path(__file__).resolve())

# Normalization for legacy 3-block CNN (if you load such a .pth): (x - mean) / std
NORM_MEAN = 0.5
NORM_STD = 0.5

# Set when model loads: Shaikha's CharCNN uses img/255 only (see notebook).
_preprocess_div255_only: bool = True

# Segmentation heuristics (tune if needed)
SEGMENT_BLUR = (3, 3)
MIN_AREA_RATIO = 0.002
PAD_RATIO = 0.15
MAX_IMAGE_SIDE = 1200

# `shaikhas_model.ipynb` word pipeline (segment_characters) — use for CharCNN / CharCNNLegacy.
NOTEBOOK_SEGMENT_BLUR = (5, 5)
NOTEBOOK_CHAR_BORDER = 10
NOTEBOOK_MIN_CHAR_HEIGHT = 8
NOTEBOOK_DEDUP_X_PX = 8
NOTEBOOK_DOT_MERGE_X_PX = 10

# Default class → character mapping for torchvision EMNIST "balanced" (47 classes).
# If your training used a different split/order, edit EMNIST_CHAR_MAP or set NUM_CLASSES_OVERRIDE.
EMNIST_CHAR_MAP_47 = list(
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abdefghnqrt"
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


# Flatten size after 3× MaxPool2d on 28×28 → 14×7 → 7×3 → 3×3 (padding=1 convs).
CHAR_CNN_V2_FLAT = 128 * 3 * 3  # 1152 — current `shaikhas_model.ipynb` CharCNN
# Legacy 2-block conv (no padding on first conv) → 12×12 → 6×6 → 64 * 5 * 5
CHAR_CNN_LEGACY_FLAT = 64 * 5 * 5  # 1600


class CharCNN(nn.Module):
    """
    Character CNN as in current `shaikhas_model.ipynb`: 3× conv + BatchNorm, dropout,
    1152 → 128 → `num_classes` (EMNIST letters / finetune).
    """

    def __init__(self, num_classes: int = 26) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(CHAR_CNN_V2_FLAT, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CharCNNLegacy(nn.Module):
    """Older 2-conv Shaikha checkpoint (64×5×5 = 1600-D flatten) — load-only compatibility."""

    def __init__(self, num_classes: int = 26) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(CHAR_CNN_LEGACY_FLAT, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EmnistCNN(nn.Module):
    """Alternate 3× conv + single Linear head (for generic EMNIST .pth files)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 3 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class WriterCNN(nn.Module):
    """Same architecture as `part2.ipynb` (writer identification)."""

    def __init__(self, num_writers: int = 5, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
        )
        self.classifier = nn.Linear(embedding_dim, num_writers)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        normalize_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.features(x)
        x = self.pool(x)
        embedding = self.embedding_layer(x)
        if normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def _is_char_cnn_state(state_dict: dict) -> bool:
    return any(k.startswith("conv.") for k in state_dict) and any(
        k.startswith("fc.") for k in state_dict
    )


def _infer_char_cnn_fc_in_features(state_dict: dict) -> int:
    """First Linear in `fc` is `fc.0` for both notebook and legacy CharCNN."""
    w = state_dict.get("fc.0.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return 0


def _detect_num_classes(state_dict: dict) -> int:
    # New CharCNN: fc.3 is Linear; legacy: fc.2 is Linear (fc.2 is Dropout in new).
    preference = (
        "fc.3.weight",
        "fc.2.weight",
        "classifier.weight",
        "fc.weight",
        "fc2.weight",
        "head.weight",
    )
    for key in preference:
        t = state_dict.get(key)
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            return int(t.shape[0])
    return len(EMNIST_CHAR_MAP_47)


def _build_model(state_dict: dict) -> nn.Module:
    n = _detect_num_classes(state_dict)
    if _is_char_cnn_state(state_dict):
        fin = _infer_char_cnn_fc_in_features(state_dict)
        if fin == CHAR_CNN_V2_FLAT:
            return CharCNN(num_classes=n)
        if fin == CHAR_CNN_LEGACY_FLAT:
            return CharCNNLegacy(num_classes=n)
        raise ValueError(
            f"Unsupported CharCNN checkpoint: fc.0 in_features={fin}, "
            f"expected {CHAR_CNN_V2_FLAT} (current notebook) or {CHAR_CNN_LEGACY_FLAT} (old 2-conv)."
        )
    return EmnistCNN(num_classes=n)


def _model_num_classes(model: nn.Module) -> int:
    if isinstance(model, CharCNN):
        last = model.fc[3]
        assert isinstance(last, nn.Linear)
        return int(last.out_features)
    if isinstance(model, CharCNNLegacy):
        last = model.fc[2]
        assert isinstance(last, nn.Linear)
        return int(last.out_features)
    if isinstance(model, EmnistCNN):
        return int(model.classifier.out_features)
    raise TypeError(f"Unknown model type: {type(model)}")


def _model_search_directories() -> List[Path]:
    """Folders to look for weights (app folder + cwd — fixes IDE / duplicate-copy issues)."""
    app_dir = Path(__file__).resolve().parent
    out: List[Path] = [app_dir]
    cwd = Path.cwd().resolve()
    if cwd != app_dir:
        out.append(cwd)
    # If you have an older duplicate app.py elsewhere, weights may still live here:
    known = Path.home() / "emnist-handwriting-space"
    if known.is_dir():
        k = known.resolve()
        if k not in out:
            out.append(k)
    return out


def _resolve_model_path() -> Optional[Path]:
    dirs = _model_search_directories()
    names = (
        "char_model_finetuned.pth",
        "char_model.pth",
        "char_model,pth",  # common typo: comma instead of dot before pth
        "model.pth",
        "shaikhas_model.pth",
    )

    if MODEL_PATH_ENV:
        raw = Path(MODEL_PATH_ENV)
        if raw.is_absolute():
            if raw.is_file():
                return raw
        else:
            for d in dirs:
                p = d / raw
                if p.is_file():
                    return p

    for d in dirs:
        for name in names:
            p = d / name
            if p.is_file():
                return p

    # Any single .pth in app dir or cwd (never use writer-only checkpoints for OCR)
    for d in dirs:
        matches = sorted(d.glob("*.pth"))
        if not matches:
            continue
        non_writer = [m for m in matches if "writer" not in m.name.lower()]
        if len(non_writer) == 1:
            return non_writer[0]
        if len(matches) == 1:
            m = matches[0]
            if "writer" not in m.name.lower():
                return m
            continue
        for m in non_writer:
            if "char" in m.name.lower() or "model" in m.name.lower():
                return m
        if non_writer:
            return non_writer[0]

    return None


def _resolve_writer_model_path() -> Optional[Path]:
    dirs = _model_search_directories()
    names = (
        "writer_model.pth",
        "writer_identification_model.pth",
        "best_writer_cnn_v2.pt",
    )
    if WRITER_MODEL_PATH_ENV:
        raw = Path(WRITER_MODEL_PATH_ENV)
        if raw.is_absolute():
            if raw.is_file():
                return raw
        else:
            for d in dirs:
                p = d / raw
                if p.is_file():
                    return p
    for d in dirs:
        for name in names:
            p = d / name
            if p.is_file():
                return p
    return None


def _resolve_writer_prototypes_path() -> Optional[Path]:
    dirs = _model_search_directories()
    if WRITER_PROTOTYPES_PATH_ENV:
        raw = Path(WRITER_PROTOTYPES_PATH_ENV)
        if raw.is_absolute() and raw.is_file():
            return raw
        for d in dirs:
            p = d / raw
            if p.is_file():
                return p
    for name in ("writer_prototypes.pt", "writer_prototypes.pth"):
        for d in dirs:
            p = d / name
            if p.is_file():
                return p
    return None


def _resolve_writer_threshold_json_path() -> Optional[Path]:
    """
    part2.ipynb writes a `writer_threshold.json` like:
      { "threshold": 4.01, "threshold_percentile": 85 }
    We load it if present (next to app.py/cwd) unless overridden by env.
    """
    dirs = _model_search_directories()
    if WRITER_THRESHOLD_JSON_PATH_ENV:
        raw = Path(WRITER_THRESHOLD_JSON_PATH_ENV)
        if raw.is_absolute() and raw.is_file():
            return raw
        for d in dirs:
            p = d / raw
            if p.is_file():
                return p
    for name in ("writer_threshold.json",):
        for d in dirs:
            p = d / name
            if p.is_file():
                return p
    return None


def _weights_missing_hint() -> str:
    lines = ["Searched for .pth files in:"]
    for d in _model_search_directories():
        try:
            files = [p.name for p in d.iterdir() if p.is_file()]
            lines.append(f"  • {d}")
            lines.append(f"    files: {', '.join(sorted(files)) if files else '(none)'}")
        except OSError as exc:
            lines.append(f"  • {d} (unreadable: {exc})")
    return "\n".join(lines)


def _unwrap_checkpoint(ckpt: Any) -> dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unsupported checkpoint format; expected a state_dict or nested dict.")


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


_model: Optional[nn.Module] = None
_model_error: Optional[str] = None


def get_model() -> Tuple[Optional[nn.Module], Optional[str]]:
    """Lazy-load model once; return (model, error_message)."""
    global _model, _model_error, _preprocess_div255_only
    if _model is not None:
        return _model, None

    path = _resolve_model_path()
    if path is None:
        return None, (
            "No weights file found.\n\n"
            + _weights_missing_hint()
            + "\n\nPut `char_model.pth` in the **same folder as app.py**, or set MODEL_PATH to the "
            "full path of your .pth file. Restart the app after moving the file."
        )

    try:
        raw = _load_checkpoint(path)
        state_dict = _unwrap_checkpoint(raw)
        model = _build_model(state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        _preprocess_div255_only = _is_char_cnn_state(state_dict)
        _model = model
        _model_error = None
        return _model, None
    except Exception as exc:  # noqa: BLE001 — surface any load issue to the UI
        _model_error = f"Failed to load model from {path.name}: {exc}"
        return None, _model_error


def char_map_for_classes(num_classes: int) -> List[str]:
    # EMNIST letters in the notebook: labels 0–25 → a–z
    if num_classes == 26:
        return list(string.ascii_lowercase)
    if num_classes <= len(EMNIST_CHAR_MAP_47):
        return EMNIST_CHAR_MAP_47[:num_classes]
    base = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )
    out = list(base)
    if num_classes > len(out):
        out.extend([f"[{i}]" for i in range(len(out), num_classes)])
    return out[:num_classes]


# ---------------------------------------------------------------------------
# Image → OpenCV / preprocessing / segmentation
# ---------------------------------------------------------------------------


def gradio_image_value_to_array(image: Any) -> Any:
    """Unwrap Gradio 5 ImageEditor dict to a numpy image (or pass through)."""
    if isinstance(image, dict):
        comp = image.get("composite")
        if comp is not None:
            return comp
        bg = image.get("background")
        if bg is not None:
            return bg
    return image


def to_gray_uint8(image: Any) -> Optional[np.ndarray]:
    """Convert Gradio image (numpy RGB/RGBA or grayscale) to H×W uint8."""
    if image is None:
        return None
    image = gradio_image_value_to_array(image)
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return None


def align_style_like_emnist(gray: np.ndarray) -> np.ndarray:
    """EMNIST digits are dark background, bright strokes. Heuristic invert for light backgrounds."""
    g = gray.astype(np.float32)
    if g.mean() > 127.0:
        return (255.0 - g).clip(0, 255).astype(np.uint8)
    return gray.astype(np.uint8)


def resize_long_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def segment_character_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Return bounding boxes (x, y, w, h) sorted left-to-right using OpenCV contours.
    """
    g = resize_long_side(gray, MAX_IMAGE_SIDE)
    g = align_style_like_emnist(g)
    blur = cv2.GaussianBlur(g, SEGMENT_BLUR, 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
    )

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = g.shape[:2]
    img_area = float(h * w)
    boxes: List[Tuple[int, int, int, int]] = []

    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < MIN_AREA_RATIO * img_area:
            continue
        if cw < 2 or ch < 2:
            continue
        boxes.append((x, y, cw, ch))

    boxes.sort(key=lambda b: b[0])
    return boxes


def crop_padded_square(
    gray: np.ndarray, box: Tuple[int, int, int, int], pad_ratio: float
) -> np.ndarray:
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * pad_ratio)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))

    H, W = gray.shape[:2]
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, W), min(y1, H)
    patch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    src = gray[y0c:y1c, x0c:x1c]
    dst_y = y0c - y0
    dst_x = x0c - x0
    patch[dst_y : dst_y + src.shape[0], dst_x : dst_x + src.shape[1]] = src
    return patch




def _patches_from_image_notebook(
    gray: np.ndarray,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Match `shaikhas_model.ipynb` `segment_characters` exactly:
    Gaussian Blur(5,5) -> BINARY_INV + OTSU -> i-dot merge -> padding -> square -> 28x28.

    Returns (characters, bounding_boxes) so the caller can compute inter-character
    gaps for space detection, exactly as the notebook's predict_word does.
    """
    # 1. Blur and threshold (exact notebook parameters)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Find and sort contours left → right
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # 3. STEP 1 from notebook: merge dot + stem (letter i/j)
    merged: List[np.ndarray] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        added = False
        for i, m in enumerate(merged):
            mx, my, mw, mh = cv2.boundingRect(m)
            if abs(x - mx) < NOTEBOOK_DOT_MERGE_X_PX and y < my:
                merged[i] = np.concatenate((m, cnt))
                added = True
                break
        if not added:
            merged.append(cnt)

    contours = merged

    # 4. STEP 2 from notebook: filter + extract
    characters: List[np.ndarray] = []
    valid_boxes: List[Tuple[int, int, int, int]] = []  # bounding boxes for space detection
    prev_x = -100

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h > NOTEBOOK_MIN_CHAR_HEIGHT:
            if abs(x - prev_x) < NOTEBOOK_DEDUP_X_PX:
                continue
            prev_x = x

            char = thresh[y : y + h, x : x + w]

            char = cv2.copyMakeBorder(
                char,
                NOTEBOOK_CHAR_BORDER, NOTEBOOK_CHAR_BORDER,
                NOTEBOOK_CHAR_BORDER, NOTEBOOK_CHAR_BORDER,
                cv2.BORDER_CONSTANT, value=0,
            )

            h_c, w_c = char.shape
            size = max(h_c, w_c)
            square = np.zeros((size, size), dtype=np.uint8)
            y_off = (size - h_c) // 2
            x_off = (size - w_c) // 2
            square[y_off : y_off + h_c, x_off : x_off + w_c] = char

            char_resized = cv2.resize(square, (28, 28))
            characters.append(char_resized)
            valid_boxes.append((x, y, w, h))  # original bbox for gap measurement

    # Fallback: treat entire image as one character
    if not characters:
        char = cv2.copyMakeBorder(
            thresh,
            NOTEBOOK_CHAR_BORDER, NOTEBOOK_CHAR_BORDER,
            NOTEBOOK_CHAR_BORDER, NOTEBOOK_CHAR_BORDER,
            cv2.BORDER_CONSTANT, value=0,
        )
        h_c, w_c = char.shape
        size = max(h_c, w_c)
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h_c) // 2
        x_off = (size - w_c) // 2
        square[y_off : y_off + h_c, x_off : x_off + w_c] = char
        characters.append(cv2.resize(square, (28, 28)))
        valid_boxes.append((0, 0, gray.shape[1], gray.shape[0]))

    return characters, valid_boxes


def align_style_like_emnist(gray_patch: np.ndarray) -> np.ndarray:
    """
    Match EMNIST preprocessing: 
    1. Binarize (INV+OTSU)
    2. Center/Crop
    3. Resize to 20x20
    4. Pad to 28x28
    5. Dilation (Added to match bold training data)
    """
    _, thresh = cv2.threshold(gray_patch, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Simple centering by bounding box
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    char = thresh[y:y+h, x:x+w]
    
    # Resize long side to 20px
    if h > w:
        new_h = 20
        new_w = int(w * (20 / h))
    else:
        new_w = 20
        new_h = int(h * (20 / w))
    
    if new_h < 1 or new_w < 1:
        return np.zeros((28, 28), dtype=np.uint8)
        
    resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to 28x28
    pad_h = (28 - new_h) // 2
    pad_w = (28 - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized, 
        pad_h, 28 - new_h - pad_h, 
        pad_w, 28 - new_w - pad_w, 
        cv2.BORDER_CONSTANT, value=0
    )

    # STEP 5: Neural Thickening (Dilation)
    # This helps thin pen strokes match the bold EMNIST training data.
    kernel = np.ones((2, 2), np.uint8)
    padded = cv2.dilate(padded, kernel, iterations=1)

    return padded


def _patches_from_image_classic(gray: np.ndarray) -> List[np.ndarray]:
    """Contour + EMNIST-style align (for generic EmnistCNN weights)."""
    boxes = segment_character_boxes(gray)
    base = resize_long_side(gray, MAX_IMAGE_SIDE)
    base = align_style_like_emnist(base)

    if not boxes:
        h, w = base.shape[:2]
        return [crop_padded_square(base, (0, 0, w, h), pad_ratio=0.08)]

    return [crop_padded_square(base, b, PAD_RATIO) for b in boxes]


def patches_from_image(
    gray: np.ndarray, *, notebook_style: bool
) -> Tuple[List[np.ndarray], Optional[List[Tuple[int, int, int, int]]]]:
    """
    Returns (patches, boxes_or_None).
    boxes is a list of (x, y, w, h) bounding boxes for notebook-style segmentation
    (used for space detection); None for the classic path.
    """
    if notebook_style:
        return _patches_from_image_notebook(gray)
    return _patches_from_image_classic(gray), None


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    """28×28 uint8 patch → float [0,1], shape (1,1,28,28). Resizes if not already 28×28."""
    if patch.shape[0] != 28 or patch.shape[1] != 28:
        resized = cv2.resize(patch, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        resized = patch
    x = resized.astype(np.float32) / 255.0
    if not _preprocess_div255_only:
        x = (x - NORM_MEAN) / NORM_STD
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    return t


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference_on_patches(
    patches: List[np.ndarray],
    model: nn.Module,
    mapping: List[str],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[str, List[List[Tuple[str, float]]], List[np.ndarray]]:
    """
    Returns (word, [[(char, confidence), ...], ...], patches)
    One list of confidences per character patch.
    """
    if not patches:
        return "", [], []

    batch = torch.cat([patch_to_tensor(p) for p in patches], dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

    # Store full distribution for top-3 display
    char_confs_all: List[List[Tuple[str, float]]] = []
    for i in range(len(preds)):
        p_probs = probs[i].cpu().numpy()
        sorted_indices = np.argsort(p_probs)[::-1]
        p_confs = []
        for idx in sorted_indices:
            p_confs.append((mapping[idx], float(p_probs[idx])))
        char_confs_all.append(p_confs)

    # Word string from top guesses
    word_chars = [mapping[p] for p in preds.cpu().numpy().tolist()]

    # Space Detection
    if boxes and len(boxes) > 1:
        # Sort characters by x-coordinate for gap analysis
        sorted_indices = np.argsort([b[0] for b in boxes])
        gaps: List[float] = []
        for i in range(1, len(sorted_indices)):
            prev_idx = sorted_indices[i-1]
            curr_idx = sorted_indices[i]
            px, py, pw, ph = boxes[prev_idx]
            cx, cy, cw, ch = boxes[curr_idx]
            gaps.append(float(cx - (px + pw)))

        if gaps:
            median_gap = float(np.median(gaps))
            max_gap = float(max(gaps))
            widths = [float(b[2]) for b in boxes]
            avg_width = sum(widths) / len(widths)
            space_threshold = max(median_gap * 1.5, avg_width * 0.6)
            gap_ratio_threshold = 0.4

            offset = 0
            for i, gap in enumerate(gaps):
                if gap > space_threshold and gap > max_gap * gap_ratio_threshold:
                    word_chars.insert(i + 1 + offset, " ")
                    offset += 1
        word = "".join(word_chars).strip()
    else:
        word = "".join(word_chars)

    return word, char_confs_all, patches


def _single_char_html(char_confs: List[Tuple[str, float]]) -> str:
    """Render a single character card (simplified)."""
    main_ch, main_conf = char_confs[0]
    pct = main_conf * 100

    if pct >= 90: colour = "#0ea5e9"
    elif pct >= 70: colour = "#f59e0b"
    else: colour = "#ef4444"

    return (
        f'<div class="char-result-card">'
        f'<span class="char-main" style="color:{colour};">{main_ch}</span>'
        f'<span class="char-pct">{pct:.1f}%</span>'
        f'</div>'
    )


def _char_conf_html(char_confs_list: List[List[Tuple[str, float]]]) -> str:
    """Render list of char_confs (one list per patch) as HTML."""
    if not char_confs_list:
        return "<p class='empty-hint'>No results yet.</p>"
    
    cards = [_single_char_html(c) for c in char_confs_list]
    return (
        '<div class="char-grid-wrap">'
        + "".join(cards)
        + "</div>"
    )


def predict_word(image: Any, mode_label: str) -> Tuple[str, str, List[np.ndarray]]:
    """Returns (predicted_word, per_char_html, patches)."""
    gray = to_gray_uint8(image)
    if gray is None:
        return "", "<p style='color:#888;'>No image provided. Please upload a photo first.</p>"

    model, err = get_model()
    if model is None or err:
        return "", f"<p style='color:red;'>{err or 'Model unavailable.'}</p>"

    num_classes = _model_num_classes(model)
    mapping = char_map_for_classes(num_classes)

    try:
        use_notebook_seg = isinstance(model, (CharCNN, CharCNNLegacy))
        patches, boxes = patches_from_image(gray, notebook_style=use_notebook_seg)
        word, char_confs, final_patches = run_inference_on_patches(patches, model, mapping, boxes=boxes)
        return word, _char_conf_html(char_confs), final_patches
    except Exception as exc:  # noqa: BLE001
        return "", f"<p style='color:red;'>Processing error: {exc}</p>"


# ---------------------------------------------------------------------------
# Writer identification (part2.ipynb)
# ---------------------------------------------------------------------------

_writer_model: Optional[WriterCNN] = None
_writer_idx_to_name: Dict[int, str] = {}
_writer_error: Optional[str] = None

_writer_proto_state: str = "init"  # init | ok | missing | err
_writer_proto_protos: Optional[Dict[int, np.ndarray]] = None
_writer_proto_threshold: Optional[float] = None
_writer_proto_err: Optional[str] = None


def _load_writer_checkpoint_raw(path: Path) -> Any:
    """Full pickle (checkpoint may contain writer_to_idx strings)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def gray_to_writer_tensor(gray: np.ndarray) -> torch.Tensor:
    # Match WriterWordDataset._load_image: trim margins, resize to 64x256 canvas, /255.
    # Do not invert by brightness here - training data is dark ink on white paper.
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    img_np = np.array(img)
    threshold = 250
    rows = np.where(np.min(img_np, axis=1) < threshold)[0]
    cols = np.where(np.min(img_np, axis=0) < threshold)[0]
    if len(rows) > 0 and len(cols) > 0:
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]
        img_np = img_np[top : bottom + 1, left : right + 1]
    img = Image.fromarray(img_np, mode="L")

    w, h = img.size
    scale = WRITER_TARGET_H / h
    new_w = max(1, int(round(w * scale)))
    new_h = WRITER_TARGET_H
    if new_w > WRITER_TARGET_W:
        scale = WRITER_TARGET_W / w
        new_w = WRITER_TARGET_W
        new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (WRITER_TARGET_W, WRITER_TARGET_H), color=255)
    paste_x = 0
    paste_y = (WRITER_TARGET_H - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))

    arr = np.array(canvas).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t


def get_writer_model() -> Tuple[Optional[WriterCNN], Optional[str]]:
    global _writer_model, _writer_idx_to_name, _writer_error
    if _writer_model is not None:
        return _writer_model, None

    path = _resolve_writer_model_path()
    if path is None:
        return None, (
            "Writer model not found. Add <code>writer_model.pth</code> next to <code>app.py</code> "
            "(from <code>part2.ipynb</code>) or set <code>WRITER_MODEL_PATH</code>."
        )

    try:
        ckpt = _load_writer_checkpoint_raw(path)
        emb_dim: Optional[int] = None
        state_dict: dict
        w2i: Dict[str, int] = {}

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            raw_map = ckpt.get("writer_to_idx")
            if isinstance(raw_map, dict):
                w2i = {str(k).strip().lower(): int(v) for k, v in raw_map.items()}
            if ckpt.get("embedding_dim") is not None:
                emb_dim = int(ckpt.get("embedding_dim"))
        elif isinstance(ckpt, dict) and any(
            k.startswith("features.") for k in ckpt
        ):
            state_dict = ckpt
        else:
            state_dict = _unwrap_checkpoint(ckpt)

        cw = state_dict.get("classifier.weight")
        if not isinstance(cw, torch.Tensor) or cw.ndim != 2:
            raise ValueError("Checkpoint missing classifier.weight")
        num_writers = int(cw.shape[0])
        # Infer embedding dim robustly (some checkpoints don't store `embedding_dim`).
        if emb_dim is None:
            emb_dim = int(cw.shape[1])
        # Validate against embedding layer weights if present.
        ew = state_dict.get("embedding_layer.1.weight")
        if isinstance(ew, torch.Tensor) and ew.ndim == 2:
            inferred = int(ew.shape[0])
            if inferred != emb_dim:
                emb_dim = inferred
        idx_ok = (
            w2i
            and len(w2i) == num_writers
            and set(w2i.values()) == set(range(num_writers))
        )
        if not idx_ok:
            w2i = {
                DEFAULT_WRITER_NAMES[i]: i
                for i in range(min(num_writers, len(DEFAULT_WRITER_NAMES)))
            }
            for j in range(len(w2i), num_writers):
                w2i[f"class_{j}"] = j

        # Row i of classifier = writer index i (must match checkpoint mapping)
        idx_to_name = dict(
            sorted(
                ((int(idx), str(name)) for name, idx in w2i.items()),
                key=lambda t: t[0],
            )
        )
        model = WriterCNN(num_writers=num_writers, embedding_dim=int(emb_dim))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        _writer_model = model
        _writer_idx_to_name = idx_to_name
        _writer_error = None
        return _writer_model, None
    except Exception as exc:  # noqa: BLE001
        _writer_error = f"Failed to load writer model from {path.name}: {exc}"
        return None, _writer_error


def _tensorish_to_vec(v: Any) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float64).ravel()
    return np.asarray(v, dtype=np.float64).ravel()


def _parse_prototypes_file(path: Path) -> Tuple[Dict[int, np.ndarray], Optional[float]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    thr_out: Optional[float] = None

    if isinstance(raw, dict) and "prototypes" in raw and isinstance(
        raw["prototypes"], dict
    ):
        blob = dict(raw["prototypes"])
        for key in ("threshold", "distance_threshold", "threshold_75"):
            if key in raw and raw[key] is not None:
                thr_out = float(raw[key])
                break
    elif isinstance(raw, dict):
        blob = {}
        for k, v in raw.items():
            if isinstance(k, str) and k.lower() in (
                "threshold",
                "distance_threshold",
                "threshold_75",
                "meta",
            ):
                if k.lower() != "meta" and v is not None:
                    thr_out = float(v)
                continue
            blob[k] = v
    else:
        raise ValueError("Prototype file must be a dict (optionally with a 'prototypes' key).")

    protos: Dict[int, np.ndarray] = {}
    for k, v in blob.items():
        if isinstance(k, str) and k.lower() in ("threshold", "distance_threshold"):
            thr_out = float(v)
            continue
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        protos[idx] = _tensorish_to_vec(v)

    if not protos:
        raise ValueError(
            "No prototypes loaded (expected int keys → vectors, as in part2 torch.save)."
        )

    return protos, thr_out


def _load_threshold_json(path: Path) -> Tuple[Optional[float], Optional[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to read {path.name}: {exc}"
    if not isinstance(obj, dict):
        return None, f"{path.name} must contain a JSON object."
    thr = obj.get("threshold")
    if thr is None:
        return None, f"{path.name} is missing key 'threshold'."
    try:
        return float(thr), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{path.name} threshold is not a number: {exc}"


def get_writer_prototypes() -> Tuple[
    Optional[Dict[int, np.ndarray]], Optional[float], Optional[str]
]:
    """
    Load writer_prototypes.pt once. Returns (protos, distance_threshold, error).
    If no file: (None, None, None). Threshold: from file, env, or default 4.5.
    """
    global _writer_proto_state, _writer_proto_protos, _writer_proto_threshold, _writer_proto_err
    if _writer_proto_state == "err":
        return None, None, _writer_proto_err
    if _writer_proto_state == "ok":
        return _writer_proto_protos, _writer_proto_threshold, None
    if _writer_proto_state == "missing":
        return None, None, None

    path = _resolve_writer_prototypes_path()
    if path is None:
        _writer_proto_state = "missing"
        return None, None, None
    try:
        protos, thr_file = _parse_prototypes_file(path)
        thr_env = float(WRITER_PROTO_DISTANCE_THRESHOLD_ENV) if WRITER_PROTO_DISTANCE_THRESHOLD_ENV else None
        thr_json: Optional[float] = None
        thr_json_err: Optional[str] = None
        thr_json_path = _resolve_writer_threshold_json_path()
        if thr_json_path is not None:
            thr_json, thr_json_err = _load_threshold_json(thr_json_path)
        if thr_json_err:
            _writer_proto_err = thr_json_err
            _writer_proto_state = "err"
            return None, None, thr_json_err

        thr = thr_env if thr_env is not None else (thr_file if thr_file is not None else (thr_json if thr_json is not None else 4.5))
        _writer_proto_protos = protos
        _writer_proto_threshold = thr
        _writer_proto_state = "ok"
        return protos, thr, None
    except Exception as exc:  # noqa: BLE001
        _writer_proto_err = str(exc)
        _writer_proto_state = "err"
        return None, None, str(exc)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _predict_writer_by_prototype(
    embedding: np.ndarray, prototypes: Dict[int, np.ndarray]
) -> Tuple[int, float, Dict[int, float]]:
    distances = {lbl: _euclidean(embedding, p) for lbl, p in prototypes.items()}
    best_lbl = int(min(distances, key=distances.get))
    return best_lbl, distances[best_lbl], distances


def _writer_proto_verdict_html(
    inside: bool,
    nearest: str,
    best_dist: float,
    threshold: float,
    all_dist: Dict[int, float],
    idx_to_name: Dict[int, str],
) -> str:
    table = _proto_distances_table_html(all_dist, idx_to_name, threshold)
    if inside:
        return (
            '<div style="background:#ecfdf5;border:2px solid #059669;padding:16px;'
            'border-radius:12px;margin-bottom:14px;">'
            '<div style="font-size:1.15rem;font-weight:800;color:#065f46;">'
            "In the trained writer group</div>"
            '<div style="margin-top:8px;color:#064e3b;line-height:1.5;">'
            f"Closest match: <b>{nearest}</b><br/>"
            f"Embedding distance to that prototype: <b>{best_dist:.4f}</b> "
            f"(cutoff &lt; <b>{threshold:.4f}</b> — same rule as <code>part2.ipynb</code>)"
            "</div></div>"
            + table
        )
    return (
        '<div style="background:#fee2e2;border:2px solid #dc2626;padding:16px;'
        'border-radius:12px;margin-bottom:14px;">'
        '<div style="font-size:1.15rem;font-weight:800;color:#991b1b;">'
        "Outside the trained writer group</div>"
        '<div style="margin-top:8px;color:#7f1d1d;line-height:1.5;">'
        f"The nearest trained writer would be <b>{nearest}</b>, "
        f"but distance <b>{best_dist:.4f}</b> is not below the cutoff "
        f"(<b>{threshold:.4f}</b>), so this is treated as "
        "<b>not one of the five training writers</b>."
        "</div></div>"
        + table
    )


def _proto_distances_table_html(
    distances: Dict[int, float],
    idx_to_name: Dict[int, str],
    threshold: float,
) -> str:
    items = sorted(distances.items(), key=lambda x: x[1])
    rows = []
    for lbl, d in items:
        nm = idx_to_name.get(int(lbl), str(lbl))
        ok = "✓" if d < threshold else "✗"
        rows.append(
            f"<tr><td>{ok}</td><td><b>{nm}</b></td>"
            f"<td style='text-align:right;font-family:monospace'>{d:.4f}</td></tr>"
        )
    return (
        "<div style='margin-top:4px;'><b>Distances to each prototype</b> "
        f"(in-group needs best &lt; {threshold:.4f})</div>"
        "<table style='width:100%;max-width:440px;border-collapse:collapse;"
        "margin-top:8px;font-size:0.9rem;'>"
        "<thead><tr><th></th><th>Writer</th><th style='text-align:right'>L2</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _writer_topk_html(probs: torch.Tensor, idx_to_name: Dict[int, str], k: int = 5) -> str:
    k = min(k, probs.numel())
    topv, topi = probs.topk(k)
    rows = []
    for rank, (val, idx) in enumerate(zip(topv.tolist(), topi.tolist()), start=1):
        name = idx_to_name.get(int(idx), f"#{idx}")
        pct = val * 100.0
        bar_w = int(min(100, max(4, pct)))
        rows.append(
            f"<div style='margin:8px 0;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:0.9rem;'>"
            f"<span>{rank}. <b>{name}</b></span><span>{pct:.1f}%</span></div>"
            f"<div style='height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;'>"
            f"<div style='width:{bar_w}%;height:100%;background:#4f46e5;'></div></div></div>"
        )
    return '<div style="max-width:420px;">' + "".join(rows) + "</div>"


def predict_writer_ui(image: Any) -> Tuple[str, str]:
    """
    Returns (short summary line, HTML).

    If `writer_prototypes.pt` exists: **in / out of group** uses embedding distance vs threshold
    (same rule as part2: in group if best distance < threshold).

    Otherwise: softmax-only heuristics + note to add prototypes.
    """
    gray = to_gray_uint8(image)
    if gray is None:
        return "", "<p style='color:#888;'>Upload a handwriting image first.</p>"

    model, err = get_writer_model()
    if model is None or err:
        return "", f"<p style='color:red;'>{err}</p>"

    try:
        batch = gray_to_writer_tensor(gray)
        with torch.no_grad():
            logits, emb = model(batch, return_embedding=True)
            probs = F.softmax(logits, dim=1)[0]

        emb_np = emb[0].detach().cpu().numpy().ravel()
        protos, proto_thr, proto_err = get_writer_prototypes()

        parts: List[str] = []

        if proto_err:
            parts.append(
                f"<p style='color:#b91c1c;padding:10px;background:#fef2f2;border-radius:8px;'>"
                f"<b>Prototype file error:</b> {proto_err}</p>"
            )

        headline = ""

        if protos is not None and proto_thr is not None and not proto_err:
            best_lbl, best_dist, all_dist = _predict_writer_by_prototype(emb_np, protos)
            nearest = _writer_idx_to_name.get(best_lbl, str(best_lbl))
            inside = best_dist < proto_thr
            headline = (
                f"{'In group: ' + nearest if inside else 'Outside group'} "
                f"(dist {best_dist:.3f}, cutoff {proto_thr:.3f})"
            )
            parts.append(
                _writer_proto_verdict_html(
                    inside, nearest, best_dist, proto_thr, all_dist, _writer_idx_to_name
                )
            )
            parts.append(
                "<p style='font-size:0.82rem;color:#444;margin:10px 0 6px 0;'>"
                "<b>Classifier softmax</b> (always picks one of five — use the verdict above for in/out):</p>"
            )
        else:
            sorted_p, sorted_i = probs.sort(descending=True)
            p1 = float(sorted_p[0].item())
            p2 = float(sorted_p[1].item()) if sorted_p.numel() > 1 else 0.0
            margin = p1 - p2
            pred_i = int(sorted_i[0].item())
            name = _writer_idx_to_name.get(pred_i, f"writer_{pred_i}")
            likely_unknown = p1 < WRITER_UNK_MAX_PROB
            uncertain = likely_unknown or (
                p1 < WRITER_SOFT_MAX_PROB and margin < WRITER_SOFT_MARGIN
            )
            if likely_unknown:
                headline = f"Heuristic: likely not trained (softmax → {name}, {p1*100:.0f}%)"
            elif uncertain:
                headline = f"Heuristic: uncertain (best softmax {name}, {p1*100:.0f}%)"
            else:
                headline = f"Softmax best: {name} ({p1*100:.1f}%) — add prototypes for in/out verdict"

            if protos is None and not proto_err:
                parts.append(
                    "<div style='background:#fffbeb;border:1px solid #f59e0b;padding:12px;"
                    "border-radius:10px;margin-bottom:12px;color:#92400e;'>"
                    "<b>No <code>writer_prototypes.pt</code> found</b> next to <code>app.py</code>. "
                    "Place the file from part2 (or set <code>WRITER_PROTOTYPES_PATH</code>) "
                    "to show a clear <b>inside / outside the training group</b> verdict. "
                    "Optional: set <code>WRITER_PROTO_DISTANCE_THRESHOLD</code> to your notebook’s "
                    "75th-percentile value.</div>"
                )

        parts.append(_writer_topk_html(probs, _writer_idx_to_name, k=5))
        return headline, "\n".join(parts)
    except Exception as exc:  # noqa: BLE001
        return "", f"<p style='color:red;'>Writer prediction error: {exc}</p>"


def predict_transcription_and_writer(image: Any) -> Tuple[str, str, str, str, List[np.ndarray]]:
    """One upload -> OCR + writer (same image)."""
    word, char_html, patches = predict_word(image, "Upload")
    wh, whtml = predict_writer_ui(image)
    return word, char_html, wh, whtml, patches


# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read image to numpy
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
             return jsonify({"error": "Invalid image file"}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run analysis
        raw_word, char_html, writer_headline, writer_html, patches = predict_transcription_and_writer(img_rgb)

        # Dictionary correction
        words = raw_word.split()
        corrected_words = []
        for w in words:
            # Handle empty or single-char words
            if len(w) <= 1:
                corrected_words.append(w)
                continue
            
            # Get best correction
            cor = spell.correction(w)
            # Use original if no correction found or if it's name-like (Capitalized)
            if cor and cor.lower() != w.lower():
                # Preserve capitalization if original was capitalized
                if w[0].isupper():
                    cor = cor.capitalize()
                corrected_words.append(cor)
            else:
                corrected_words.append(w)
        
        corrected_word = " ".join(corrected_words)

        # Convert patches to base64
        encoded_patches = []
        for p in patches:
            _, buffer = cv2.imencode('.png', p)
            encoded_patches.append(base64.b64encode(buffer).decode('utf-8'))

        return jsonify({
            "word": corrected_word,
            "raw_word": raw_word,
            "char_html": char_html,
            "writer_headline": writer_headline,
            "writer_html": writer_html,
            "patches": encoded_patches
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    _wp = _resolve_model_path()
    _ww = _resolve_writer_model_path()
    _wproto = _resolve_writer_prototypes_path()
    _wthr = _resolve_writer_threshold_json_path()
    
    print("=" * 72)
    print(f"EMNIST handwriting app (Flask) | {APP_LOADER_VERSION}")
    print(f"OCR weights:     {_wp if _wp else '(not found)'}")
    print(f"Writer weights:  {_ww if _ww else '(not found)'}")
    print("=" * 72)

    print("Warming up OCR model...", end=" ", flush=True)
    _m, _me = get_model()
    print("OK" if _m else f"WARN: {_me}")
    
    print("Warming up writer model...", end=" ", flush=True)
    _wm, _wme = get_writer_model()
    print("OK" if _wm else f"WARN: {_wme}")
    
    get_writer_prototypes()
    print("=" * 72)

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)

