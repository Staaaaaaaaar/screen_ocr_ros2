"""基于标定切片的七段数码管模板匹配识别。"""
from __future__ import annotations

import json
import re
from typing import Any

import cv2
import numpy as np

from meter_ocr.paths import DIGIT_SLOTS_CONFIG, TEMPLATE_DIR

_DEFAULT_SLOTS = {
    "current": {
        "top_ratio": 0.72,
        "slots": [[0.03, 0.22], [0.25, 0.44], [0.47, 0.66]],
        "decimal_after": None,
    },
    "depth": {
        "top_ratio": 0.82,
        "slots": [[0.03, 0.28], [0.32, 0.58], [0.62, 0.90]],
        "decimal_after": None,
    },
    "intensity": {
        "top_ratio": 0.92,
        "slots": [[0.04, 0.30], [0.42, 0.52], [0.58, 0.84]],
        "decimal_after": 1,
    },
}

_REFERENCE_CELLS = {
    "3": ("current", [0.03, 0.10, 0.22, 0.72]),
    "5": ("current", [0.25, 0.10, 0.44, 0.72]),
    "0": ("current", [0.47, 0.10, 0.66, 0.72]),
    "1": ("depth", [0.03, 0.08, 0.28, 0.78]),
    "4": ("depth", [0.32, 0.08, 0.58, 0.78]),
}


def load_digit_slots() -> dict[str, Any]:
    if DIGIT_SLOTS_CONFIG.exists():
        with open(DIGIT_SLOTS_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return _DEFAULT_SLOTS


def _normalize_cell(cell: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    gray = cv2.resize(gray, (32, 48), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _crop_ratio(roi: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = roi.shape[:2]
    x1, y1, x2, y2 = box
    return roi[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def build_digit_templates(img: np.ndarray, rois: dict[str, list[int]]) -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {}
    for digit, (field, box) in _REFERENCE_CELLS.items():
        x1, y1, x2, y2 = rois[field]
        roi = img[y1:y2, x1:x2]
        cell = _crop_ratio(roi, box)
        templates.setdefault(digit, []).append(_normalize_cell(cell))
    return templates


def save_digit_templates(templates: dict[str, list[np.ndarray]]) -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    for digit, items in templates.items():
        for idx, tpl in enumerate(items):
            suffix = "" if idx == 0 else f"_{idx}"
            cv2.imwrite(str(TEMPLATE_DIR / f"{digit}{suffix}.png"), tpl)


def load_digit_templates() -> dict[str, list[np.ndarray]]:
    if not TEMPLATE_DIR.is_dir():
        return {}

    templates: dict[str, list[np.ndarray]] = {}
    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        digit = path.stem.split("_")[0]
        tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        templates.setdefault(digit, []).append(cv2.resize(tpl, (32, 48), interpolation=cv2.INTER_AREA))
    return templates


def ensure_digit_templates(img: np.ndarray, rois: dict[str, list[int]]) -> dict[str, list[np.ndarray]]:
    templates = load_digit_templates()
    if templates:
        return templates

    templates = build_digit_templates(img, rois)
    save_digit_templates(templates)
    return templates


def _match_digit(cell: np.ndarray, templates: dict[str, list[np.ndarray]]) -> str | None:
    if cell.size == 0:
        return None

    norm = _normalize_cell(cell)
    best_digit = None
    best_score = -1.0
    for digit, items in templates.items():
        for tpl in items:
            score = float(cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_score = score
                best_digit = digit

    return best_digit if best_score >= 0.55 else None


def read_meter_digits(
    roi: np.ndarray,
    field: str,
    templates: dict[str, list[np.ndarray]],
) -> str:
    if roi is None or roi.size == 0 or not templates:
        return ""

    config = load_digit_slots().get(field, {})
    top_ratio = config.get("top_ratio", 0.8)
    slots = config.get("slots", [])
    decimal_after = config.get("decimal_after")

    roi = roi[: int(roi.shape[0] * top_ratio), :]
    digits: list[str] = []
    for slot in slots:
        if len(slot) == 2:
            cell = _crop_ratio(roi, [slot[0], 0.0, slot[1], 1.0])
        else:
            cell = _crop_ratio(roi, slot)
        digit = _match_digit(cell, templates)
        if digit is None:
            return ""
        digits.append(digit)

    if decimal_after is not None and 0 < decimal_after < len(digits):
        return f"{''.join(digits[:decimal_after])}.{''.join(digits[decimal_after:])}"

    return "".join(digits)


def read_percent_display(
    roi: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> str:
    value = read_meter_digits(roi, "intensity", templates)
    if not value:
        return ""
    if not re.fullmatch(r"\d+\.\d+", value):
        return ""
    return f"{value}%"
