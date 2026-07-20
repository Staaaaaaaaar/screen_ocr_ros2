"""基于标定切片的七段数码管模板匹配识别。"""
from __future__ import annotations

import json
import re
from typing import Any

import cv2
import numpy as np

from screen_ocr.paths import DIGIT_SLOTS_CONFIG, TEMPLATE_DIR

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass

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

_DIGIT_KEY_RE = re.compile(r"^\d$")
_INTENSITY_TEMPLATE_RE = re.compile(r"^(\d_i)(?:_\d+)?$")
_PLAIN_DIGIT_TEMPLATE_RE = re.compile(r"^(\d)(?:_\d+)?$")
_DOT_TEMPLATE_RE = re.compile(r"^(dot)(?:_\d+)?$")
_MATCH_THRESHOLD = 0.55
_LOW_CONTRAST_OPTIONAL_THRESHOLD = 18.0
_OPTIONAL_INTENSITY_SCORE_THRESHOLD = 0.80
_FAST_INTENSITY_SCORE_THRESHOLD = 0.80
_EARLY_INTENSITY_SCORE_THRESHOLD = 0.80
_OPTIONAL_INTENSITY_ABSENT_SCORE_THRESHOLD = 0.88
_DIGIT_SLOTS_CACHE: dict[str, Any] | None = None
_DIGIT_VECTOR_CACHE: dict[tuple[tuple[str, tuple[int, ...]], ...], list[tuple[str, np.ndarray]]] = {}
_FLAT_DIGIT_VECTOR_CACHE: dict[
    tuple[tuple[str, tuple[int, ...]], ...], tuple[list[str], np.ndarray]
] = {}
_CANDIDATE_BOX_CACHE: dict[tuple[str, str, tuple[float, float, float, float]], list[list[float]]] = {}
_INTENSITY_PRIORITY_OFFSETS = [
    (-0.04, -0.04, 0.0),
    (-0.02, -0.04, 0.0),
    (0.0, -0.02, 0.0),
    (0.04, 0.02, 0.02),
    (0.04, 0.04, 0.02),
    (0.04, 0.04, 0.0),
    (-0.04, -0.04, 0.02),
    (0.02, -0.04, 0.0),
    (-0.02, -0.02, 0.0),
    (0.0, 0.0, 0.0),
]


def load_digit_slots() -> dict[str, Any]:
    global _DIGIT_SLOTS_CACHE
    if _DIGIT_SLOTS_CACHE is not None:
        return _DIGIT_SLOTS_CACHE

    if DIGIT_SLOTS_CONFIG.exists():
        with open(DIGIT_SLOTS_CONFIG, encoding="utf-8") as f:
            _DIGIT_SLOTS_CACHE = json.load(f)
            return _DIGIT_SLOTS_CACHE

    _DIGIT_SLOTS_CACHE = _DEFAULT_SLOTS
    return _DIGIT_SLOTS_CACHE


def _normalize_cell(cell: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    gray = cv2.resize(gray, (32, 48), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _crop_ratio(roi: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = roi.shape[:2]
    x1, y1, x2, y2 = box
    return roi[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def _slot_to_box(slot: Any) -> list[float]:
    if isinstance(slot, dict):
        return slot["box"]
    if len(slot) == 2:
        return [slot[0], 0.0, slot[1], 1.0]
    return slot


def _candidate_boxes(box: list[float], field: str, slot_name: str = "") -> list[list[float]]:
    if field != "intensity":
        return [box]

    cache_key = (field, slot_name, tuple(round(value, 6) for value in box))
    if cache_key in _CANDIDATE_BOX_CACHE:
        return _CANDIDATE_BOX_CACHE[cache_key]

    x1, y1, x2, y2 = box
    boxes: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    x_offsets = (-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04) if slot_name == "d1" else (
        -0.04,
        -0.02,
        0.0,
        0.02,
        0.04,
    )
    for dx1 in x_offsets:
        for dx2 in x_offsets:
            for dy1 in (0.0, 0.02, 0.03, 0.04):
                candidate = [
                    _clamp_ratio(x1 + dx1),
                    _clamp_ratio(y1 + dy1),
                    _clamp_ratio(x2 + dx2),
                    _clamp_ratio(y2),
                ]
                if candidate[2] - candidate[0] >= 0.18 and candidate[3] - candidate[1] >= 0.55:
                    key = tuple(round(value, 4) for value in candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    boxes.append(candidate)

    if slot_name != "d1":
        priority = {offset: idx for idx, offset in enumerate(_INTENSITY_PRIORITY_OFFSETS)}
        boxes.sort(
            key=lambda candidate: (
                priority.get(
                    (
                        round(candidate[0] - x1, 2),
                        round(candidate[2] - x2, 2),
                        round(candidate[1] - y1, 2),
                    ),
                    999,
                ),
                abs(candidate[0] - x1) + abs(candidate[2] - x2) + abs(candidate[1] - y1) * 2,
                abs((candidate[2] - candidate[0]) - (x2 - x1)),
            )
        )

    _CANDIDATE_BOX_CACHE[cache_key] = boxes
    return boxes


def _cell_contrast(cell: np.ndarray) -> float:
    if cell.size == 0:
        return 0.0

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    foreground = gray[binary > 0]
    background = gray[binary == 0]
    if len(foreground) < 10 or len(background) < 10:
        return 0.0

    return float(background.mean() - foreground.mean())


def _has_decimal_point(roi: np.ndarray, box: list[float]) -> bool:
    dot_roi = _crop_ratio(roi, box)
    if dot_roi is None or dot_roi.size == 0:
        return False

    gray = cv2.cvtColor(dot_roi, cv2.COLOR_BGR2GRAY) if dot_roi.ndim == 3 else dot_roi
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    h, w = binary.shape[:2]
    min_area = max(2.0, w * h * 0.01)
    max_area = max(6.0, w * h * 0.45)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 2 or ch < 2:
            continue

        aspect = cw / max(ch, 1)
        if not 0.35 <= aspect <= 2.8:
            continue

        center_y = y + ch / 2
        if center_y < h * 0.35:
            continue

        return True

    return False


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


def _template_key_from_stem(stem: str) -> str | None:
    if match := _INTENSITY_TEMPLATE_RE.fullmatch(stem):
        return match.group(1)
    if match := _PLAIN_DIGIT_TEMPLATE_RE.fullmatch(stem):
        return match.group(1)
    if match := _DOT_TEMPLATE_RE.fullmatch(stem):
        return match.group(1)
    return None


def load_digit_templates() -> dict[str, list[np.ndarray]]:
    if not TEMPLATE_DIR.is_dir():
        return {}

    templates: dict[str, list[np.ndarray]] = {}
    for path in sorted(TEMPLATE_DIR.glob("*.png")):
        digit = _template_key_from_stem(path.stem)
        if digit is None:
            continue

        tpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        templates.setdefault(digit, []).append(cv2.resize(tpl, (32, 48), interpolation=cv2.INTER_AREA))
    return templates


def _templates_for_field(
    templates: dict[str, list[np.ndarray]],
    field: str,
) -> dict[str, list[np.ndarray]]:
    field_templates: dict[str, list[np.ndarray]] = {}

    if field == "intensity":
        for key, items in templates.items():
            if key.endswith("_i") and _DIGIT_KEY_RE.fullmatch(key[0]):
                field_templates[key[0]] = items

        if field_templates:
            return field_templates

    for key, items in templates.items():
        if _DIGIT_KEY_RE.fullmatch(key):
            field_templates[key] = items

    if field != "intensity" and "dot" in templates:
        field_templates["dot"] = templates["dot"]

    return field_templates


def ensure_digit_templates(img: np.ndarray, rois: dict[str, list[int]]) -> dict[str, list[np.ndarray]]:
    templates = load_digit_templates()
    if templates:
        return templates

    templates = build_digit_templates(img, rois)
    save_digit_templates(templates)
    return templates


def _best_digit_match(
    cell: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> tuple[str | None, float]:
    if cell.size == 0:
        return None, -1.0

    normalized = _normalize_cell(cell)
    norm = _match_vector(normalized)
    if norm is None:
        return None, -1.0

    labels, matrix = _flat_template_vectors(templates)
    if not labels:
        return None, -1.0

    scores = matrix @ norm
    best_index = int(np.argmax(scores))
    return labels[best_index], float(scores[best_index])


def _template_vector_key(
    templates: dict[str, list[np.ndarray]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((digit, tuple(id(tpl) for tpl in items)) for digit, items in sorted(templates.items()))


def _match_vector(image: np.ndarray) -> np.ndarray | None:
    vector = image.astype(np.float32).reshape(-1)
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return vector / norm


def _template_vectors(templates: dict[str, list[np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    key = _template_vector_key(templates)
    if key in _DIGIT_VECTOR_CACHE:
        return _DIGIT_VECTOR_CACHE[key]

    vectors: list[tuple[str, np.ndarray]] = []
    for digit, items in templates.items():
        if not _DIGIT_KEY_RE.fullmatch(digit):
            continue

        digit_vectors = []
        for tpl in items:
            vector = _match_vector(tpl)
            if vector is not None:
                digit_vectors.append(vector)

        if digit_vectors:
            vectors.append((digit, np.stack(digit_vectors)))

    _DIGIT_VECTOR_CACHE[key] = vectors
    return vectors


def _flat_template_vectors(
    templates: dict[str, list[np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    """Flatten template vectors so each cell uses one matrix multiply."""
    key = _template_vector_key(templates)
    if key in _FLAT_DIGIT_VECTOR_CACHE:
        return _FLAT_DIGIT_VECTOR_CACHE[key]

    labels: list[str] = []
    vectors: list[np.ndarray] = []
    for digit, digit_vectors in _template_vectors(templates):
        for vector in digit_vectors:
            labels.append(digit)
            vectors.append(vector)

    if vectors:
        matrix = np.stack(vectors).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, 32 * 48), dtype=np.float32)

    result = (labels, matrix)
    _FLAT_DIGIT_VECTOR_CACHE[key] = result
    return result


def _ordered_offsets(values: list[float] | tuple[float, ...]) -> list[float]:
    """Try the calibrated position first, then nearby fallback positions."""
    return sorted(values, key=lambda value: (abs(float(value)), float(value)))


def _match_digit(cell: np.ndarray, templates: dict[str, list[np.ndarray]]) -> str | None:
    best_digit, best_score = _best_digit_match(cell, templates)
    return best_digit if best_score >= _MATCH_THRESHOLD else None


def _match_digit_slot(
    roi: np.ndarray,
    box: list[float],
    field: str,
    templates: dict[str, list[np.ndarray]],
    slot_name: str = "",
) -> tuple[str | None, float, float]:
    cell = _crop_ratio(roi, box)
    best_digit, best_score = _best_digit_match(cell, templates)
    best_contrast = _cell_contrast(cell)

    if field == "intensity" and best_score >= _FAST_INTENSITY_SCORE_THRESHOLD:
        return best_digit, best_score, best_contrast

    for candidate in _candidate_boxes(box, field, slot_name):
        if candidate == box:
            continue

        cell = _crop_ratio(roi, candidate)
        digit, score = _best_digit_match(cell, templates)
        if score > best_score:
            best_digit = digit
            best_score = score
            best_contrast = _cell_contrast(cell)
            if field == "intensity" and score >= _EARLY_INTENSITY_SCORE_THRESHOLD:
                break

    if best_score < _MATCH_THRESHOLD:
        return None, best_score, best_contrast

    return best_digit, best_score, best_contrast


def _match_dot(cell: np.ndarray, templates: dict[str, list[np.ndarray]]) -> bool:
    dot_templates = templates.get("dot", [])
    if cell.size == 0 or not dot_templates:
        return False

    norm = _normalize_cell(cell)
    best_score = -1.0

    for tpl in dot_templates:
        score = float(cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED).max())
        best_score = max(best_score, score)

    return best_score >= _MATCH_THRESHOLD


def _format_value(
    digit_values: dict[str, str],
    dot_after: str,
    allowed_formats: list[str],
) -> str:
    d1 = digit_values.get("d1", "")
    d2 = digit_values.get("d2", "")
    d3 = digit_values.get("d3", "")

    if dot_after == "d1":
        value = f"{d1}.{d2}{d3}"
    elif dot_after == "d2":
        value = f"{d1}{d2}.{d3}" if d1 else f"{d2}.{d3}"
    else:
        return ""

    if allowed_formats:
        value_format = re.sub(r"\d", "D", value)
        if value_format not in allowed_formats:
            return ""

    return value


def _format_fixed_decimal_before_last(
    digit_values: dict[str, str],
    allowed_formats: list[str],
) -> str:
    digits = "".join(digit_values[key] for key in sorted(digit_values.keys()))
    if len(digits) < 2:
        return ""

    value = f"{digits[:-1]}.{digits[-1]}"
    if allowed_formats:
        value_format = re.sub(r"\d", "D", value)
        if value_format not in allowed_formats:
            return ""

    return value


def _read_aligned_integer(
    roi: np.ndarray,
    slots: list[dict[str, Any]],
    templates: dict[str, list[np.ndarray]],
    config: dict[str, Any],
) -> str:
    digit_slots = [slot for slot in slots if slot.get("type", "digit") == "digit"]
    threshold = float(config.get("match_threshold", _MATCH_THRESHOLD))
    x_offsets = _ordered_offsets(config.get("x_offsets", [0.0]))
    y_offsets = _ordered_offsets(config.get("y_offsets", [0.0]))

    best_digits: list[str] = []
    best_score = -1.0
    best_offset = (0.0, 0.0)

    for dx in x_offsets:
        for dy in y_offsets:
            digits: list[str] = []
            scores: list[float] = []

            for slot in digit_slots:
                x1, y1, x2, y2 = _slot_to_box(slot)
                box = [
                    _clamp_ratio(x1 + dx),
                    _clamp_ratio(y1 + dy),
                    _clamp_ratio(x2 + dx),
                    _clamp_ratio(y2 + dy),
                ]
                digit, score = _best_digit_match(_crop_ratio(roi, box), templates)
                if digit is None or score < threshold:
                    break
                digits.append(digit)
                scores.append(score)

            average_score = sum(scores) / len(scores) if scores else 0.0
            quality = len(digits) * 2.0 + average_score
            if quality > best_score:
                best_digits = digits
                best_score = quality
                best_offset = (dx, dy)

    trailing_contrast_threshold = float(config.get("trailing_contrast_threshold", 15.0))
    dx, dy = best_offset
    for slot in digit_slots[len(best_digits) :]:
        x1, y1, x2, y2 = _slot_to_box(slot)
        box = [
            _clamp_ratio(x1 + dx),
            _clamp_ratio(y1 + dy),
            _clamp_ratio(x2 + dx),
            _clamp_ratio(y2 + dy),
        ]
        if _cell_contrast(_crop_ratio(roi, box)) >= trailing_contrast_threshold:
            return ""

    return "".join(best_digits)


def _read_aligned_fixed_decimal(
    roi: np.ndarray,
    slots: list[dict[str, Any]],
    templates: dict[str, list[np.ndarray]],
    config: dict[str, Any],
) -> str:
    digit_slots = [slot for slot in slots if slot.get("type", "digit") == "digit"]
    threshold = float(config.get("match_threshold", _MATCH_THRESHOLD))
    x_offsets = _ordered_offsets(config.get("x_offsets", [0.0]))
    y_offsets = _ordered_offsets(config.get("y_offsets", [0.0]))
    leading_contrast_threshold = float(config.get("leading_contrast_threshold", 15.0))
    leading_box = _slot_to_box(digit_slots[0])
    has_leading_digit = _cell_contrast(_crop_ratio(roi, leading_box)) >= leading_contrast_threshold
    patterns = [digit_slots if has_leading_digit else digit_slots[-2:]]

    best_digits: list[str] = []
    best_quality = -1.0

    for pattern in patterns:
        for dx in x_offsets:
            for dy in y_offsets:
                digits: list[str] = []
                scores: list[float] = []

                for slot in pattern:
                    x1, y1, x2, y2 = _slot_to_box(slot)
                    box = [
                        _clamp_ratio(x1 + dx),
                        _clamp_ratio(y1 + dy),
                        _clamp_ratio(x2 + dx),
                        _clamp_ratio(y2 + dy),
                    ]
                    digit, score = _best_digit_match(_crop_ratio(roi, box), templates)
                    if digit is None or score < threshold:
                        break
                    digits.append(digit)
                    scores.append(score)

                if len(digits) != len(pattern):
                    continue

                quality = sum(scores) / len(scores)
                if quality > best_quality:
                    best_digits = digits
                    best_quality = quality

    if len(best_digits) not in {2, 3}:
        return ""
    if best_quality < float(config.get("minimum_average_score", threshold)):
        return ""
    return f"{''.join(best_digits[:-1])}.{best_digits[-1]}"


def read_meter_digits(
    roi: np.ndarray,
    field: str,
    templates: dict[str, list[np.ndarray]],
) -> str:
    if roi is None or roi.size == 0 or not templates:
        return ""

    templates = _templates_for_field(templates, field)
    if not templates:
        return ""

    config = load_digit_slots().get(field, {})
    top_ratio = config.get("top_ratio", 0.8)
    slots = config.get("slots", [])
    decimal_after = config.get("decimal_after")

    roi = roi[: int(roi.shape[0] * top_ratio), :]
    if config.get("integer_only", False):
        return _read_aligned_integer(roi, slots, templates, config)
    if config.get("aligned_fixed_decimal", False):
        return _read_aligned_fixed_decimal(roi, slots, templates, config)

    if slots and isinstance(slots[0], dict):
        digit_values: dict[str, str] = {}
        dot_after: str | None = None

        for idx, slot in enumerate(slots):
            slot_type = slot.get("type", "digit")
            box = _slot_to_box(slot)
            cell = _crop_ratio(roi, box)

            if slot_type == "digit":
                name = slot.get("name", f"d{idx + 1}")
                required = slot.get("required", True)

                digit, score, contrast = _match_digit_slot(roi, box, field, templates, name)
                if digit is None:
                    if required:
                        return ""
                    continue
                digit_values[name] = digit

            elif slot_type == "dot":
                if _match_dot(cell, templates):
                    dot_after = slot.get("after")

        if config.get("fixed_decimal_before_last", False):
            return _format_fixed_decimal_before_last(
                digit_values,
                config.get("allowed_formats", []),
            )

        if dot_after is None:
            if config.get("decimal_required", False):
                return ""

            return "".join(digit_values[key] for key in sorted(digit_values.keys()))

        return _format_value(
            digit_values,
            dot_after,
            config.get("allowed_formats", []),
        )

    digits: list[str] = []
    for slot in slots:
        cell = _crop_ratio(roi, _slot_to_box(slot))
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
