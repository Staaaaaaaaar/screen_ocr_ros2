import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from screen_ocr.digit_reader import (
    ensure_digit_templates,
    load_digit_templates,
    read_meter_digits,
    read_percent_display,
)
from screen_ocr.paths import COMPASS_CONFIG, OUTPUT_DIR, ROI_CONFIG

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("ppocr").setLevel(logging.ERROR)

_TEMPLATE_CACHE: dict[str, list] | None = None


def _get_templates(img: np.ndarray, rois: dict) -> dict:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = ensure_digit_templates(img, rois)
    return _TEMPLATE_CACHE


def preload_runtime() -> dict[str, bool]:
    """Preload reusable digit templates for the HTTP process."""
    global _TEMPLATE_CACHE

    loaded_templates = load_digit_templates()
    if loaded_templates:
        _TEMPLATE_CACHE = loaded_templates

    return {
        "templates_loaded": bool(_TEMPLATE_CACHE),
    }


def reset_template_cache() -> None:
    global _TEMPLATE_CACHE
    _TEMPLATE_CACHE = None


def reload_runtime() -> dict[str, bool]:
    reset_template_cache()
    return preload_runtime()


_ROI_COLORS: dict[str, tuple[int, int, int]] = {
    "intensity": (0, 255, 0),
    "current": (255, 128, 0),
    "depth": (255, 0, 128),
    "arrow_left": (0, 255, 255),
    "arrow_right": (255, 255, 0),
}


@dataclass
class DebugContext:
    enabled: bool = False
    output_path: str = ""

    def setup(self, tag: str = "api"):
        if not self.enabled:
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() * 1000) % 1000)
        self.output_path = str(OUTPUT_DIR / f"{timestamp}_{ms}_{tag}.jpg")


def load_calib_config() -> dict[str, Any]:
    config = {"rois": None, "compass": None}

    if ROI_CONFIG.exists():
        with open(ROI_CONFIG, encoding="utf-8") as f:
            config["rois"] = json.load(f)

    if COMPASS_CONFIG.exists():
        with open(COMPASS_CONFIG, encoding="utf-8") as f:
            config["compass"] = json.load(f)

    return config


def save_calib_config(rois: dict, compass: dict) -> None:
    CONFIG_DIR = ROI_CONFIG.parent
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROI_CONFIG, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=4, ensure_ascii=False)

    with open(COMPASS_CONFIG, "w", encoding="utf-8") as f:
        json.dump(compass, f, indent=4, ensure_ascii=False)


def is_calibrated(config: dict[str, Any]) -> bool:
    return bool(config.get("rois") and config.get("compass"))


def _format_debug_value(result: dict[str, Any], key: str) -> str:
    value = result.get(key)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _draw_roi_overlay(
    img: np.ndarray,
    rois: dict,
    compass: dict,
    result: dict[str, Any],
) -> np.ndarray:
    overlay = img.copy()
    result_labels = {
        "intensity": f"signal_strength={_format_debug_value(result, 'signal_strength_percent')}",
        "current": f"current={_format_debug_value(result, 'current_milliamps')}mA",
        "depth": f"depth={_format_debug_value(result, 'depth_meters')}m",
        "arrow_left": f"left={_format_debug_value(result, 'left_arrow')}",
        "arrow_right": f"right={_format_debug_value(result, 'right_arrow')}",
    }

    for name, rect in rois.items():
        x1, y1, x2, y2 = rect
        color = _ROI_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {result_labels.get(name, '')}".strip()
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    center_x, center_y = compass["center"]
    radius = int(compass["radius"])
    compass_color = (255, 0, 255)
    cv2.circle(overlay, (int(center_x), int(center_y)), radius, compass_color, 2)
    heading_label = (
        f"compass heading={_format_debug_value(result, 'pipeline_heading_degrees')}deg"
    )
    cv2.putText(
        overlay,
        heading_label,
        (max(int(center_x - radius), 0), max(int(center_y - radius) - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        compass_color,
        2,
        cv2.LINE_AA,
    )

    return overlay


def _save_debug_overlay(
    img: np.ndarray,
    rois: dict,
    compass: dict,
    result: dict[str, Any],
    debug: DebugContext,
) -> None:
    if not debug.enabled or not debug.output_path:
        return
    overlay = _draw_roi_overlay(img, rois, compass, result)
    cv2.imwrite(debug.output_path, overlay)


def ocr_signal_strength(roi: np.ndarray, templates: dict) -> str:
    value = read_percent_display(roi, templates)
    return value if value else "none"


def ocr_integer_value(roi: np.ndarray, field: str, templates: dict) -> str:
    value = read_meter_digits(roi, field, templates)
    return value if value else "none"


def arrow_detect(
    img: np.ndarray,
    roi_right: list[int],
    roi_left: list[int],
) -> str:
    def detect_arrow_direction(roi: np.ndarray, debug_name: str = "debug") -> str | None:
        if roi is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)
        if area < 20:
            return None

        x, y, w, h = cv2.boundingRect(best_cnt)
        if w < 8 or h < 5 or w / max(h, 1) < 0.6:
            return None
        if h > roi.shape[0] * 0.9 and y <= 2:
            return None
        if y <= 2 and w > roi.shape[1] * 0.75:
            return None

        return "→" if "RIGHT" in debug_name else "←"

    right_roi = img[roi_right[1] : roi_right[3], roi_right[0] : roi_right[2]]
    left_roi = img[roi_left[1] : roi_left[3], roi_left[0] : roi_left[2]]

    right_result = detect_arrow_direction(right_roi, "RIGHT_ARROW")
    left_result = detect_arrow_direction(left_roi, "LEFT_ARROW")

    if right_result == "→" and left_result == "←":
        return "→←"
    if right_result == "→":
        return "→"
    if left_result == "←":
        return "←"
    return "none"


def point_line_distance(px, py, x1, y1, x2, y2) -> float:
    line_len = np.hypot(x2 - x1, y2 - y1)
    if line_len == 0:
        return np.hypot(px - x1, py - y1)

    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_len


def _locate_compass_geometry(
    gray: np.ndarray, compass_config: dict
) -> tuple[int, int, int]:
    """Locate the compass circle in the current frame before detecting its needle."""
    configured_cx, configured_cy = compass_config["center"]
    configured_r = int(compass_config["radius"])

    if not compass_config.get("refine_geometry", True):
        return int(configured_cx), int(configured_cy), configured_r

    # The hub is near the calibrated center. Searching only this small region
    # keeps HoughCircles fast while returning the same dynamic center.
    hub_margin = max(32, int(configured_r * 0.4))
    x1 = max(0, int(configured_cx - hub_margin))
    y1 = max(0, int(configured_cy - hub_margin))
    x2 = min(gray.shape[1], int(configured_cx + hub_margin))
    y2 = min(gray.shape[0], int(configured_cy + hub_margin))
    search = gray[y1:y2, x1:x2]
    if search.size == 0:
        return int(configured_cx), int(configured_cy), configured_r

    # The small hub remains visible when the outer compass ring is low contrast.
    hub_blurred = cv2.medianBlur(search, 3)
    for param2 in (30, 25, 20):
        hub_circles = cv2.HoughCircles(
            hub_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(20, int(configured_r * 0.23)),
            param1=80,
            param2=param2,
            minRadius=9,
            maxRadius=28,
        )
        if hub_circles is None:
            continue
        hub_candidates = []
        for circle_x, circle_y, _ in hub_circles[0]:
            absolute_x = float(circle_x + x1)
            absolute_y = float(circle_y + y1)
            distance = np.hypot(
                absolute_x - configured_cx, absolute_y - configured_cy
            )
            if distance <= configured_r * 0.6:
                hub_candidates.append((distance, absolute_x, absolute_y))
        if hub_candidates:
            _, center_x, center_y = min(hub_candidates, key=lambda item: item[0])
            return int(round(center_x)), int(round(center_y)), configured_r

    # Rare low-contrast frames fall back to the wider outer-ring search.
    search_margin = max(35, int(configured_r * 0.55))
    x1 = max(0, int(configured_cx - configured_r - search_margin))
    y1 = max(0, int(configured_cy - configured_r - search_margin))
    x2 = min(gray.shape[1], int(configured_cx + configured_r + search_margin))
    y2 = min(gray.shape[0], int(configured_cy + configured_r + search_margin))
    search = gray[y1:y2, x1:x2]
    if search.size == 0:
        return int(configured_cx), int(configured_cy), configured_r

    blurred = cv2.medianBlur(search, 5)
    candidates: list[tuple[float, float, float, float]] = []
    for param2 in (25, 21, 17):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(30, int(configured_r * 0.4)),
            param1=80,
            param2=param2,
            minRadius=max(40, int(configured_r * 0.72)),
            maxRadius=int(configured_r * 1.3),
        )
        if circles is None:
            continue
        for circle_x, circle_y, radius in circles[0]:
            absolute_x = float(circle_x + x1)
            absolute_y = float(circle_y + y1)
            center_distance = np.hypot(
                absolute_x - configured_cx, absolute_y - configured_cy
            )
            if center_distance <= configured_r * 0.6:
                candidates.append(
                    (
                        center_distance + abs(float(radius) - configured_r) * 1.5,
                        absolute_x,
                        absolute_y,
                        float(radius),
                    )
                )

    if not candidates:
        return int(configured_cx), int(configured_cy), configured_r

    _, center_x, center_y, radius = min(candidates, key=lambda item: item[0])
    return int(round(center_x)), int(round(center_y)), int(round(radius))


def _estimate_compass_axis(
    gray: np.ndarray, center_x: int, center_y: int, radius: int
) -> float | None:
    """Estimate the needle axis from edge support on both sides of the hub."""
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(gray.shape[1], center_x + radius)
    y2 = min(gray.shape[0], center_y + radius)
    compass_gray = gray[y1:y2, x1:x2]
    if compass_gray.size == 0:
        return None

    edges = cv2.Canny(cv2.GaussianBlur(compass_gray, (3, 3), 0), 5, 15)
    local_center_x = center_x - x1
    local_center_y = center_y - y1
    distances = np.arange(
        max(18, int(radius * 0.22)), int(radius * 0.86), dtype=np.float32
    )
    offsets = np.arange(-3, 4, dtype=np.float32)
    angles = np.arange(0.0, 180.0, 0.5, dtype=np.float32)
    angle_rad = np.deg2rad(angles)
    unit_x = np.sin(angle_rad)[:, None, None]
    unit_y = -np.cos(angle_rad)[:, None, None]
    normal_x = np.cos(angle_rad)[:, None, None]
    normal_y = np.sin(angle_rad)[:, None, None]
    distance_grid = distances[None, :, None]
    offset_grid = offsets[None, None, :]

    def side_support(side: float) -> np.ndarray:
        sample_x = (
            local_center_x
            + side * unit_x * distance_grid
            + normal_x * offset_grid
        )
        sample_y = (
            local_center_y
            + side * unit_y * distance_grid
            + normal_y * offset_grid
        )
        sample_x = np.clip(
            np.rint(sample_x).astype(np.int32),
            0,
            edges.shape[1] - 1,
        )
        sample_y = np.clip(
            np.rint(sample_y).astype(np.int32),
            0,
            edges.shape[0] - 1,
        )
        sampled = edges[sample_y, sample_x]
        return (sampled.max(axis=2) > 0).sum(axis=1)

    support = np.minimum(side_support(1.0), side_support(-1.0))
    best_index = int(np.argmax(support))
    best_support = int(support[best_index])
    best_angle = float(angles[best_index])

    if best_support < int(radius * 0.46):
        return None

    angle_rad = np.deg2rad(best_angle)
    unit_x = np.sin(angle_rad)
    unit_y = -np.cos(angle_rad)
    yy, xx = np.nonzero(edges)
    relative_x = xx.astype(np.float32) - local_center_x
    relative_y = yy.astype(np.float32) - local_center_y
    distance_from_center = np.hypot(relative_x, relative_y)
    distance_from_axis = np.abs(relative_x * unit_y - relative_y * unit_x)
    line_mask = (
        (distance_from_center >= distances[0])
        & (distance_from_center < distances[-1])
        & (distance_from_axis <= 5.0)
    )
    line_points = np.column_stack(
        [xx[line_mask].astype(np.float32), yy[line_mask].astype(np.float32)]
    )
    if len(line_points) >= 10:
        vx, vy, _, _ = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if vy > 0:
            vx, vy = -vx, -vy
        signed_clockwise_angle = np.degrees(np.arctan2(vx, -vy))
        axis_angle = (
            signed_clockwise_angle
            if signed_clockwise_angle >= 0
            else 180.0 + signed_clockwise_angle
        )
        return float(round(axis_angle))

    heading = -best_angle if best_angle <= 90.0 else -(best_angle - 180.0)
    return float(round(heading))


def get_compass_info(img: np.ndarray, compass_config: dict) -> str:
    try:
        configured_cx, configured_cy = compass_config["center"]
        configured_r = int(compass_config["radius"])
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cx, cy, r = _locate_compass_geometry(
            gray_full,
            {"center": [configured_cx, configured_cy], "radius": configured_r},
        )

        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(img.shape[1], cx + r)
        y2 = min(img.shape[0], cy + r)

        roi = img[y1:y2, x1:x2].copy()
        if roi is None or roi.size == 0:
            return "none"

        roi_cx = cx - x1
        roi_cy = cy - y1

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # The radial axis check uses both needle halves and avoids unrelated Hough lines.
        heading = _estimate_compass_axis(gray_full, cx, cy, r)
        if heading is None:
            return "none"
        return f"{int(heading)}°"

        def find_best_line(
            edge_img: np.ndarray,
            threshold: int,
            min_line_ratio: float,
            max_line_gap: int,
            center_dist_ratio: float,
            tip_dist_ratio: float,
        ) -> tuple[int, int, int, int] | None:
            lines = cv2.HoughLinesP(
                edge_img,
                1,
                np.pi / 180,
                threshold=threshold,
                minLineLength=int(r * min_line_ratio),
                maxLineGap=max_line_gap,
            )
            if lines is None:
                return None

            best_line = None
            best_score = -1
            for line in lines:
                xA, yA, xB, yB = line[0]
                length = np.hypot(xB - xA, yB - yA)
                if length < r * min_line_ratio:
                    continue

                center_dist = point_line_distance(roi_cx, roi_cy, xA, yA, xB, yB)
                if center_dist > r * center_dist_ratio:
                    continue

                dA = np.hypot(xA - roi_cx, yA - roi_cy)
                dB = np.hypot(xB - roi_cx, yB - roi_cy)
                if max(dA, dB) < r * tip_dist_ratio:
                    continue

                score = length - center_dist * 2
                if score > best_score:
                    best_score = score
                    best_line = (xA, yA, xB, yB)

            return best_line

        edges = cv2.Canny(gray, 40, 120)

        best_line = find_best_line(edges, 25, 0.35, 8, 0.25, 0.35)
        if best_line is None:
            enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            enhanced_edges = cv2.Canny(enhanced, 20, 70)
            best_line = find_best_line(enhanced_edges, 14, 0.25, 12, 0.32, 0.28)

        if best_line is None:
            return "none"

        xA, yA, xB, yB = best_line
        dA = np.hypot(xA - roi_cx, yA - roi_cy)
        dB = np.hypot(xB - roi_cx, yB - roi_cy)
        tip_x, tip_y = (xA, yA) if dA > dB else (xB, yB)

        dx = tip_x - roi_cx
        dy = tip_y - roi_cy
        angle = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360
        if angle > 180:
            angle -= 180
        angle = round(angle)

        return f"{angle}°"
    except Exception as e:
        print(f"罗盘异常: {e}")
        return "none"


def normalize_arrow(arrow: str) -> str:
    mapping = {
        "→": "right",
        "←": "left",
        "→←": "both",
        "none": "none",
    }
    return mapping.get(arrow, "none")


def parse_compass_angle(compass_angle: str) -> float | None:
    if compass_angle == "none":
        return None
    match = re.search(r"(\d+)", compass_angle)
    return float(match.group(1)) if match else None


def parse_display_number(value: str) -> float | None:
    if not value or value == "none":
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def parse_display_integer(value: str) -> int | None:
    number = parse_display_number(value)
    return int(number) if number is not None else None


def normalize_pipeline_heading(angle: float | None) -> float | None:
    if angle is None:
        return None
    clockwise_angle = angle if angle <= 90 else angle - 180
    return float(round(-clockwise_angle))


def prepare_image(img: np.ndarray) -> np.ndarray:
    """Use the input image directly; ROIs are calibrated on the original frame."""
    return img


def recognize(
    img: np.ndarray,
    rois: dict,
    compass: dict,
    debug: bool = False,
    frame_id: str = "",
) -> dict[str, Any] | None:
    debug_ctx = DebugContext(enabled=debug)
    debug_ctx.setup(tag=frame_id or "api")

    img = prepare_image(img)
    templates = _get_templates(img, rois)

    intensity_roi = img[rois["intensity"][1] : rois["intensity"][3], rois["intensity"][0] : rois["intensity"][2]]
    intensity = ocr_signal_strength(intensity_roi, templates)

    current_roi = img[rois["current"][1] : rois["current"][3], rois["current"][0] : rois["current"][2]]
    current = ocr_integer_value(current_roi, "current", templates)

    depth_roi = img[rois["depth"][1] : rois["depth"][3], rois["depth"][0] : rois["depth"][2]]
    depth = ocr_integer_value(depth_roi, "depth", templates)

    arrow = normalize_arrow(
        arrow_detect(img, rois["arrow_right"], rois["arrow_left"])
    )
    compass_angle = get_compass_info(img, compass)

    heading = normalize_pipeline_heading(parse_compass_angle(compass_angle))

    result = {
        "signal_strength_percent": parse_display_number(intensity),
        "depth_meters": parse_display_number(depth),
        "current_milliamps": parse_display_integer(current),
        "pipeline_heading_degrees": heading,
        "left_arrow": arrow in {"left", "both"},
        "right_arrow": arrow in {"right", "both"},
    }
    _save_debug_overlay(img, rois, compass, result, debug_ctx)
    return result
