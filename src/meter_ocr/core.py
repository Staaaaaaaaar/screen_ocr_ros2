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

from meter_ocr.digit_reader import (
    ensure_digit_templates,
    read_meter_digits,
    read_percent_display,
)
from meter_ocr.paths import COMPASS_CONFIG, OUTPUT_DIR, ROI_CONFIG

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("ppocr").setLevel(logging.ERROR)

_TEMPLATE_CACHE: dict[str, list] | None = None


def _get_templates(img: np.ndarray, rois: dict) -> dict:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = ensure_digit_templates(img, rois)
    return _TEMPLATE_CACHE


@dataclass
class DebugContext:
    enabled: bool = False
    ocr_dir: str = ""
    arrow_dir: str = ""
    compass_dir: str = ""
    img_cnt: int = 1

    def reset(self):
        self.img_cnt = 1

    def setup(self, tag: str = "api"):
        if not self.enabled:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() * 1000) % 1000)
        folder_name = f"{timestamp}_{ms}_{tag}"

        self.ocr_dir = str(OUTPUT_DIR / "ocr_preprocess_img" / folder_name)
        self.arrow_dir = str(OUTPUT_DIR / "debug_arrow" / folder_name)
        self.compass_dir = str(OUTPUT_DIR / "compass_debug" / folder_name)

        os.makedirs(self.ocr_dir, exist_ok=True)
        os.makedirs(self.arrow_dir, exist_ok=True)
        os.makedirs(self.compass_dir, exist_ok=True)
        self.reset()


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


def _save_debug_image(path: str, image: np.ndarray, debug: DebugContext) -> None:
    if debug.enabled and path:
        cv2.imwrite(path, image)


def ocr_signal_strength(roi: np.ndarray, templates: dict, debug: DebugContext | None = None) -> str:
    value = read_percent_display(roi, templates)
    return value if value else "none"


def ocr_integer_value(roi: np.ndarray, field: str, templates: dict) -> str:
    value = read_meter_digits(roi, field, templates)
    return value if value else "none"


def arrow_detect(
    img: np.ndarray,
    roi_right: list[int],
    roi_left: list[int],
    debug: DebugContext | None = None,
) -> str:
    debug = debug or DebugContext()

    def detect_arrow_direction(roi: np.ndarray, debug_name: str = "debug") -> str | None:
        if roi is None or roi.size == 0:
            return None

        _save_debug_image(
            os.path.join(debug.arrow_dir, f"{debug_name}_0_roi.png") if debug.enabled else "",
            roi,
            debug,
        )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        _save_debug_image(
            os.path.join(debug.arrow_dir, f"{debug_name}_4_morph.png") if debug.enabled else "",
            morph,
            debug,
        )

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)
        if area < 20:
            return None

        x, y, w, h = cv2.boundingRect(best_cnt)
        if w < 8 or h < 5 or w / max(h, 1) < 1.2:
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


def get_compass_info(img: np.ndarray, compass_config: dict, debug: DebugContext | None = None) -> str:
    debug = debug or DebugContext()

    try:
        cx, cy = compass_config["center"]
        r = compass_config["radius"]

        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(img.shape[1], cx + r)
        y2 = min(img.shape[0], cy + r)

        roi = img[y1:y2, x1:x2].copy()
        if roi is None or roi.size == 0:
            return "none"

        _save_debug_image(
            os.path.join(debug.compass_dir, "1_roi.jpg") if debug.enabled else "",
            roi,
            debug,
        )

        roi_cx = cx - x1
        roi_cy = cy - y1

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _save_debug_image(
            os.path.join(debug.compass_dir, "2_gray_img.jpg") if debug.enabled else "",
            gray,
            debug,
        )

        edges = cv2.Canny(gray, 40, 120)
        _save_debug_image(
            os.path.join(debug.compass_dir, "3_edges.jpg") if debug.enabled else "",
            edges,
            debug,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=int(r * 0.35),
            maxLineGap=8,
        )
        if lines is None:
            return "none"

        best_line = None
        best_score = -1

        for line in lines:
            xA, yA, xB, yB = line[0]
            length = np.hypot(xB - xA, yB - yA)
            if length < r * 0.3:
                continue

            center_dist = point_line_distance(roi_cx, roi_cy, xA, yA, xB, yB)
            if center_dist > r * 0.25:
                continue

            dA = np.hypot(xA - roi_cx, yA - roi_cy)
            dB = np.hypot(xB - roi_cx, yB - roi_cy)
            if max(dA, dB) < r * 0.35:
                continue

            score = length - center_dist * 2
            if score > best_score:
                best_score = score
                best_line = (xA, yA, xB, yB)

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

        if debug.enabled:
            final_img = roi.copy()
            cv2.circle(final_img, (int(roi_cx), int(roi_cy)), 4, (0, 0, 255), -1)
            cv2.line(
                final_img,
                (int(roi_cx), int(roi_cy)),
                (int(tip_x), int(tip_y)),
                (0, 255, 0),
                3,
            )
            _save_debug_image(
                os.path.join(debug.compass_dir, "4_final_perfect_line.jpg"),
                final_img,
                debug,
            )

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


def prepare_image(img: np.ndarray) -> np.ndarray:
    """仪表盘实拍图通常为镜像，水平翻转后再识别。"""
    return cv2.flip(img, 1)


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
    intensity = ocr_signal_strength(intensity_roi, templates, debug=debug_ctx)

    current_roi = img[rois["current"][1] : rois["current"][3], rois["current"][0] : rois["current"][2]]
    current = ocr_integer_value(current_roi, "current", templates)

    depth_roi = img[rois["depth"][1] : rois["depth"][3], rois["depth"][0] : rois["depth"][2]]
    depth = ocr_integer_value(depth_roi, "depth", templates)

    arrow = arrow_detect(img, rois["arrow_right"], rois["arrow_left"], debug=debug_ctx)
    compass_angle = get_compass_info(img, compass, debug=debug_ctx)

    if compass_angle == "none" or intensity == "none":
        return None

    return {
        "signal_strength": intensity,
        "pipeline_current": f"{current} mA",
        "burial_depth": f"{depth} m",
        "arrow_direction": normalize_arrow(arrow),
        "compass_angle": compass_angle,
        "compass_angle_deg": parse_compass_angle(compass_angle),
    }
