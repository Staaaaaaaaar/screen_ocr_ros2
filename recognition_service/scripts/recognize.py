#!/usr/bin/env python3
"""管线仪屏幕识别：读入截图，输出 JSON。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from screen_ocr import is_calibrated, load_calib_config, recognize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="管线仪屏幕识别（型号: <MODEL_NAME>）",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(ROOT_DIR / "examples" / "images" / "image0000001.png"),
        help="图片路径（示例集为 image0000001.png ~ image0000047.png）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="保存一张带 ROI 标注的调试图到 output/",
    )
    args = parser.parse_args()

    config = load_calib_config()
    if not is_calibrated(config):
        print("缺少 config/rois.json 或 config/compass.json", file=sys.stderr)
        return 1

    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取图片: {args.image}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    result = recognize(
        img,
        config["rois"],
        config["compass"],
        debug=args.debug,
        frame_id="cli",
    )
    cost_ms = (time.perf_counter() - t0) * 1000

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"耗时: {cost_ms:.0f} ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
