"""PaddleOCR 管线仪屏幕识别核心库。

适用于特定型号管线探测仪（管线仪）的 LCD 屏幕信息解析。
设备信息为占位符，部署前请根据实际型号更新 config/ 标定。
"""

from meter_ocr.core import (
    is_calibrated,
    load_calib_config,
    recognize,
    save_calib_config,
)

__all__ = ["is_calibrated", "load_calib_config", "recognize", "save_calib_config"]
