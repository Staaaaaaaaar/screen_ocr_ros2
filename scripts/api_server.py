"""PaddleOCR 管线仪屏幕识别 HTTP API。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from meter_ocr import is_calibrated, load_calib_config, recognize, save_calib_config  # noqa: E402

app = FastAPI(
    title="PaddleOCR Pipeline Locator Screen API",
    description="OCR API for a specific pipeline locator model (<MODEL_NAME>).",
    version="1.0.0",
)

CALIB = load_calib_config()
OCR_LOCK = threading.Lock()


class CompassConfig(BaseModel):
    center: list[int] = Field(..., min_length=2, max_length=2)
    radius: int = Field(..., gt=0)


class CalibConfig(BaseModel):
    rois: dict[str, list[int]]
    compass: CompassConfig


def _decode_image(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "calibrated": is_calibrated(CALIB),
    }


@app.get("/v1/config")
def get_config() -> dict[str, Any]:
    return {
        "rois": CALIB.get("rois"),
        "compass": CALIB.get("compass"),
        "calibrated": is_calibrated(CALIB),
    }


@app.put("/v1/config")
def put_config(config: CalibConfig) -> dict[str, Any]:
    global CALIB

    required_rois = {
        "intensity",
        "current",
        "depth",
        "arrow_left",
        "arrow_right",
    }
    missing = required_rois - set(config.rois.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing ROI keys: {sorted(missing)}",
        )

    for name, rect in config.rois.items():
        if len(rect) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"ROI '{name}' must have 4 integers [x1, y1, x2, y2]",
            )

    rois = config.rois
    compass = config.compass.model_dump()
    save_calib_config(rois, compass)
    CALIB = {"rois": rois, "compass": compass}

    return {
        "success": True,
        "calibrated": True,
        "config": CALIB,
    }


@app.post("/v1/recognize")
async def recognize_api(
    image: UploadFile = File(...),
    debug: bool = Form(False),
    frame_id: str = Form(""),
) -> dict[str, Any]:
    if not is_calibrated(CALIB):
        return {
            "success": False,
            "error": {
                "code": "NOT_CALIBRATED",
                "message": "Missing config files. Use PUT /v1/config to upload calibration.",
            },
        }

    content = await image.read()
    if not content:
        return {
            "success": False,
            "error": {
                "code": "EMPTY_IMAGE",
                "message": "Uploaded image is empty",
            },
        }

    img = _decode_image(content)
    if img is None:
        return {
            "success": False,
            "error": {
                "code": "INVALID_IMAGE",
                "message": "Failed to decode image. Use jpg/png/bmp.",
            },
        }

    t0 = time.perf_counter()
    with OCR_LOCK:
        result = recognize(
            img,
            CALIB["rois"],
            CALIB["compass"],
            debug=debug,
            frame_id=frame_id,
        )
    cost_ms = (time.perf_counter() - t0) * 1000
    fps = 1000 / cost_ms if cost_ms > 0 else 0

    meta = {
        "process_time_ms": round(cost_ms, 1),
        "fps": round(fps, 2),
        "frame_id": frame_id,
    }

    if result is None:
        return {
            "success": False,
            "error": {
                "code": "RECOGNITION_FAILED",
                "message": "Compass or signal strength not detected",
            },
            "meta": meta,
        }

    return {
        "success": True,
        "data": result,
        "meta": meta,
    }


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
