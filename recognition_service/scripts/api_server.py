"""管线仪屏幕识别 HTTP API。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from screen_ocr import is_calibrated, load_calib_config, recognize, save_calib_config  # noqa: E402
from screen_ocr import core as ocr_core  # noqa: E402

app = FastAPI(
    title="Pipeline Locator Screen API",
    description="OCR API for a specific pipeline locator model (<MODEL_NAME>).",
    version="1.0.0",
)

CALIB = load_calib_config()
OCR_LOCK = threading.Lock()
RUNTIME_STATE: dict[str, Any] = {}


@app.on_event("startup")
def preload_runtime() -> None:
    global RUNTIME_STATE

    if not is_calibrated(CALIB):
        RUNTIME_STATE = {"ready": False}
        return

    with OCR_LOCK:
        RUNTIME_STATE = {
            "ready": True,
            **ocr_core.preload_runtime(),
        }


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
        "model_loaded": RUNTIME_STATE.get("ready", False),
        "runtime": RUNTIME_STATE,
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
    global CALIB, RUNTIME_STATE

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

    with OCR_LOCK:
        RUNTIME_STATE = {
            "ready": is_calibrated(CALIB),
            **ocr_core.reload_runtime(),
        }

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
        raise HTTPException(status_code=503, detail="OCR is not calibrated")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    img = _decode_image(content)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    with OCR_LOCK:
        result = recognize(
            img,
            CALIB["rois"],
            CALIB["compass"],
            debug=debug,
            frame_id=frame_id,
        )

    return result


def main() -> None:
    import os

    host = os.environ.get("SCREEN_OCR_API_HOST", "127.0.0.1")
    port = int(os.environ.get("SCREEN_OCR_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
