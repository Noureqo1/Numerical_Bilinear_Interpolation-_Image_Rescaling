"""
=============================================================================
  Bilinear Interpolation -- FastAPI Microservice Backend
  -------------------------------------------------------
  A standalone REST API that exposes the core numerical interpolation
  algorithms and error metrics as HTTP endpoints.

  This service is designed to be consumed by ANY client (Streamlit,
  React, CLI, mobile, etc.) -- it has zero UI dependencies.

  ENDPOINTS
  ~~~~~~~~~
    GET  /health                -> {"status": "ok"}
    POST /upscale               -> upscaled image (base64) + timing
    POST /evaluate              -> MSE / PSNR / MAE metrics (JSON)

  HOW TO RUN
  ~~~~~~~~~~
    pip install fastapi uvicorn python-multipart opencv-python numpy
    uvicorn api:app --reload --port 8000

  Stack: FastAPI, Uvicorn, NumPy, OpenCV
=============================================================================
"""

import os
import sys
import time
import base64
import io

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================================
#  Ensure src/ package is importable
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.bilinear_interpolation import bilinear_interpolation, nearest_neighbor
from src.metrics import calculate_error_metrics


# #########################################################################
#                     APP INITIALISATION
# #########################################################################

app = FastAPI(
    title="Bilinear Interpolation API",
    description=(
        "A microservice exposing manual Nearest-Neighbour and Bilinear "
        "Interpolation algorithms for image upscaling, plus numerical "
        "error analysis (MSE, PSNR, MAE)."
    ),
    version="1.0.0",
)

# ---- CORS: allow the Streamlit frontend (and any localhost dev) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:8502",
        "http://localhost:3000",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# #########################################################################
#                     HELPER FUNCTIONS
# #########################################################################

def _bytes_to_bgr(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR numpy array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image. "
                   "Ensure it is a valid PNG/JPG/BMP file.",
        )
    return img


def _bgr_to_base64_png(img: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG string."""
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _scale_image(image: np.ndarray, factor: int, algorithm: str):
    """
    Upscale *image* by *factor* using the specified *algorithm*.
    Returns (result_bgr, elapsed_seconds).

    Delegates to the manual loop-based implementations -- NOT cv2.resize.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    t0 = time.perf_counter()
    if algorithm == "nearest":
        result = nearest_neighbor(image, new_h, new_w, disable_tqdm=True)
    elif algorithm == "bilinear":
        result = bilinear_interpolation(image, new_h, new_w, disable_tqdm=True)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm '{algorithm}'. Use 'nearest' or 'bilinear'.",
        )
    elapsed = time.perf_counter() - t0

    return result, elapsed


# #########################################################################
#                     RESPONSE MODELS
# #########################################################################

class UpscaleResponse(BaseModel):
    """Response schema for /upscale."""
    image_b64: str
    elapsed: float
    width: int
    height: int
    algorithm: str
    scale_factor: int


class EvaluateResponse(BaseModel):
    """Response schema for /evaluate."""
    mse: float
    psnr: float
    mae: float


class HealthResponse(BaseModel):
    """Response schema for /health."""
    status: str


# #########################################################################
#                     ENDPOINTS
# #########################################################################

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale_image(
    image: UploadFile = File(..., description="Source image file (PNG/JPG)"),
    scale_factor: int = Form(4, description="Integer upscale factor (2-8)"),
    algorithm: str = Form("bilinear", description="'nearest' or 'bilinear'"),
):
    """
    Upscale an uploaded image using the specified algorithm.

    Returns the result as a base64-encoded PNG string alongside
    timing and dimension metadata.
    """
    # ---- Validate inputs ----
    if scale_factor < 1 or scale_factor > 16:
        raise HTTPException(status_code=400, detail="scale_factor must be 1-16.")

    # ---- Decode the uploaded file ----
    raw = await image.read()
    src_bgr = _bytes_to_bgr(raw)

    # ---- Run the interpolation ----
    result, elapsed = _scale_image(src_bgr, scale_factor, algorithm)

    # ---- Encode result as base64 PNG ----
    result_b64 = _bgr_to_base64_png(result)
    dst_h, dst_w = result.shape[:2]

    return UpscaleResponse(
        image_b64=result_b64,
        elapsed=round(elapsed, 4),
        width=dst_w,
        height=dst_h,
        algorithm=algorithm,
        scale_factor=scale_factor,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_images(
    ground_truth: UploadFile = File(..., description="Ground truth HR image"),
    upscaled: UploadFile = File(..., description="Upscaled image to evaluate"),
):
    """
    Calculate numerical error metrics (MSE, PSNR, MAE) between
    a ground truth image and an upscaled reconstruction.

    Both images must have the same dimensions.
    """
    gt_bytes = await ground_truth.read()
    up_bytes = await upscaled.read()

    gt_bgr = _bytes_to_bgr(gt_bytes)
    up_bgr = _bytes_to_bgr(up_bytes)

    if gt_bgr.shape != up_bgr.shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Shape mismatch: ground_truth={gt_bgr.shape}, "
                f"upscaled={up_bgr.shape}. Both must be identical."
            ),
        )

    metrics = calculate_error_metrics(gt_bgr, up_bgr)

    # Handle inf PSNR for JSON serialisation
    psnr_val = metrics["psnr"]
    if psnr_val == float('inf'):
        psnr_val = 999.0  # sentinel for "perfect match"

    return EvaluateResponse(
        mse=round(metrics["mse"], 4),
        psnr=round(psnr_val, 4),
        mae=round(metrics["mae"], 4),
    )
