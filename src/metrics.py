"""
=============================================================================
  Numerical Error Metrics  (shared module)
  -----------------------------------------
  Generic, standalone image quality metrics used by both the FastAPI
  backend and the batch processing pipeline.

  Functions are stateless -- they accept raw NumPy arrays and return
  plain Python dicts, making them safe to call from any context.
=============================================================================
"""

import numpy as np


def calculate_error_metrics(ground_truth_array, upscaled_array):
    """
    Calculate standard numerical error metrics between two images.

    This is a strictly generic, standalone function -- no global
    variables, no hardcoded paths.  It accepts any two NumPy arrays
    of the same shape and returns a metrics dictionary.

    Parameters
    ----------
    ground_truth_array : np.ndarray (uint8)
        The reference high-resolution image.
    upscaled_array : np.ndarray (uint8)
        The reconstructed image to evaluate.

    Returns
    -------
    dict with keys:
        mse   : float  -- Mean Squared Error
        psnr  : float  -- Peak Signal-to-Noise Ratio (dB)
        mae   : float  -- Mean Absolute Error

    Edge Cases
    ----------
    * If MSE == 0 (identical images), PSNR is set to float('inf').
    """
    gt = ground_truth_array.astype(np.float64)
    up = upscaled_array.astype(np.float64)

    # ---- MSE: (1/MN) * sum( (I - K)^2 ) ----
    mse = float(np.mean((gt - up) ** 2))

    # ---- PSNR: 10 * log10( MAX_I^2 / MSE ) ----
    MAX_I = 255.0
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = float(10.0 * np.log10((MAX_I ** 2) / mse))

    # ---- MAE: (1/MN) * sum( |I - K| ) ----
    mae = float(np.mean(np.abs(gt - up)))

    return {"mse": mse, "psnr": psnr, "mae": mae}
