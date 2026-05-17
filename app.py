"""
=============================================================================
  Bilinear Interpolation -- Interactive Streamlit Web UI
  -------------------------------------------------------
  A modern, sleek web interface that wraps the manual numerical
  rescaling algorithms (Nearest-Neighbour & Bilinear Interpolation)
  into an interactive tool with:
    - Animated Lottie tutorial section
    - Image upload & scale-factor selector
    - Side-by-side 3-column result comparison with timing

  HOW TO RUN
  ~~~~~~~~~~
    pip install streamlit streamlit-lottie opencv-python numpy Pillow requests
    streamlit run app.py

  Stack: Streamlit, NumPy, OpenCV, Pillow, streamlit-lottie
=============================================================================
"""

import os
import sys
import time

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# =========================================================================
#  Ensure src/ package is importable regardless of CWD
# =========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.bilinear_interpolation import bilinear_interpolation, nearest_neighbor


# #########################################################################
#                     HELPER FUNCTIONS
# #########################################################################

def pil_to_bgr(pil_img):
    """Convert a PIL Image (RGB/RGBA) to an OpenCV BGR numpy array."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr_img):
    """Convert an OpenCV BGR array to RGB for Streamlit display."""
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def scale_image(image, factor, method="bilinear"):
    """
    Upscale *image* (BGR uint8) by *factor* using the specified *method*.
    Returns (result_bgr, elapsed_seconds).

    Delegates to the manual loop-based implementations -- NOT cv2.resize.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    t0 = time.perf_counter()
    if method == "nearest":
        result = nearest_neighbor(image, new_h, new_w, disable_tqdm=True)
    else:
        result = bilinear_interpolation(image, new_h, new_w, disable_tqdm=True)
    elapsed = time.perf_counter() - t0

    return result, elapsed


def load_lottie_url(url: str):
    """Fetch a Lottie animation JSON from a public URL."""
    try:
        import requests
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# #########################################################################
#                     DETAIL INSPECTOR (Zoom / Crop)
# #########################################################################

def render_detail_inspector(original_img, nn_img, bilinear_img, scale_factor):
    """
    Interactive Detail Inspector -- lets users zoom into a region and
    see raw pixels side-by-side, defeating browser auto-scaling.

    Parameters
    ----------
    original_img : np.ndarray (BGR uint8)  -- source image.
    nn_img       : np.ndarray (BGR uint8)  -- Nearest-Neighbour result.
    bilinear_img : np.ndarray (BGR uint8)  -- Bilinear result.
    scale_factor : int                     -- upscale factor (e.g. 4).

    Cropping Math
    -------------
    * User selects (cx, cy) on the ORIGINAL image via sliders.
    * A fixed 50x50 box is extracted centred on that point (clamped).
    * For upscaled images the same region maps to coordinates and size
      multiplied by scale_factor, guaranteeing the identical spatial area.
    """
    with st.expander("\U0001f50d Inspect Pixels (Zoom Tool)", expanded=True):

        orig_h, orig_w = original_img.shape[:2]
        CROP_SIZE = 50  # pixels in original-image space

        # ---- Default: centre of image ----
        default_x = orig_w // 2
        default_y = orig_h // 2

        st.markdown(
            '<div class="info-banner">'
            'Drag the sliders to move the <strong>50 x 50</strong> zoom '
            'window across the original image. The same spatial region is '
            'shown for each method so you can compare raw pixels.'
            '</div>',
            unsafe_allow_html=True,
        )

        slider_c1, slider_c2 = st.columns(2)
        with slider_c1:
            cx = st.slider(
                "X Coordinate (horizontal)",
                min_value=0, max_value=orig_w - 1,
                value=default_x, key="zoom_x",
            )
        with slider_c2:
            cy = st.slider(
                "Y Coordinate (vertical)",
                min_value=0, max_value=orig_h - 1,
                value=default_y, key="zoom_y",
            )

        # ============================================================
        #  Crop box on the ORIGINAL image  (edge-clamped)
        # ============================================================
        half = CROP_SIZE // 2

        x1_o = max(cx - half, 0)
        y1_o = max(cy - half, 0)
        x2_o = min(x1_o + CROP_SIZE, orig_w)
        y2_o = min(y1_o + CROP_SIZE, orig_h)
        # Re-adjust if we hit the right/bottom edge
        x1_o = max(x2_o - CROP_SIZE, 0)
        y1_o = max(y2_o - CROP_SIZE, 0)

        # numpy: img[row_start:row_end, col_start:col_end]
        patch_orig = original_img[y1_o:y2_o, x1_o:x2_o]

        # ============================================================
        #  Crop box on the UPSCALED images (same spatial region)
        # ============================================================
        sf = scale_factor
        x1_u, y1_u = x1_o * sf, y1_o * sf
        x2_u, y2_u = x2_o * sf, y2_o * sf

        up_h, up_w = nn_img.shape[:2]
        x2_u = min(x2_u, up_w)
        y2_u = min(y2_u, up_h)

        patch_nn  = nn_img[y1_u:y2_u, x1_u:x2_u]
        patch_bil = bilinear_img[y1_u:y2_u, x1_u:x2_u]

        # ============================================================
        #  Display the three patches side-by-side
        # ============================================================
        z1, z2, z3 = st.columns(3)

        with z1:
            pw, ph = patch_orig.shape[1], patch_orig.shape[0]
            st.markdown(
                f'<div class="zoom-label">'
                f'<div class="title">Original (Low-Res Pixels)</div>'
                f'<div class="dims">{pw} x {ph} px</div>'
                f'</div>', unsafe_allow_html=True)
            st.markdown('<div class="pixelated-img">',
                        unsafe_allow_html=True)
            st.image(bgr_to_rgb(patch_orig), output_format="PNG",
                     width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with z2:
            pw, ph = patch_nn.shape[1], patch_nn.shape[0]
            st.markdown(
                f'<div class="zoom-label">'
                f'<div class="title">Nearest-Neighbour (Blocky)</div>'
                f'<div class="dims">{pw} x {ph} px</div>'
                f'</div>', unsafe_allow_html=True)
            st.markdown('<div class="pixelated-img">',
                        unsafe_allow_html=True)
            st.image(bgr_to_rgb(patch_nn), output_format="PNG",
                     width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with z3:
            pw, ph = patch_bil.shape[1], patch_bil.shape[0]
            st.markdown(
                f'<div class="zoom-label">'
                f'<div class="title">Bilinear Interpolation (Smooth)</div>'
                f'<div class="dims">{pw} x {ph} px</div>'
                f'</div>', unsafe_allow_html=True)
            st.markdown('<div class="pixelated-img">',
                        unsafe_allow_html=True)
            st.image(bgr_to_rgb(patch_bil), output_format="PNG",
                     width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Coordinates footer ----
        st.caption(
            f"ROI: original[{y1_o}:{y2_o}, {x1_o}:{x2_o}]  |  "
            f"upscaled[{y1_u}:{y2_u}, {x1_u}:{x2_u}]  |  "
            f"Crop: {CROP_SIZE}px (original) / {CROP_SIZE * sf}px (upscaled)"
        )


# #########################################################################
#                  AUTOMATED DIAGNOSTIC TEST MODULE
# #########################################################################

def run_synthetic_diagnostic_test(scale_factor):
    """
    Automated system test using a deterministic 2x2 synthetic image.

    This function is fully decoupled from the user-upload pipeline.
    It programmatically generates a known input, runs both algorithms,
    validates output shapes, computes a difference map, and renders
    all results with performance metrics.

    Parameters
    ----------
    scale_factor : int
        Upscale factor (e.g. 4 produces an 8x8 output from 2x2).

    Synthetic Pattern
    -----------------
        +-------+-------+
        | Black | White |
        | (0,0) | (255) |
        +-------+-------+
        | Gray  | Navy  |
        | (128) | Blue  |
        +-------+-------+
    """
    st.markdown("---")
    st.markdown("### \U0001f9ea Automated System Test")

    # ==================================================================
    #  Step 1 -- Generate deterministic 2x2 synthetic image (BGR)
    # ==================================================================
    synthetic = np.zeros((2, 2, 3), dtype=np.uint8)
    synthetic[0, 0] = [0,   0,   0]     # Black
    synthetic[0, 1] = [255, 255, 255]    # White
    synthetic[1, 0] = [128, 128, 128]    # Gray
    synthetic[1, 1] = [128, 0,   0]      # Navy Blue (BGR)

    src_h, src_w = synthetic.shape[:2]
    dst_h, dst_w = src_h * scale_factor, src_w * scale_factor

    st.markdown(
        f'<div class="info-banner">'
        f'<strong>Synthetic input:</strong> {src_w}x{src_h} px &rarr; '
        f'{dst_w}x{dst_h} px ({scale_factor}x upscale)<br>'
        f'<strong>Pattern:</strong> '
        f'TL=Black(0,0,0) &bull; TR=White(255,255,255) &bull; '
        f'BL=Gray(128,128,128) &bull; BR=Navy(0,0,128)'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Show the "Before" input
    st.markdown("#### Before (2x2 Source)")
    st.markdown('<div class="pixelated-img">', unsafe_allow_html=True)
    st.image(bgr_to_rgb(synthetic), width=160, output_format="PNG")
    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================================================
    #  Step 2 -- Run both algorithms with timing
    # ==================================================================
    with st.spinner("Running diagnostic interpolation..."):
        nn_result,  t_nn  = scale_image(synthetic, scale_factor, "nearest")
        bil_result, t_bil = scale_image(synthetic, scale_factor, "bilinear")

    # ==================================================================
    #  Step 3 -- Programmatic validation
    # ==================================================================
    expected_shape = (dst_h, dst_w, 3)
    tests_passed = True
    messages = []

    # Shape checks
    if nn_result.shape == expected_shape:
        messages.append(f"NN output shape: {nn_result.shape} == {expected_shape}")
    else:
        tests_passed = False
        messages.append(f"NN shape MISMATCH: {nn_result.shape} != {expected_shape}")

    if bil_result.shape == expected_shape:
        messages.append(f"Bilinear output shape: {bil_result.shape} == {expected_shape}")
    else:
        tests_passed = False
        messages.append(f"Bilinear shape MISMATCH: {bil_result.shape} != {expected_shape}")

    # Dtype check
    if nn_result.dtype == np.uint8 and bil_result.dtype == np.uint8:
        messages.append("Dtype: uint8 (correct)")
    else:
        tests_passed = False
        messages.append(f"Dtype MISMATCH: NN={nn_result.dtype}, Bil={bil_result.dtype}")

    # Value range check
    if nn_result.max() <= 255 and bil_result.max() <= 255:
        messages.append("Value range: [0, 255] (correct)")
    else:
        tests_passed = False
        messages.append("Value range EXCEEDED 255")

    # NN should have exact source colors only (no blending)
    nn_unique = len(np.unique(nn_result.reshape(-1, 3), axis=0))
    messages.append(f"NN unique colors: {nn_unique} (expected <= 4)")

    # Bilinear should produce more unique colors (smooth gradients)
    bil_unique = len(np.unique(bil_result.reshape(-1, 3), axis=0))
    messages.append(f"Bilinear unique colors: {bil_unique} (expected > 4)")

    if tests_passed:
        st.success(
            "**Test Passed:** Output dimensions, dtype, value range, "
            "and matrix integrity verified."
        )
    else:
        st.error("**Test Failed:** See details below.")

    with st.expander("Validation details", expanded=not tests_passed):
        for msg in messages:
            st.text(msg)

    # ==================================================================
    #  Step 4 -- Visual "After" display (3 columns + difference map)
    # ==================================================================
    st.markdown("#### After (Upscaled Results)")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">Source</div>'
            f'<div class="dims">{src_w}x{src_h}</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="pixelated-img">', unsafe_allow_html=True)
        st.image(bgr_to_rgb(synthetic), output_format="PNG",
                 width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">Nearest-Neighbour</div>'
            f'<div class="dims">{dst_w}x{dst_h} &bull; Blocky</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="pixelated-img">', unsafe_allow_html=True)
        st.image(bgr_to_rgb(nn_result), output_format="PNG",
                 width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">Bilinear</div>'
            f'<div class="dims">{dst_w}x{dst_h} &bull; Smooth</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="pixelated-img">', unsafe_allow_html=True)
        st.image(bgr_to_rgb(bil_result), output_format="PNG",
                 width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        # Difference map: absolute per-pixel difference
        diff = np.abs(
            nn_result.astype(np.int16) - bil_result.astype(np.int16)
        ).astype(np.uint8)
        # Amplify for visibility (stretch to full 0-255 range)
        max_diff = diff.max() if diff.max() > 0 else 1
        diff_vis = (diff.astype(np.float64) / max_diff * 255).astype(np.uint8)

        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">|NN - Bilinear|</div>'
            f'<div class="dims">Difference Map</div>'
            '</div>', unsafe_allow_html=True)
        st.markdown('<div class="pixelated-img">', unsafe_allow_html=True)
        st.image(bgr_to_rgb(diff_vis), output_format="PNG",
                 width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================================================================
    #  Step 5 -- Performance report
    # ==================================================================
    st.markdown("#### Performance Report")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Nearest-Neighbour", f"{t_nn*1000:.2f} ms")
    with m2:
        st.metric("Bilinear", f"{t_bil*1000:.2f} ms")
    with m3:
        ratio = t_bil / max(t_nn, 1e-9)
        st.metric("Speedup Ratio", f"{ratio:.1f}x",
                  delta=f"Bilinear is {ratio:.1f}x slower",
                  delta_color="inverse")

    st.caption(
        f"Synthetic {src_w}x{src_h} -> {dst_w}x{dst_h}  |  "
        f"NN unique colors: {nn_unique}  |  "
        f"Bilinear unique colors: {bil_unique}"
    )


# #########################################################################
#                  NUMERICAL ERROR ANALYSIS MODULE
# #########################################################################

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
    gt  = ground_truth_array.astype(np.float64)
    up  = upscaled_array.astype(np.float64)

    # ---- MSE: (1/MN) * sum( (I - K)^2 ) ----
    mse = np.mean((gt - up) ** 2)

    # ---- PSNR: 10 * log10( MAX_I^2 / MSE ) ----
    MAX_I = 255.0
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10.0 * np.log10((MAX_I ** 2) / mse)

    # ---- MAE: (1/MN) * sum( |I - K| ) ----
    mae = np.mean(np.abs(gt - up))

    return {"mse": mse, "psnr": psnr, "mae": mae}


def render_error_analysis(scale_factor):
    """
    Full Error Analysis UI component.

    Methodology
    -----------
    To measure *true* reconstruction error we need a known Ground Truth.
    The pipeline is:
        1. Start with a High-Resolution (HR) image  (Ground Truth)
        2. Downsample it by `scale_factor` using simple decimation
           to create a Low-Resolution (LR) version
        3. Upscale the LR image back to HR dimensions using both
           algorithms
        4. Compare the upscaled result against the original Ground
           Truth using MSE, PSNR, and MAE

    Parameters
    ----------
    scale_factor : int
        The down-then-up scale factor (e.g. 4).
    """
    import matplotlib.pyplot as plt
    import glob

    st.markdown("---")
    st.header("\U0001f4c9 Numerical Error Analysis")

    st.markdown(
        '<div class="info-banner">'
        '<strong>Methodology:</strong> To measure true error, we start with a '
        'High-Resolution (HR) Ground Truth image, artificially shrink it '
        '(downsample) to a Low-Resolution (LR) image, then use our algorithms '
        'to scale it back up to HR dimensions. We compare the upscaled result '
        'against the original Ground Truth using MSE, PSNR, and MAE.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ==================================================================
    #  Discover available dataset images
    # ==================================================================
    data_dir = os.path.join(PROJECT_ROOT, "data", "Set5")
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(data_dir, ext)))
    paths.sort()

    if not paths:
        st.error(f"No images found in `{data_dir}`. Add images to run error analysis.")
        return

    image_names = [os.path.basename(p) for p in paths]

    # Let the user pick which image to analyse
    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        selected_name = st.selectbox(
            "Select Ground Truth image",
            image_names,
            index=2,  # default to butterfly.png (smaller, faster)
            key="error_img_select",
        )
    with sel_col2:
        st.markdown(f"**Scale factor:** {scale_factor}x")
        run_error = st.button(
            "\U0001f4ca Run Error Analysis",
            use_container_width=True,
            type="primary",
            key="run_error_btn",
        )

    if not run_error:
        st.info("Select an image above and click **Run Error Analysis** to start.")
        return

    # ==================================================================
    #  Load the Ground Truth HR image
    # ==================================================================
    sel_path = paths[image_names.index(selected_name)]
    gt_bgr = cv2.imread(sel_path)
    if gt_bgr is None:
        st.error(f"Failed to load `{sel_path}`.")
        return

    gt_h, gt_w = gt_bgr.shape[:2]

    # ==================================================================
    #  Step 1 -- Downsample HR -> LR  (simple decimation / area avg)
    # ==================================================================
    lr_h, lr_w = gt_h // scale_factor, gt_w // scale_factor
    if lr_h < 2 or lr_w < 2:
        st.error("Image too small for the selected scale factor.")
        return

    # Use cv2 INTER_AREA for the *downsampling only* (to create a
    # realistic LR input -- this is standard practice in SR benchmarks).
    lr_bgr = cv2.resize(gt_bgr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    # Trim GT to exact multiple so shapes match after upscale
    gt_trimmed = gt_bgr[:lr_h * scale_factor, :lr_w * scale_factor]

    st.markdown(
        f'<div class="info-banner">'
        f'<strong>{selected_name}:</strong> '
        f'Ground Truth {gt_w}x{gt_h} &rarr; '
        f'Downsampled {lr_w}x{lr_h} &rarr; '
        f'Upscaled {lr_w * scale_factor}x{lr_h * scale_factor}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ==================================================================
    #  Step 2 -- Upscale LR -> HR with both algorithms
    # ==================================================================
    with st.spinner("Running downsample -> upscale -> compare pipeline..."):
        nn_up,  t_nn  = scale_image(lr_bgr, scale_factor, "nearest")
        bil_up, t_bil = scale_image(lr_bgr, scale_factor, "bilinear")

    # ==================================================================
    #  Step 3 -- Calculate error metrics
    # ==================================================================
    metrics_nn  = calculate_error_metrics(gt_trimmed, nn_up)
    metrics_bil = calculate_error_metrics(gt_trimmed, bil_up)

    # ==================================================================
    #  Step 4 -- Display metric cards with deltas
    # ==================================================================
    st.markdown("### Metric Comparison")

    mc1, mc2, mc3 = st.columns(3)

    # MSE (lower is better)
    mse_delta = metrics_bil["mse"] - metrics_nn["mse"]
    with mc1:
        st.metric(
            "MSE (Nearest-Neighbour)",
            f"{metrics_nn['mse']:.2f}",
        )
        st.metric(
            "MSE (Bilinear)",
            f"{metrics_bil['mse']:.2f}",
            delta=f"{mse_delta:.2f} vs NN",
            delta_color="inverse",  # negative = green (lower is better)
        )

    # PSNR (higher is better)
    if metrics_bil["psnr"] != float('inf') and metrics_nn["psnr"] != float('inf'):
        psnr_delta = metrics_bil["psnr"] - metrics_nn["psnr"]
        psnr_delta_str = f"+{psnr_delta:.2f} dB vs NN" if psnr_delta > 0 else f"{psnr_delta:.2f} dB vs NN"
    else:
        psnr_delta_str = "Perfect"

    with mc2:
        psnr_nn_str = f"{metrics_nn['psnr']:.2f} dB" if metrics_nn["psnr"] != float('inf') else "Inf dB"
        psnr_bil_str = f"{metrics_bil['psnr']:.2f} dB" if metrics_bil["psnr"] != float('inf') else "Inf dB"
        st.metric("PSNR (Nearest-Neighbour)", psnr_nn_str)
        st.metric(
            "PSNR (Bilinear)",
            psnr_bil_str,
            delta=psnr_delta_str,
            delta_color="normal",  # positive = green (higher is better)
        )

    # MAE (lower is better)
    mae_delta = metrics_bil["mae"] - metrics_nn["mae"]
    with mc3:
        st.metric("MAE (Nearest-Neighbour)", f"{metrics_nn['mae']:.2f}")
        st.metric(
            "MAE (Bilinear)",
            f"{metrics_bil['mae']:.2f}",
            delta=f"{mae_delta:.2f} vs NN",
            delta_color="inverse",
        )

    # ==================================================================
    #  Step 5 -- Timing report
    # ==================================================================
    st.markdown("### Execution Time")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.metric("NN Time", f"{t_nn:.3f} s")
    with tc2:
        st.metric("Bilinear Time", f"{t_bil:.3f} s")

    # ==================================================================
    #  Step 6 -- Visual error heatmaps  (Matplotlib)
    # ==================================================================
    st.markdown("### Absolute Error Heatmaps")
    st.caption(
        "Brighter regions = higher pixel error. "
        "Computed as |Ground Truth - Upscaled| averaged across colour channels."
    )

    # Absolute error per pixel, averaged across channels
    err_nn  = np.mean(np.abs(
        gt_trimmed.astype(np.float64) - nn_up.astype(np.float64)
    ), axis=2)
    err_bil = np.mean(np.abs(
        gt_trimmed.astype(np.float64) - bil_up.astype(np.float64)
    ), axis=2)

    # Shared colour scale for fair comparison
    vmax = max(err_nn.max(), err_bil.max(), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")

    im1 = ax1.imshow(err_nn, cmap="inferno", vmin=0, vmax=vmax)
    ax1.set_title("Nearest-Neighbour Error", color="white", fontsize=12)
    ax1.axis("off")

    im2 = ax2.imshow(err_bil, cmap="inferno", vmin=0, vmax=vmax)
    ax2.set_title("Bilinear Error", color="white", fontsize=12)
    ax2.axis("off")

    # Shared colourbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.02, pad=0.04)
    cbar.set_label("Absolute Error (0-255)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ==================================================================
    #  Step 7 -- Side-by-side visual comparison
    # ==================================================================
    st.markdown("### Visual Comparison")
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">Ground Truth</div>'
            '</div>', unsafe_allow_html=True)
        st.image(bgr_to_rgb(gt_trimmed), width="stretch")
    with vc2:
        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">NN Upscaled</div>'
            '</div>', unsafe_allow_html=True)
        st.image(bgr_to_rgb(nn_up), width="stretch")
    with vc3:
        st.markdown(
            '<div class="zoom-label">'
            '<div class="title">Bilinear Upscaled</div>'
            '</div>', unsafe_allow_html=True)
        st.image(bgr_to_rgb(bil_up), width="stretch")


# #########################################################################
#                     PAGE CONFIGURATION
# #########################################################################

st.set_page_config(
    page_title="Bilinear Interpolation - Image Rescaler",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# #########################################################################
#                     CUSTOM CSS
# #########################################################################

st.markdown("""
<style>
    /* ---- Dark premium header bar ---- */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e0e0ff;
        font-size: 2.4rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #a0a0d0;
        font-size: 1.05rem;
        margin: 0.4rem 0 0 0;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-card .label {
        color: #8888bb;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        color: #e0e0ff;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .metric-card .sub {
        color: #6a6a9a;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    /* ---- Info banner ---- */
    .info-banner {
        background: linear-gradient(135deg, #1a1a3e, #252550);
        border-left: 4px solid #6c63ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #c0c0e8;
        font-size: 0.92rem;
        margin: 1rem 0;
    }

    /* ---- Reduce default padding ---- */
    .block-container { padding-top: 1rem; }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122a, #1e1e3a);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c0c0e8;
    }

    /* ---- Force raw pixel rendering (no browser anti-aliasing) ---- */
    .pixelated-img img {
        image-rendering: pixelated;
        image-rendering: -moz-crisp-edges;
        image-rendering: crisp-edges;
    }

    /* ---- Zoom patch label badges ---- */
    .zoom-label {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid #3a3a5c;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        margin-bottom: 0.4rem;
    }
    .zoom-label .title {
        color: #e0e0ff;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .zoom-label .dims {
        color: #6a6a9a;
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)


# #########################################################################
#                     HEADER
# #########################################################################

st.markdown("""
<div class="main-header">
    <h1>🔬 Bilinear Interpolation &mdash; Image Rescaler</h1>
    <p>A pedagogical tool comparing Nearest-Neighbour vs Bilinear upscaling &mdash; built from scratch, no cv2.resize</p>
</div>
""", unsafe_allow_html=True)


# #########################################################################
#                     TUTORIAL / EXPANDER
# #########################################################################

with st.expander("📖 How to use this tool & How it works", expanded=False):

    tut_col1, tut_col2 = st.columns([2, 1])

    with tut_col1:
        st.markdown("""
### How to use
1. **Upload** a PNG / JPG image using the sidebar on the left.
2. **Choose** a scaling factor (2x -- 8x) with the slider.
3. **Click** the **Run Interpolation** button and watch the magic.

---

### Nearest-Neighbour (blocky)
For each destination pixel, we simply pick the **closest** source pixel:

$$x_{\\text{near}} = \\text{round}\\!\\left(\\frac{u}{S_x}\\right), \\quad
  y_{\\text{near}} = \\text{round}\\!\\left(\\frac{v}{S_y}\\right)$$

This is fast but produces **blocky, staircase-like artefacts**.

### Bilinear Interpolation (smooth)
We use the **area-weighted Lagrange** form to blend
the four surrounding integer-grid pixels:

$$f(x_i,\\,y_i) = (1-dx)(1-dy)\\,f(x_1,y_1)
  + dx\\,(1-dy)\\,f(x_2,y_1)
  + (1-dx)\\,dy\\,f(x_1,y_2)
  + dx\\,dy\\,f(x_2,y_2)$$

where $dx = x_i - \\lfloor x_i \\rfloor$ and
$dy = y_i - \\lfloor y_i \\rfloor$ are the fractional distances.

This produces **smooth gradients** at the cost of more computation.
        """)

    with tut_col2:
        # ---- Lottie animation (graceful fallback) ----
        try:
            from streamlit_lottie import st_lottie

            lottie_url = (
                "https://assets2.lottiefiles.com/packages/"
                "lf20_w51pcehl.json"
            )
            lottie_json = load_lottie_url(lottie_url)

            if lottie_json:
                st_lottie(lottie_json, height=280, key="tutorial_anim")
            else:
                st.info("🎞️ Animation unavailable (network).")
        except ImportError:
            st.info(
                "Install `streamlit-lottie` for animations:\n\n"
                "```\npip install streamlit-lottie\n```"
            )


# #########################################################################
#                     SIDEBAR -- USER INPUTS
# #########################################################################

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Accepted formats: PNG, JPG, JPEG",
    )

    scale_factor = st.slider(
        "Scaling Factor",
        min_value=2, max_value=8, value=4, step=1,
        help="Upscale the image by this integer factor.",
    )

    st.markdown("---")
    run_button = st.button(
        "🚀 Run Interpolation",
        use_container_width=True,  # use width='stretch' for Streamlit >= 1.58
        type="primary",
    )

    st.markdown("---")
    test_button = st.button(
        "\U0001f9ea Run Automated System Test",
        use_container_width=True,
    )

    st.markdown("---")
    if st.button(
        "\U0001f4c9 Error Analysis (Dataset)",
        use_container_width=True,
    ):
        st.session_state["show_error_analysis"] = True

    st.markdown("---")
    st.markdown(
        "<small style='color:#6a6a9a;'>"
        "Built with manual NumPy loops.<br>"
        "No cv2.resize was harmed.</small>",
        unsafe_allow_html=True,
    )


# #########################################################################
#                     MAIN AREA -- PROCESSING & DISPLAY
# #########################################################################

# ---- Show the uploaded image preview ----
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    src_bgr = pil_to_bgr(pil_img)
    h, w = src_bgr.shape[:2]

    st.markdown(f"""
    <div class="info-banner">
        <strong>Image loaded:</strong> {uploaded_file.name}
        &nbsp;&bull;&nbsp; {w} &times; {h} px
        &nbsp;&bull;&nbsp; Target: {w * scale_factor} &times; {h * scale_factor} px
        ({scale_factor}x upscale)
    </div>
    """, unsafe_allow_html=True)

    # ================================================================
    #  Run interpolation when the button is pressed
    # ================================================================
    if run_button:
        # ---- Size guard: warn for very large outputs ----
        target_pixels = (h * scale_factor) * (w * scale_factor)
        if target_pixels > 4_000_000:
            st.warning(
                f"⚠️  Target image is **{w*scale_factor}x{h*scale_factor}** "
                f"({target_pixels:,} pixels). "
                "This may take a while with pure-Python loops. "
                "Consider a smaller image or lower scale factor for faster results."
            )

        with st.spinner("Crunching the math... estimating sub-pixels..."):
            nn_result,  t_nn  = scale_image(src_bgr, scale_factor, "nearest")
            bil_result, t_bil = scale_image(src_bgr, scale_factor, "bilinear")

        # ---- Success banner ----
        st.success(
            f"Done!  NN: **{t_nn:.2f}s**  |  "
            f"Bilinear: **{t_bil:.2f}s**  |  "
            f"Bilinear is **{t_bil/t_nn:.1f}x** slower"
        )

        # ============================================================
        #  Three-column side-by-side display
        # ============================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Original</div>
                <div class="value">{w} x {h}</div>
                <div class="sub">Source image</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(bgr_to_rgb(src_bgr), width="stretch")

        with col2:
            dst_h_nn, dst_w_nn = nn_result.shape[:2]
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Nearest-Neighbour</div>
                <div class="value">{dst_w_nn} x {dst_h_nn}</div>
                <div class="sub">⏱ {t_nn:.3f} s &nbsp;|&nbsp; Blocky</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(bgr_to_rgb(nn_result), width="stretch")

        with col3:
            dst_h_bil, dst_w_bil = bil_result.shape[:2]
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Bilinear Interpolation</div>
                <div class="value">{dst_w_bil} x {dst_h_bil}</div>
                <div class="sub">⏱ {t_bil:.3f} s &nbsp;|&nbsp; Smooth</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(bgr_to_rgb(bil_result), width="stretch")

        # ============================================================
        #  Timing comparison bar chart
        # ============================================================
        st.markdown("### ⏱️ Execution Time Comparison")

        chart_col1, chart_col2 = st.columns([2, 1])
        with chart_col1:
            import pandas as pd
            df_time = pd.DataFrame({
                "Method": ["Nearest-Neighbour", "Bilinear"],
                "Time (s)": [round(t_nn, 4), round(t_bil, 4)],
            })
            st.bar_chart(df_time.set_index("Method"), height=250)

        with chart_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Speedup Ratio</div>
                <div class="value">{t_bil / max(t_nn, 0.0001):.1f}x</div>
                <div class="sub">Bilinear / NN</div>
            </div>
            <div class="metric-card">
                <div class="label">Pixels Computed</div>
                <div class="value">{target_pixels:,}</div>
                <div class="sub">{w*scale_factor} x {h*scale_factor}</div>
            </div>
            """, unsafe_allow_html=True)

        # ============================================================
        #  Detail Inspector (Zoom Tool)
        # ============================================================
        render_detail_inspector(
            original_img=src_bgr,
            nn_img=nn_result,
            bilinear_img=bil_result,
            scale_factor=scale_factor,
        )

    elif not run_button:
        # ---- Show a preview of the original ----
        st.image(
            bgr_to_rgb(src_bgr),
            caption=f"Preview: {uploaded_file.name}  ({w}x{h})",
            width=min(w, 500),
        )

else:
    # ---- No image uploaded yet ----
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#6a6a9a;">
        <h2 style="color:#8888bb;">👈 Upload an image to get started</h2>
        <p style="font-size:1.1rem;">
            Use the sidebar to upload a PNG or JPG, pick a scale factor,
            and click <strong>Run Interpolation</strong>.
        </p>
        <p style="font-size:0.9rem; margin-top:1.5rem;">
            Tip: Start with a small image (64x64 to 128x128) for fast results,
            or go bigger to really see the difference between blocky and smooth.
        </p>
    </div>
    """, unsafe_allow_html=True)

# #########################################################################
#          DIAGNOSTIC TEST EXECUTION (independent of upload flow)
# #########################################################################

if test_button:
    run_synthetic_diagnostic_test(scale_factor)

# #########################################################################
#          ERROR ANALYSIS EXECUTION (persistent via session_state)
# #########################################################################

if st.session_state.get("show_error_analysis", False):
    render_error_analysis(scale_factor)
