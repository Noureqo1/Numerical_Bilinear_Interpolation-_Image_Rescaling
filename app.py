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
        result = nearest_neighbor(image, new_h, new_w)
    else:
        result = bilinear_interpolation(image, new_h, new_w)
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
