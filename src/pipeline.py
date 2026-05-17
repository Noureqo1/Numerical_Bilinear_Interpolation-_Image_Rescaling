"""
=============================================================================
  Image Rescaling -- Testing & Reporting Pipeline
  ------------------------------------------------
  A modular pipeline that:
    1. Loads a dataset of images (local folder or HuggingFace)
    2. Runs a LIVE Matplotlib animation for the first image
    3. Batch-processes the remaining images (headless, tqdm progress)
    4. Generates a formatted statistical report via pandas

  Architecture
  ~~~~~~~~~~~~
    load_dataset(source)           -> list of (name, BGR ndarray)
    scale_image(image, factor, m)  -> (result ndarray, elapsed_sec)
    live_demo_upscale(image, f)    -> animated side-by-side dashboard
    generate_report(results)       -> pandas table + summary stats

  The core interpolation algorithms are imported from
  src/bilinear_interpolation.py to avoid code duplication.

  Stack: Python 3, NumPy, OpenCV, Matplotlib, pandas, tqdm
=============================================================================
"""

import os
import sys
import time
import glob

import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================================
#  TkAgg backend for interactive animation on Windows
# =========================================================================
matplotlib.use("TkAgg")

# =========================================================================
#  Import core algorithms from the src package
# =========================================================================
# We add the project root to sys.path so that `src.*` resolves correctly
# regardless of the working directory used to invoke the script.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.bilinear_interpolation import bilinear_interpolation, nearest_neighbor


# #########################################################################
#                      1.  DATA INGESTION MODULE
# #########################################################################

def load_dataset(source="data/Set14/Set14"):
    """
    Load images from *source* and return a list of (name, BGR ndarray).

    Supported sources
    -----------------
    * **Local directory path** (default) -- scans for .png/.jpg/.bmp files.
      Examples:
          load_dataset("data/Set14/Set14")
          load_dataset("data/Set5")

    * **HuggingFace dataset string** (e.g. "eugenesiow/Set14") --
      requires the `datasets` and `Pillow` packages.  Falls back to
      local folder if the import fails.

    Returns
    -------
    list[tuple[str, np.ndarray]]
        Each element is (image_name, BGR_image_array).
    """
    # ---- Resolve relative paths against the project root ----
    abs_source = os.path.join(PROJECT_ROOT, source) if not os.path.isabs(source) else source

    # ------------------------------------------------------------------
    #  Path A:  Local directory  (fast, no network)
    # ------------------------------------------------------------------
    if os.path.isdir(abs_source):
        print(f"[LOAD] Scanning local folder: {abs_source}")
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(abs_source, ext)))
        paths.sort()

        if not paths:
            raise FileNotFoundError(
                f"No image files found in '{abs_source}'. "
                "Check the path or add images."
            )

        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                images.append((os.path.basename(p), img))

        print(f"[LOAD] Loaded {len(images)} images from disk.\n")
        return images

    # ------------------------------------------------------------------
    #  Path B:  HuggingFace `datasets` library  (needs network + pip pkg)
    # ------------------------------------------------------------------
    try:
        from datasets import load_dataset as hf_load
        print(f"[LOAD] Fetching HuggingFace dataset: {source} ...")
        ds = hf_load(source, split="validation")

        images = []
        for i, sample in enumerate(ds):
            # HuggingFace image datasets usually expose a PIL Image
            # under the key "hr" (high-res) or "image".
            pil_img = sample.get("hr") or sample.get("image")
            if pil_img is None:
                continue
            arr = np.array(pil_img)
            # PIL gives RGB; OpenCV needs BGR
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            name = sample.get("filename", f"image_{i:03d}.png")
            images.append((name, arr))

        print(f"[LOAD] Loaded {len(images)} images from HuggingFace.\n")
        return images

    except ImportError:
        raise ImportError(
            f"'{source}' is not a local directory and the `datasets` "
            "package is not installed.\n"
            "  pip install datasets   OR   provide a local folder path."
        )


# #########################################################################
#                    2.  CORE SCALING WRAPPER
# #########################################################################

def scale_image(image, factor, method="bilinear"):
    """
    Upscale *image* by *factor* using the specified *method*.

    Parameters
    ----------
    image  : np.ndarray  (BGR, uint8)
    factor : float       (e.g. 4.0 for 4x upscale)
    method : str         "bilinear" | "nearest"

    Returns
    -------
    (result, elapsed)
        result  : np.ndarray (BGR, uint8)
        elapsed : float      (seconds)

    The wrapper delegates to the manual, loop-based implementations
    imported from src/bilinear_interpolation.py -- NOT cv2.resize.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    t0 = time.perf_counter()

    if method == "nearest":
        result = nearest_neighbor(image, new_h, new_w)
    elif method == "bilinear":
        result = bilinear_interpolation(image, new_h, new_w)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bilinear' or 'nearest'.")

    elapsed = time.perf_counter() - t0
    return result, elapsed


# #########################################################################
#              3.  LIVE DEMO MODULE  (animated for image #1)
# #########################################################################

def live_demo_upscale(image, factor, name="demo", pause_sec=0.05):
    """
    Run a live, row-by-row animated Matplotlib dashboard for a single
    image upscale, showing three side-by-side panels:

        [Original]  |  [Nearest-Neighbour]  |  [Bilinear Interpolation]

    Animation mechanics
    -------------------
    * plt.ion() enables non-blocking interactive mode.
    * Destination arrays start as blank (black) grids.
    * After computing every pixel in row *u* for BOTH algorithms,
      we push the updated arrays to the canvas via im.set_data()
      and call plt.pause() to let the human eye track progression.
    * For large images we only refresh every N rows to keep it
      watchable (~60 visual updates max).

    Parameters
    ----------
    image     : np.ndarray (BGR uint8) -- source image.
    factor    : float                  -- upscale factor (e.g. 4.0).
    name      : str                    -- display name for the title.
    pause_sec : float                  -- delay between row refreshes.
    """
    src = image
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = int(src_h * factor), int(src_w * factor)

    S_x = dst_h / src_h
    S_y = dst_w / src_w

    print("=" * 65)
    print(f"  LIVE DEMO: {name}")
    print(f"  {src_w}x{src_h} -> {dst_w}x{dst_h}  ({factor}x upscale)")
    print("=" * 65)

    # ---- Blank destination canvases ----
    nn_dst  = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    bil_dst = np.zeros((dst_h, dst_w, 3), dtype=np.float64)

    def to_rgb(img):
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # ================================================================
    #  Set up the live figure
    # ================================================================
    plt.ion()
    fig, (ax_src, ax_nn, ax_bil) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Live Demo: {name}  ({factor}x upscale)",
                 fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    # -- Left: original (static) --
    ax_src.imshow(to_rgb(src), interpolation="none")
    ax_src.set_title(f"Original\n({src_w}x{src_h})", fontsize=11, color="white")
    ax_src.axis("off")

    # -- Middle: nearest-neighbour (live) --
    im_nn = ax_nn.imshow(to_rgb(nn_dst), interpolation="none",
                         vmin=0, vmax=255)
    ax_nn.set_title(f"Nearest-Neighbour\n(row 0/{dst_h})",
                    fontsize=11, color="white")
    ax_nn.axis("off")

    # -- Right: bilinear (live) --
    im_bil = ax_bil.imshow(to_rgb(bil_dst), interpolation="none",
                           vmin=0, vmax=255)
    ax_bil.set_title(f"Bilinear Interpolation\n(row 0/{dst_h})",
                     fontsize=11, color="white")
    ax_bil.axis("off")

    for ax in (ax_src, ax_nn, ax_bil):
        ax.set_facecolor("#0f0f23")

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.3)

    # ================================================================
    #  Adaptive refresh rate
    # ================================================================
    # For large images, refreshing every single row is too slow.
    # We cap at ~60 visual refreshes so the animation stays watchable.
    MAX_REFRESHES = 60
    refresh_every = max(1, dst_h // MAX_REFRESHES)

    # ================================================================
    #  Row-by-row computation + animation loop
    # ================================================================
    print(f"\n[ANIM] Painting {dst_h} rows live (refresh every {refresh_every})...")

    for u in tqdm(range(dst_h), desc="LiveDemo", unit="row",
                  bar_format="{l_bar}{bar:40}{r_bar}"):

        # ---- Inverse mapping: row direction ----
        x_i = u / S_x
        x1  = int(np.floor(x_i))
        x2  = min(x1 + 1, src_h - 1)
        x1  = min(x1, src_h - 1)
        dx  = x_i - int(np.floor(x_i))

        for v in range(dst_w):
            # ---- Inverse mapping: col direction ----
            y_i = v / S_y
            y1  = int(np.floor(y_i))
            y2  = min(y1 + 1, src_w - 1)
            y1  = min(y1, src_w - 1)
            dy  = y_i - int(np.floor(y_i))

            # -- Nearest-neighbour --
            x_near = min(int(round(x_i)), src_h - 1)
            y_near = min(int(round(y_i)), src_w - 1)
            nn_dst[u, v] = src[x_near, y_near]

            # -- Bilinear (area-weighted Lagrange) --
            f00 = src[x1, y1].astype(np.float64)
            f10 = src[x2, y1].astype(np.float64)
            f01 = src[x1, y2].astype(np.float64)
            f11 = src[x2, y2].astype(np.float64)

            bil_dst[u, v] = (
                (1 - dx) * (1 - dy) * f00 +
                     dx  * (1 - dy) * f10 +
                (1 - dx) *      dy  * f01 +
                     dx  *      dy  * f11
            )

        # ---- Refresh the canvas every N rows ----
        if (u + 1) % refresh_every == 0 or u == dst_h - 1:
            im_nn.set_data(to_rgb(nn_dst))
            im_bil.set_data(to_rgb(np.clip(bil_dst, 0, 255).astype(np.uint8)))

            ax_nn.set_title(
                f"Nearest-Neighbour\n(row {u+1}/{dst_h})",
                fontsize=11, color="white")
            ax_bil.set_title(
                f"Bilinear Interpolation\n(row {u+1}/{dst_h})",
                fontsize=11, color="white")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(pause_sec)

    # ---- Mark completion ----
    fig.suptitle(f"Demo Complete: {name}  ({factor}x)",
                 fontsize=14, fontweight="bold", color="lime")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    save_path = os.path.join(PROJECT_ROOT, "output", "live_demo.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[PLOT] Live demo saved -> {save_path}")

    # Brief pause then close -- pipeline continues headless
    plt.pause(1.5)
    plt.close(fig)
    plt.ioff()

    return nn_dst, np.clip(bil_dst, 0, 255).astype(np.uint8)


# #########################################################################
#              4.  BATCH PROCESSING MODULE  (images 2..N)
# #########################################################################

def batch_process(images, factor=4.0):
    """
    Upscale every image in *images* with both methods, recording metrics.

    Parameters
    ----------
    images : list[tuple[str, np.ndarray]]
        Each element is (name, BGR array).
    factor : float
        Upscale factor (default 4x).

    Returns
    -------
    list[dict]
        One dict per image with keys:
            image_name, orig_h, orig_w, dst_h, dst_w,
            time_nn, time_bilinear
    """
    results = []

    print("\n" + "=" * 65)
    print(f"  BATCH PROCESSING: {len(images)} images @ {factor}x upscale")
    print("=" * 65 + "\n")

    for name, img in tqdm(images, desc="Batch", unit="img",
                          bar_format="{l_bar}{bar:40}{r_bar}"):
        h, w = img.shape[:2]
        dst_h, dst_w = int(h * factor), int(w * factor)

        # ---- Nearest-neighbour (timed) ----
        _, t_nn = scale_image(img, factor, method="nearest")

        # ---- Bilinear (timed) ----
        _, t_bil = scale_image(img, factor, method="bilinear")

        results.append({
            "image_name":   name,
            "orig_h":       h,
            "orig_w":       w,
            "dst_h":        dst_h,
            "dst_w":        dst_w,
            "time_nn":      round(t_nn, 4),
            "time_bilinear": round(t_bil, 4),
        })

    return results


# #########################################################################
#              5.  REPORTING MODULE
# #########################################################################

def generate_report(results, save_csv=True):
    """
    Compile batch results into a pandas DataFrame, print a formatted
    ASCII/Markdown table, and display summary statistics.

    Parameters
    ----------
    results  : list[dict]  -- output of batch_process().
    save_csv : bool         -- if True, save to output/report.csv.
    """
    df = pd.DataFrame(results)

    # ---- Derived columns ----
    df["orig_resolution"] = df["orig_w"].astype(str) + "x" + df["orig_h"].astype(str)
    df["dst_resolution"]  = df["dst_w"].astype(str) + "x" + df["dst_h"].astype(str)
    df["speedup"]         = (df["time_bilinear"] / df["time_nn"]).round(2)

    # ---- Select display columns ----
    display = df[[
        "image_name", "orig_resolution", "dst_resolution",
        "time_nn", "time_bilinear", "speedup"
    ]].copy()
    display.columns = [
        "Image", "Original", "Target",
        "NN Time (s)", "Bilinear Time (s)", "Speedup (Bil/NN)"
    ]

    # ==================================================================
    #  Print the report
    # ==================================================================
    print("\n")
    print("=" * 80)
    print("  RESCALING BENCHMARK REPORT")
    print("=" * 80)

    # Markdown-style table
    print(display.to_markdown(index=False, floatfmt=".4f"))

    # ---- Summary statistics ----
    print("\n" + "-" * 80)
    print("  SUMMARY STATISTICS")
    print("-" * 80)
    print(f"  Total images processed   : {len(df)}")
    print(f"  Upscale factor           : {df['dst_h'].iloc[0] / df['orig_h'].iloc[0]:.0f}x")
    print(f"")
    print(f"  Nearest-Neighbour (avg)  : {df['time_nn'].mean():.4f} s")
    print(f"  Nearest-Neighbour (total): {df['time_nn'].sum():.4f} s")
    print(f"")
    print(f"  Bilinear (avg)           : {df['time_bilinear'].mean():.4f} s")
    print(f"  Bilinear (total)         : {df['time_bilinear'].sum():.4f} s")
    print(f"")
    print(f"  Avg speedup ratio        : {df['speedup'].mean():.2f}x  (Bilinear is slower)")
    print(f"  Max single-image bilinear: {df['time_bilinear'].max():.4f} s  "
          f"({df.loc[df['time_bilinear'].idxmax(), 'image_name']})")
    print("=" * 80)

    # ---- Save CSV ----
    if save_csv:
        csv_path = os.path.join(PROJECT_ROOT, "output", "report.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\n[CSV] Report saved -> {csv_path}")

    return df


# #########################################################################
#              6.  MAIN ENTRY POINT
# #########################################################################

def main():
    """
    Pipeline execution flow
    ~~~~~~~~~~~~~~~~~~~~~~~
    1. Load dataset from local folder (default: data/Set14/Set14).
       Override via CLI:  python pipeline.py data/Set5
    2. Run the live animated demo on image #1.
    3. Batch-process images #2..N with timing.
    4. Generate & print the statistical report.
    """
    # ---- Determine dataset source ----
    default_source = "data/Set5"
    source = sys.argv[1] if len(sys.argv) > 1 else default_source

    UPSCALE_FACTOR = 4.0

    # ==================================================================
    #  Step 1 -- Load dataset
    # ==================================================================
    images = load_dataset(source)
    if not images:
        print("[ERROR] No images loaded. Exiting.")
        return

    # ==================================================================
    #  Step 2 -- Live demo on image #1
    # ==================================================================
    demo_name, demo_img = images[0]
    print(f"\n[PIPELINE] Live demo image: {demo_name}")
    live_demo_upscale(demo_img, factor=UPSCALE_FACTOR,
                      name=demo_name, pause_sec=0.05)

    # ==================================================================
    #  Step 3 -- Batch process remaining images
    # ==================================================================
    remaining = images[1:]
    if remaining:
        results = batch_process(remaining, factor=UPSCALE_FACTOR)
    else:
        print("[INFO] Only one image in dataset -- skipping batch.")
        results = []

    # Include image #1 timing at the front of the report
    # (run it once more headless for fair timing)
    print(f"\n[PIPELINE] Timing image #1 ({demo_name}) for report...")
    _, t_nn  = scale_image(demo_img, UPSCALE_FACTOR, method="nearest")
    _, t_bil = scale_image(demo_img, UPSCALE_FACTOR, method="bilinear")
    h, w = demo_img.shape[:2]
    results.insert(0, {
        "image_name":    demo_name,
        "orig_h":        h,
        "orig_w":        w,
        "dst_h":         int(h * UPSCALE_FACTOR),
        "dst_w":         int(w * UPSCALE_FACTOR),
        "time_nn":       round(t_nn, 4),
        "time_bilinear": round(t_bil, 4),
    })

    # ==================================================================
    #  Step 4 -- Generate report
    # ==================================================================
    generate_report(results)

    print("\n[DONE] Pipeline finished successfully.")


# =========================================================================
if __name__ == "__main__":
    main()
