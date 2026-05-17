"""
=============================================================================
  Bilinear Interpolation for Image Rescaling  (with Live Animation)
  ------------------------------------------------------------------
  A pedagogical Python implementation demonstrating:
    1. Inverse affine mapping from destination -> source coordinates
    2. Area-weighted Lagrange (bilinear) interpolation
    3. Nearest-neighbor baseline for comparison
    4. Live row-by-row animation dashboard (plt.ion)
    5. tqdm terminal progress bars

  Mathematical Formulation
  ~~~~~~~~~~~~~~~~~~~~~~~~
  Given sub-pixel source coordinates (x_i, y_i):

      x1 = floor(x_i),  x2 = x1 + 1
      y1 = floor(y_i),  y2 = y1 + 1
      dx = x_i - x1     (horizontal fractional distance)
      dy = y_i - y1     (vertical fractional distance)

  The interpolated intensity is:

      f(x_i, y_i) = (1-dx)(1-dy) . f(x1,y1)
                   +  dx (1-dy) . f(x2,y1)
                   + (1-dx) dy  . f(x1,y2)
                   +  dx   dy  . f(x2,y2)

  Stack  : Python 3, NumPy, OpenCV (cv2), Matplotlib, tqdm
=============================================================================
"""

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Use TkAgg backend for interactive animation on Windows
# (only when run directly -- the pipeline sets its own backend)
if __name__ == "__main__":
    matplotlib.use("TkAgg")


# =========================================================================
#  1.  BILINEAR INTERPOLATION  (core algorithm - from scratch)
# =========================================================================

def bilinear_interpolation(src, dst_height, dst_width):
    """
    Rescale *src* to (dst_height x dst_width) using bilinear interpolation.

    For every pixel (u, v) in the destination image:
      1. Inverse affine mapping:  x_i = u / S_x ,  y_i = v / S_y
      2. Four integer neighbours with boundary clamping
      3. Fractional distances dx, dy
      4. Area-weighted Lagrange formula
    """
    src_h, src_w = src.shape[:2]
    is_color = (src.ndim == 3)

    S_x = dst_height / src_h
    S_y = dst_width  / src_w

    if is_color:
        dst = np.zeros((dst_height, dst_width, src.shape[2]), dtype=np.float64)
    else:
        dst = np.zeros((dst_height, dst_width), dtype=np.float64)

    print("\n[BILINEAR] Bilinear Interpolation -- processing rows...")
    for u in tqdm(range(dst_height), desc="Bilinear", unit="row",
                  bar_format="{l_bar}{bar:40}{r_bar}"):

        # Step 1 - Inverse mapping: destination row -> source row
        x_i = u / S_x

        # Step 2a - Integer neighbours (row) & clamping
        x1 = int(np.floor(x_i))
        x2 = min(x1 + 1, src_h - 1)
        x1 = min(x1, src_h - 1)

        # Step 3a - Vertical fractional distance
        dx = x_i - int(np.floor(x_i))

        for v in range(dst_width):
            # Step 1 - Inverse mapping: destination col -> source col
            y_i = v / S_y

            # Step 2b - Integer neighbours (col) & clamping
            y1 = int(np.floor(y_i))
            y2 = min(y1 + 1, src_w - 1)
            y1 = min(y1, src_w - 1)

            # Step 3b - Horizontal fractional distance
            dy = y_i - int(np.floor(y_i))

            # Step 4 - Bilinear (area-weighted Lagrange) formula
            #   f(x_i,y_i) = (1-dx)(1-dy).f(x1,y1) + dx(1-dy).f(x2,y1)
            #              + (1-dx)dy.f(x1,y2)      + dx.dy.f(x2,y2)
            f_x1y1 = src[x1, y1].astype(np.float64)
            f_x2y1 = src[x2, y1].astype(np.float64)
            f_x1y2 = src[x1, y2].astype(np.float64)
            f_x2y2 = src[x2, y2].astype(np.float64)

            dst[u, v] = (
                (1 - dx) * (1 - dy) * f_x1y1 +
                     dx  * (1 - dy) * f_x2y1 +
                (1 - dx) *      dy  * f_x1y2 +
                     dx  *      dy  * f_x2y2
            )

    return np.clip(dst, 0, 255).astype(np.uint8)


# =========================================================================
#  2.  NEAREST-NEIGHBOUR BASELINE
# =========================================================================

def nearest_neighbor(src, dst_height, dst_width):
    """
    Rescale using nearest-neighbour: round to closest source pixel.
    Produces characteristic "blocky" artefacts.
    """
    src_h, src_w = src.shape[:2]
    is_color = (src.ndim == 3)

    S_x = dst_height / src_h
    S_y = dst_width  / src_w

    if is_color:
        dst = np.zeros((dst_height, dst_width, src.shape[2]), dtype=np.uint8)
    else:
        dst = np.zeros((dst_height, dst_width), dtype=np.uint8)

    print("\n[NN] Nearest-Neighbour -- processing rows...")
    for u in tqdm(range(dst_height), desc="Nearest ", unit="row",
                  bar_format="{l_bar}{bar:40}{r_bar}"):
        x_near = min(int(round(u / S_x)), src_h - 1)
        for v in range(dst_width):
            y_near = min(int(round(v / S_y)), src_w - 1)
            dst[u, v] = src[x_near, y_near]

    return dst


# =========================================================================
#  3.  LIVE ANIMATED TEST CASE  (2x2 -> 8x8 with row-by-row painting)
# =========================================================================

def run_animated_test_case(pause_sec=0.35):
    """
    Animate the 2x2 -> 8x8 upscaling row-by-row in a live Matplotlib
    dashboard with three side-by-side panels.

    Animation Mechanics
    -------------------
    1. plt.ion() enables interactive (non-blocking) mode so the figure
       window stays responsive while we update it inside a loop.
    2. We pre-create three AxesImage objects via imshow() and then call
       im.set_data() each iteration -- this is far cheaper than
       redrawing the entire figure.
    3. After computing ALL pixels in row *u* for both algorithms, we
       push the updated arrays to the canvas and call plt.pause() to
       flush the draw event queue AND give the human eye time to follow.
    4. Grid lines are overlaid at integer ticks to emphasise the
       discrete 8x8 structure of the destination images.

    Input pixel layout (2x2, BGR):
        (0,0) Black   (0,0,0)       |  (0,1) White  (255,255,255)
        (1,0) Gray    (128,128,128)  |  (1,1) Navy   BGR(128,0,0)
    """
    print("=" * 65)
    print("  ANIMATED TEST CASE:  2x2 -> 8x8 Upscaling")
    print("=" * 65)

    # ---- Build the 2x2 source image (BGR) ----
    src = np.zeros((2, 2, 3), dtype=np.uint8)
    src[0, 0] = [0,   0,   0  ]    # Black
    src[0, 1] = [255, 255, 255]    # White
    src[1, 0] = [128, 128, 128]    # Gray
    src[1, 1] = [128, 0,   0  ]    # Navy  (BGR)

    src_h, src_w = 2, 2
    dst_h, dst_w = 8, 8

    S_x = dst_h / src_h    # = 4.0
    S_y = dst_w / src_w    # = 4.0

    # ---- Initialise blank destination arrays (all zeros = black) ----
    nn_dst  = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    bil_dst = np.zeros((dst_h, dst_w, 3), dtype=np.float64)

    # ---- Helper: BGR -> RGB for Matplotlib display ----
    def to_rgb(img):
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # ================================================================
    #  Set up the live Matplotlib figure with three subplots
    # ================================================================
    plt.ion()   # <-- Enable interactive mode (non-blocking draws)

    fig, (ax_src, ax_nn, ax_bil) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Live Animation: 2x2 -> 8x8 Upscaling",
                 fontsize=15, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    # --- Left panel: Original 2x2 (static, shown enlarged) ---
    src_display = cv2.resize(src, (dst_w, dst_h),
                             interpolation=cv2.INTER_NEAREST)
    ax_src.imshow(to_rgb(src_display), interpolation="none")
    ax_src.set_title("Original 2x2\n(enlarged for visibility)",
                     fontsize=11, color="white")
    ax_src.set_xticks(np.arange(-0.5, dst_w, 1), minor=True)
    ax_src.set_yticks(np.arange(-0.5, dst_h, 1), minor=True)
    ax_src.grid(which="minor", color="white", linewidth=0.5, alpha=0.4)
    ax_src.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)

    # --- Middle panel: Nearest-Neighbour (updates live) ---
    im_nn = ax_nn.imshow(to_rgb(nn_dst), interpolation="none",
                         vmin=0, vmax=255)
    ax_nn.set_title("Nearest-Neighbour\n(row 0/8)", fontsize=11,
                    color="white")
    ax_nn.set_xticks(np.arange(-0.5, dst_w, 1), minor=True)
    ax_nn.set_yticks(np.arange(-0.5, dst_h, 1), minor=True)
    ax_nn.grid(which="minor", color="yellow", linewidth=0.8, alpha=0.6)
    ax_nn.tick_params(which="both", bottom=False, left=False,
                      labelbottom=False, labelleft=False)

    # --- Right panel: Bilinear Interpolation (updates live) ---
    im_bil = ax_bil.imshow(to_rgb(bil_dst), interpolation="none",
                           vmin=0, vmax=255)
    ax_bil.set_title("Bilinear Interpolation\n(row 0/8)", fontsize=11,
                     color="white")
    ax_bil.set_xticks(np.arange(-0.5, dst_w, 1), minor=True)
    ax_bil.set_yticks(np.arange(-0.5, dst_h, 1), minor=True)
    ax_bil.grid(which="minor", color="cyan", linewidth=0.8, alpha=0.6)
    ax_bil.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)

    for ax in (ax_src, ax_nn, ax_bil):
        ax.set_facecolor("#0f0f23")

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.5)  # Initial pause so the user sees the blank canvas

    # ================================================================
    #  Row-by-row animation loop
    # ================================================================
    print("\n[ANIM] Painting rows live...")

    for u in range(dst_h):
        # ----- Inverse mapping (row direction) -----
        x_i   = u / S_x
        x1    = int(np.floor(x_i))
        x2    = min(x1 + 1, src_h - 1)
        x1    = min(x1, src_h - 1)
        dx    = x_i - int(np.floor(x_i))

        for v in range(dst_w):
            # ----- Inverse mapping (col direction) -----
            y_i = v / S_y
            y1  = int(np.floor(y_i))
            y2  = min(y1 + 1, src_w - 1)
            y1  = min(y1, src_w - 1)
            dy  = y_i - int(np.floor(y_i))

            # ---- Nearest-Neighbour: round to closest source pixel ----
            x_near = min(int(round(x_i)), src_h - 1)
            y_near = min(int(round(y_i)), src_w - 1)
            nn_dst[u, v] = src[x_near, y_near]

            # ---- Bilinear: area-weighted Lagrange formula ----
            f_x1y1 = src[x1, y1].astype(np.float64)
            f_x2y1 = src[x2, y1].astype(np.float64)
            f_x1y2 = src[x1, y2].astype(np.float64)
            f_x2y2 = src[x2, y2].astype(np.float64)

            bil_dst[u, v] = (
                (1 - dx) * (1 - dy) * f_x1y1 +
                     dx  * (1 - dy) * f_x2y1 +
                (1 - dx) *      dy  * f_x1y2 +
                     dx  *      dy  * f_x2y2
            )

        # ============================================================
        #  UPDATE CANVAS once per completed row
        # ============================================================
        # Convert current state of both destination arrays to RGB
        # and push into the pre-existing AxesImage objects.
        im_nn.set_data(to_rgb(nn_dst))
        im_bil.set_data(to_rgb(np.clip(bil_dst, 0, 255).astype(np.uint8)))

        # Update subplot titles with progress
        ax_nn.set_title(
            f"Nearest-Neighbour\n(row {u+1}/{dst_h})",
            fontsize=11, color="white")
        ax_bil.set_title(
            f"Bilinear Interpolation\n(row {u+1}/{dst_h})",
            fontsize=11, color="white")

        # Flush the draw queue and pause so the human eye can follow
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)

        print(f"  Row {u+1}/{dst_h} painted")

    # ---- Final result ----
    bil_final = np.clip(bil_dst, 0, 255).astype(np.uint8)

    # Update title to indicate completion
    fig.suptitle("Animation Complete: 2x2 -> 8x8 Upscaling",
                 fontsize=15, fontweight="bold", color="lime")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # Save the final frame
    fig.savefig("animated_test_case_final.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("[PLOT] Figure saved -> animated_test_case_final.png")

    # Print numerical matrices for verification
    print("\n-- Blue-channel matrix (Nearest-Neighbour 8x8) --")
    print(nn_dst[:, :, 0])
    print("\n-- Blue-channel matrix (Bilinear 8x8) --")
    print(bil_final[:, :, 0])

    # Keep the final figure open; turn off interactive mode
    plt.ioff()
    plt.show()

    return src, nn_dst, bil_final


# =========================================================================
#  4.  REAL IMAGE RESCALING
# =========================================================================

def rescale_image(image_path, scale=2.0):
    """Load a real image, upscale by *scale*, return (original, nn, bilinear)."""
    src = cv2.imread(image_path)
    if src is None:
        print(f"[WARN] Could not read '{image_path}'. Skipping.")
        return None, None, None

    h, w = src.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    print("=" * 65)
    print(f"  IMAGE RESCALING : {os.path.basename(image_path)}")
    print(f"  {w}x{h}  ->  {new_w}x{new_h}   (scale = {scale}x)")
    print("=" * 65)

    nn_result  = nearest_neighbor(src, new_h, new_w)
    bil_result = bilinear_interpolation(src, new_h, new_w)

    return src, nn_result, bil_result


# =========================================================================
#  5.  GENERATE A SAMPLE TEST IMAGE
# =========================================================================

def generate_sample_image(path="sample_input.png", size=64):
    """Create a small colour test image with gradients and shapes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for r in range(size):
        img[r, :, 2] = int(255 * r / (size - 1))
    for c in range(size):
        img[:, c, 1] = int(255 * c / (size - 1))
    for r in range(size):
        for c in range(size):
            if abs(r - c) < size // 8:
                img[r, c, 0] = 200

    s = size // 4
    img[s:3*s, s:3*s] = [255, 255, 255]
    cv2.circle(img, (size // 2, size // 2), size // 8, (30, 30, 30), -1)

    cv2.imwrite(path, img)
    print(f"[OK] Generated sample image -> {path}  ({size}x{size})")
    return path


# =========================================================================
#  6.  STATIC VISUALIZATION (for real-image comparison)
# =========================================================================

def visualize(original, nn_img, bil_img, title="Rescaling Comparison",
              save_path=None):
    """Three-panel static comparison figure (BGR -> RGB)."""
    def to_rgb(img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    panels = [
        (to_rgb(original),  "Original",               original.shape),
        (to_rgb(nn_img),    "Nearest-Neighbour",       nn_img.shape),
        (to_rgb(bil_img),   "Bilinear Interpolation",  bil_img.shape),
    ]

    for ax, (img, label, shape) in zip(axes, panels):
        ax.imshow(img, interpolation="none")
        h, w = shape[:2]
        ax.set_title(f"{label}\n({w}x{h})", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Figure saved -> {save_path}")
    plt.show()


# =========================================================================
#  7.  MAIN ENTRY POINT
# =========================================================================

def main():
    """
    Execution flow
    ~~~~~~~~~~~~~~
    A.  Run the ANIMATED 2x2 -> 8x8 test case (live row-by-row painting).
    B.  Optionally rescale a real image with tqdm progress + static plot.
    """

    # -- A. Animated test case (2x2 -> 8x8) ----------------------------
    src_tc, nn_tc, bil_tc = run_animated_test_case(pause_sec=0.35)

    # -- B. Real image rescaling ----------------------------------------
    sample_path = "sample_input.png"

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        sample_path = sys.argv[1]
        print(f"\n[INFO] Using user-supplied image: {sample_path}")
    else:
        sample_path = generate_sample_image(sample_path, size=64)

    scale_factor = 3.0
    src_img, nn_img, bil_img = rescale_image(sample_path, scale=scale_factor)

    if src_img is not None:
        visualize(src_img, nn_img, bil_img,
                  title=f"Image Rescaling: {scale_factor}x Upscale",
                  save_path="rescaling_comparison.png")

    print("\n[DONE] All results generated successfully.")


# =========================================================================
if __name__ == "__main__":
    main()
