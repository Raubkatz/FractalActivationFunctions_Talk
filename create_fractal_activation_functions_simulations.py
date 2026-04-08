import os
import math
from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio


# ============================================================
# IMPORT YOUR FRACTAL ACTIVATION FUNCTION COLLECTION
# ============================================================
# Assumes you have a Python file named:
# fractal_activation_functions.py
#
# with your existing functions inside.
# If your filename is different, change the import below.
# ============================================================
from fractal_activation_functions import (
    modulated_blancmange_curve,
    decaying_cosine_function_tf,
    modified_weierstrass_function_tanh,
    modified_weierstrass_function_relu,
    weierstrass_mandelbrot_function_xsinsquared,
    weierstrass_mandelbrot_function_xpsin,
    weierstrass_mandelbrot_function_relupsin,
    weierstrass_mandelbrot_function_tanhpsin,
    weierstrass_function_tf,
)


# ============================================================
# BASIC / CLASSICAL FRACTAL FUNCTIONS NOT YET IN COLLECTION
# ============================================================
def basic_blancmange_function_tf(x, num_terms=12):
    """
    Classical Blancmange / Takagi function:
        T(x) = sum_{n=0}^{N-1} 2^{-n} * phi(2^n x)
    where phi(u) = distance(u, nearest integer)
                 = abs(u - round(u))
    """
    x = tf.cast(x, tf.float64)
    y = tf.zeros_like(x, dtype=tf.float64)

    for n in range(num_terms):
        scale = 2.0 ** n
        u = scale * x
        phi = tf.abs(u - tf.round(u))
        y += phi / scale

    return tf.cast(y, tf.float32)


def basic_weierstrass_function_tf(x, a=0.5, b=7, num_terms=20):
    """
    Classical Weierstrass-type cosine series:
        W(x) = sum_{n=0}^{N-1} a^n cos(b^n pi x)
    Common classical constraints are 0 < a < 1 and odd integer b. :contentReference[oaicite:1]{index=1}
    """
    x = tf.cast(x, tf.float64)
    y = tf.zeros_like(x, dtype=tf.float64)

    for n in range(num_terms):
        y += (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    return tf.cast(y, tf.float32)


# ============================================================
# COLOR SETTINGS
# ============================================================
BG_COLOR = "#FFFFFF"      # background
LINE_COLOR = "#1565C0"    # curve color
ACCENT_COLOR = "#000000"  # text / frame / optional accent


# ============================================================
# ANIMATION SETTINGS
# ============================================================
ITERATIONS = 16                # depth / number of partial approximations
FRAMES_PER_SEGMENT = 8         # smoothness inside each depth step
FINAL_HOLD_FRAMES = 12         # how long final frame stays
LINE_WIDTH = 2.0
FIGSIZE = (12, 5)
DPI = 160
SPEED_MULTIPLIER = 2.5         # < 1 = faster, > 1 = slower
TRANSPARENT_BG = True          # True = transparent background
BASE_DURATION_MS = 80          # base duration per frame in ms
GIF_DURATION_MS = int(BASE_DURATION_MS * SPEED_MULTIPLIER)
FPS = 1000 / GIF_DURATION_MS   # frames per second for mp4

# x-domain for plotting
X_MIN = -2.0
X_MAX = 2.0
NUM_X = 3000

# Output root folder
OUTPUT_ROOT = "fractal_activation_animations"


# ============================================================
# HELPER: SAFE FOLDER / FILE NAME
# ============================================================
def slugify(text):
    keep = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "/", "\\", ".", ":", ",", "(", ")"):
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ============================================================
# FUNCTION WRAPPERS
# Each entry:
#   - name
#   - callable builder(depth) -> function(x_tensor)
#   - title
# ============================================================
FUNCTION_SPECS = [
    {
        "name": "basic_blancmange_function",
        "title": "Basic Blancmange Function",
        "builder": lambda depth: (
            lambda x: basic_blancmange_function_tf(x, num_terms=depth)
        ),
    },
    {
        "name": "modulated_blancmange_curve",
        "title": "Modulated Blancmange Curve",
        "builder": lambda depth: (
            lambda x: modulated_blancmange_curve(x, n_terms=depth, a=0.75)
        ),
    },
    {
        "name": "basic_weierstrass_function",
        "title": "Basic Weierstrass Function",
        "builder": lambda depth: (
            lambda x: basic_weierstrass_function_tf(x, a=0.5, b=7, num_terms=depth)
        ),
    },
    {
        "name": "weierstrass_function_tf",
        "title": "Weierstrass Function TF",
        "builder": lambda depth: (
            lambda x: weierstrass_function_tf(x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2))
        ),
    },
    {
        "name": "modified_weierstrass_function_tanh",
        "title": "Modified Weierstrass Function Tanh",
        "builder": lambda depth: (
            lambda x: modified_weierstrass_function_tanh(x, a=0.5, b=3, n_terms=depth)
        ),
    },
    {
        "name": "modified_weierstrass_function_relu",
        "title": "Modified Weierstrass Function ReLU",
        "builder": lambda depth: (
            lambda x: modified_weierstrass_function_relu(x, a=0.5, b=3, n_terms=depth)
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_xsinsquared",
        "title": "Weierstrass-Mandelbrot x*sin^2",
        "builder": lambda depth: (
            lambda x: weierstrass_mandelbrot_function_xsinsquared(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            )
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_xpsin",
        "title": "Weierstrass-Mandelbrot x+sin",
        "builder": lambda depth: (
            lambda x: weierstrass_mandelbrot_function_xpsin(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            )
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_relupsin",
        "title": "Weierstrass-Mandelbrot relu+sin",
        "builder": lambda depth: (
            lambda x: weierstrass_mandelbrot_function_relupsin(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            )
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_tanhpsin",
        "title": "Weierstrass-Mandelbrot tanh+sin",
        "builder": lambda depth: (
            lambda x: weierstrass_mandelbrot_function_tanhpsin(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            )
        ),
    },
    {
        "name": "decaying_cosine_function_tf",
        "title": "Decaying Cosine Function TF",
        "builder": lambda depth: (
            lambda x: decaying_cosine_function_tf(
                x, a=0.5, b=3, c=0.5, d=2, n_terms=max(depth, 2), zeta=0.2666
            )
        ),
    },
]


# ============================================================
# SAMPLING
# ============================================================
def make_x_tensor(x_min=X_MIN, x_max=X_MAX, num_x=NUM_X):
    x_np = np.linspace(x_min, x_max, num_x, dtype=np.float32)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    return x_np, x_tf


# ============================================================
# BUILD LEVELS
# Each level is a partial approximation with increasing depth.
# ============================================================
def build_levels(function_builder, iterations):
    x_np, x_tf = make_x_tensor()
    levels = []

    for depth in range(1, iterations + 1):
        fn = function_builder(depth)
        y_tf = fn(x_tf)
        y_np = np.asarray(y_tf.numpy(), dtype=np.float32)
        levels.append(
            {
                "depth": depth,
                "x": x_np.copy(),
                "y": y_np.copy(),
            }
        )

    return levels


# ============================================================
# PARTIAL DRAWING FOR SMOOTH BUILD-UP
# We reveal the curve from left to right.
# ============================================================
def partial_curve(x, y, fraction):
    if fraction <= 0:
        return x[:1], y[:1]
    if fraction >= 1:
        return x, y

    count = max(2, int(len(x) * fraction))
    return x[:count], y[:count]


# ============================================================
# BOUNDS
# ============================================================
def get_bounds(levels, pad_x=0.03, pad_y=0.08):
    xs = []
    ys = []

    for level in levels:
        xs.extend(level["x"].tolist())
        ys.extend(level["y"].tolist())

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    dx = x_max - x_min
    dy = y_max - y_min if y_max > y_min else 1.0

    return (
        x_min - pad_x * dx,
        x_max + pad_x * dx,
        y_min - pad_y * dy,
        y_max + pad_y * dy,
    )


# ============================================================
# PLOTTING
# ============================================================
def render_frame(x, y, depth_idx, total_depths, bounds, title):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    ax.plot(
        x,
        y,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    ax.text(
        0.02,
        0.94,
        f"{title}  |  Depth {depth_idx}/{total_depths}",
        transform=ax.transAxes,
        fontsize=16,
        color=ACCENT_COLOR,
        ha="left",
        va="top",
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.88,
        "Increasing partial approximation / increasing number of terms",
        transform=ax.transAxes,
        fontsize=10,
        color=ACCENT_COLOR,
        ha="left",
        va="top",
    )

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(False)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG,
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


# ============================================================
# SAVE STATIC LEVEL PLOTS TOO
# ============================================================
def save_level_plot(x, y, depth_idx, total_depths, bounds, title, outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    ax.plot(
        x,
        y,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    ax.text(
        0.02,
        0.94,
        f"{title}  |  Depth {depth_idx}/{total_depths}",
        transform=ax.transAxes,
        fontsize=16,
        color=ACCENT_COLOR,
        ha="left",
        va="top",
        fontweight="bold",
    )

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(False)

    plt.savefig(
        outpath,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG,
    )
    plt.close(fig)


# ============================================================
# CREATE GIF + MP4 FOR ONE FUNCTION
# ============================================================
def create_function_animation(function_spec):
    function_name = function_spec["name"]
    title = function_spec["title"]
    builder = function_spec["builder"]

    function_slug = slugify(function_name)
    function_dir = os.path.join(OUTPUT_ROOT, function_slug)
    frames_dir = os.path.join(function_dir, "frames")
    plots_dir = os.path.join(function_dir, "plots")

    os.makedirs(function_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    output_gif = os.path.join(function_dir, f"{function_slug}.gif")
    output_mp4 = os.path.join(function_dir, f"{function_slug}.mp4")

    levels = build_levels(builder, ITERATIONS)
    bounds = get_bounds(levels)

    frames = []
    frame_counter = 0

    # Build each depth progressively
    for level_idx, level in enumerate(levels, start=1):
        x_full = level["x"]
        y_full = level["y"]

        # save full static plot for this depth
        save_level_plot(
            x_full,
            y_full,
            level_idx,
            ITERATIONS,
            bounds,
            title,
            os.path.join(plots_dir, f"{function_slug}_depth_{level_idx:03d}.png"),
        )

        for k in range(1, FRAMES_PER_SEGMENT + 1):
            frac = k / FRAMES_PER_SEGMENT
            x_part, y_part = partial_curve(x_full, y_full, frac)

            frame = render_frame(
                x_part,
                y_part,
                level_idx,
                ITERATIONS,
                bounds,
                title,
            )
            frames.append(frame)

            frame.save(os.path.join(frames_dir, f"frame_{frame_counter:05d}.png"))
            frame_counter += 1

    # Hold final frame
    final_frame = render_frame(
        levels[-1]["x"],
        levels[-1]["y"],
        ITERATIONS,
        ITERATIONS,
        bounds,
        title,
    )
    for _ in range(FINAL_HOLD_FRAMES):
        frames.append(final_frame.copy())

    # Save GIF
    gif_frames = [frame.convert("P", palette=Image.ADAPTIVE) for frame in frames]
    gif_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
        optimize=False,
        transparency=0 if TRANSPARENT_BG else None,
        disposal=2,
    )

    # Save MP4
    # MP4 does not preserve alpha in the normal H.264 workflow,
    # so transparent regions are composited onto BG_COLOR.
    mp4_frames = []
    for frame in frames:
        rgba = np.array(frame)
        if TRANSPARENT_BG:
            bg_rgb = tuple(int(BG_COLOR[i:i + 2], 16) for i in (1, 3, 5))
            alpha = rgba[:, :, 3:4] / 255.0
            rgb = rgba[:, :, :3] * alpha + np.array(bg_rgb, dtype=np.float32) * (1.0 - alpha)
            rgb = rgb.astype(np.uint8)
        else:
            rgb = np.array(frame.convert("RGB"))
        mp4_frames.append(rgb)

    imageio.mimsave(
        output_mp4,
        mp4_frames,
        fps=FPS,
        codec="libx264",
    )

    print(f"[DONE] {title}")
    print(f"  GIF: {os.path.abspath(output_gif)}")
    print(f"  MP4: {os.path.abspath(output_mp4)}")
    print(f"  Frames: {os.path.abspath(frames_dir)}")
    print(f"  Plots: {os.path.abspath(plots_dir)}")


# ============================================================
# MAIN
# ============================================================
def create_all_fractal_activation_animations():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for function_spec in FUNCTION_SPECS:
        create_function_animation(function_spec)


if __name__ == "__main__":
    create_all_fractal_activation_animations()