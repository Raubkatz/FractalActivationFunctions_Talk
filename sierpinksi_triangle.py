import math
import os
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
import numpy as np


# ============================================================
# COLOR SETTINGS (edit these three hex colors as you want)
# ============================================================
BG_COLOR = "#FFFFFF"      # background
LINE_COLOR = "#1565C0"    # Sierpinski triangle color
ACCENT_COLOR = "#000000"  # text / frame / optional accent


# ============================================================
# ANIMATION SETTINGS
# ============================================================
ITERATIONS = 5                 # recursion depth; 4 or 5 is good for presentations
FRAMES_PER_SEGMENT = 8         # smoothness while each level is drawn
FINAL_HOLD_FRAMES = 12         # how long final frame stays
LINE_WIDTH = 1.5
FIGSIZE = (8, 8)
DPI = 160
SPEED_MULTIPLIER = 0.5         # < 1 = faster, > 1 = slower
TRANSPARENT_BG = True          # True = transparent background
BASE_DURATION_MS = 80          # base duration per frame in ms
GIF_DURATION_MS = int(BASE_DURATION_MS * SPEED_MULTIPLIER)
FPS = 1000 / GIF_DURATION_MS   # frames per second for mp4
OUTPUT_GIF = "sierpinski_triangle_build.gif"
OUTPUT_MP4 = "sierpinski_triangle_build.mp4"


# ============================================================
# SIERPINSKI TRIANGLE GEOMETRY
# ============================================================
def interpolate(p1, p2, t):
    """Linear interpolation between two 2D points."""
    return (
        p1[0] + (p2[0] - p1[0]) * t,
        p1[1] + (p2[1] - p1[1]) * t
    )


def midpoint(p1, p2):
    """Midpoint between two 2D points."""
    return (
        0.5 * (p1[0] + p2[0]),
        0.5 * (p1[1] + p2[1])
    )


def sierpinski_subdivide(triangle):
    """
    Replace one triangle by the 3 corner triangles of the
    Sierpinski construction.
    """
    a, b, c = triangle
    ab = midpoint(a, b)
    bc = midpoint(b, c)
    ca = midpoint(c, a)

    t1 = (a, ab, ca)
    t2 = (ab, b, bc)
    t3 = (ca, bc, c)

    return [t1, t2, t3]


def refine_triangles(triangles):
    """Apply one Sierpinski refinement step to the whole set."""
    new_triangles = []
    for triangle in triangles:
        new_triangles.extend(sierpinski_subdivide(triangle))
    return new_triangles


def build_levels(iterations):
    """
    Create list of triangle sets.
    level 0 = one solid triangle
    level n = Sierpinski refinement of previous level
    """
    levels = []

    h = math.sqrt(3) / 2.0
    base_triangle = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, h)
    )

    current = [base_triangle]
    levels.append(current)

    for _ in range(iterations):
        current = refine_triangles(current)
        levels.append(current)

    return levels


# ============================================================
# PARTIAL DRAWING FOR SMOOTH BUILD-UP
# ============================================================
def partial_triangles(triangles, fraction):
    """
    Return a partial list of triangles according to the given fraction.
    """
    if fraction <= 0:
        return []
    if fraction >= 1:
        return triangles[:]

    count = max(1, int(len(triangles) * fraction))
    return triangles[:count]


# ============================================================
# PLOTTING
# ============================================================
def get_bounds(levels, pad=0.06):
    xs = []
    ys = []
    for level in levels:
        for triangle in level:
            for x, y in triangle:
                xs.append(x)
                ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    return (
        x_min - pad * dx,
        x_max + pad * dx,
        y_min - pad * max(dx, dy),
        y_max + pad * max(dx, dy),
    )


def render_frame(triangles, level_idx, total_levels, bounds):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    for triangle in triangles:
        xs = [triangle[0][0], triangle[1][0], triangle[2][0], triangle[0][0]]
        ys = [triangle[0][1], triangle[1][1], triangle[2][1], triangle[0][1]]

        ax.fill(xs, ys, color=LINE_COLOR, linewidth=0)
        ax.plot(
            xs,
            ys,
            color=LINE_COLOR,
            linewidth=LINE_WIDTH,
            solid_joinstyle="round",
            solid_capstyle="round",
        )

    # Title for presentation use
    ax.text(
        0.02,
        0.93,
        f"Sierpinski Triangle Construction  |  Iteration {level_idx}/{total_levels}",
        transform=ax.transAxes,
        fontsize=16,
        color=ACCENT_COLOR,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # Minimal clean style
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # Optional subtle frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = BytesIO()
    plt.savefig(
        buf,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


# ============================================================
# MAIN GIF + MP4 CREATION
# ============================================================
def create_sierpinski_gif_and_mp4(
    output_gif=OUTPUT_GIF,
    output_mp4=OUTPUT_MP4,
    iterations=ITERATIONS
):
    levels = build_levels(iterations)
    bounds = get_bounds(levels)

    frames = []

    # Build each iteration progressively
    for level_idx, level_triangles in enumerate(levels):
        for k in range(1, FRAMES_PER_SEGMENT + 1):
            frac = k / FRAMES_PER_SEGMENT
            partial = partial_triangles(level_triangles, frac)
            frame = render_frame(partial, level_idx, iterations, bounds)
            frames.append(frame)

    # Hold the final frame a bit longer
    final_frame = render_frame(levels[-1], iterations, iterations, bounds)
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
        disposal=2
    )

    # Save MP4
    # Note: MP4 does not support transparency in the usual H.264 workflow.
    # If TRANSPARENT_BG is True, transparent areas are composited onto BG_COLOR.
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
        codec="libx264"
    )

    print(f"GIF saved to: {os.path.abspath(output_gif)}")
    print(f"MP4 saved to: {os.path.abspath(output_mp4)}")


if __name__ == "__main__":
    create_sierpinski_gif_and_mp4()