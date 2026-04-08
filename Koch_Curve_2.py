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
LINE_COLOR = "#1565C0"    # Koch curve color
ACCENT_COLOR = "#000000"  # text / frame / optional accent


# ============================================================
# ANIMATION SETTINGS
# ============================================================
ITERATIONS = 5                 # recursion depth; 4 or 5 is good for presentations
FRAMES_PER_SEGMENT = 8         # smoothness while each level is drawn
FINAL_HOLD_FRAMES = 12         # how long final frame stays
LINE_WIDTH = 2.5
FIGSIZE = (12, 4)
DPI = 160
SPEED_MULTIPLIER = 2.5         # < 1 = faster, > 1 = slower
TRANSPARENT_BG = True          # True = transparent background
BASE_DURATION_MS = 80          # base duration per frame in ms
GIF_DURATION_MS = int(BASE_DURATION_MS * SPEED_MULTIPLIER)
FPS = 1000 / GIF_DURATION_MS   # frames per second for mp4
OUTPUT_GIF = "koch_curve_build.gif"
OUTPUT_MP4 = "koch_curve_build.mp4"


# ============================================================
# KOCH CURVE GEOMETRY
# ============================================================
def interpolate(p1, p2, t):
    """Linear interpolation between two 2D points."""
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


def koch_subdivide(p1, p2):
    """
    Replace one line segment by the 4 Koch segments.
    Returns 5 points: A, B, C, D, E
    where segment A->E becomes A->B->C->D->E.
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # points at 1/3 and 2/3
    b = (x1 + dx / 3.0, y1 + dy / 3.0)
    d = (x1 + 2.0 * dx / 3.0, y1 + 2.0 * dy / 3.0)

    # rotate vector (d - b) by +60° around b to get peak c
    angle = math.radians(60)
    vx = d[0] - b[0]
    vy = d[1] - b[1]

    cx = b[0] + vx * math.cos(angle) - vy * math.sin(angle)
    cy = b[1] + vx * math.sin(angle) + vy * math.cos(angle)
    c = (cx, cy)

    return [p1, b, c, d, p2]


def refine_polyline(points):
    """Apply one Koch refinement step to the whole polyline."""
    new_points = []
    for i in range(len(points) - 1):
        sub = koch_subdivide(points[i], points[i + 1])
        if i == 0:
            new_points.extend(sub)
        else:
            new_points.extend(sub[1:])  # avoid duplicate point
    return new_points


def build_levels(iterations):
    """
    Create list of polyline levels.
    level 0 = straight line
    level n = Koch refinement of previous level
    """
    levels = []
    base = [(0.0, 0.0), (1.0, 0.0)]
    levels.append(base)

    current = base
    for _ in range(iterations):
        current = refine_polyline(current)
        levels.append(current)

    return levels


# ============================================================
# PARTIAL DRAWING FOR SMOOTH BUILD-UP
# ============================================================
def path_length(points):
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += math.hypot(dx, dy)
    return total


def partial_polyline(points, fraction):
    """
    Return a partial polyline containing the first `fraction`
    of the total path length.
    """
    if fraction <= 0:
        return [points[0]]
    if fraction >= 1:
        return points[:]

    total_len = path_length(points)
    target_len = fraction * total_len

    result = [points[0]]
    accumulated = 0.0

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if accumulated + seg_len < target_len:
            result.append(p2)
            accumulated += seg_len
        else:
            remaining = target_len - accumulated
            if seg_len > 0:
                t = remaining / seg_len
                result.append(interpolate(p1, p2, t))
            break

    return result


# ============================================================
# PLOTTING
# ============================================================
def get_bounds(levels, pad=0.06):
    xs = []
    ys = []
    for level in levels:
        for x, y in level:
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


def render_frame(points, level_idx, total_levels, bounds):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

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
        f"Koch Curve Construction  |  Iteration {level_idx}/{total_levels}",
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
def create_koch_gif_and_mp4(
    output_gif=OUTPUT_GIF,
    output_mp4=OUTPUT_MP4,
    iterations=ITERATIONS
):
    levels = build_levels(iterations)
    bounds = get_bounds(levels)

    frames = []

    # Build each iteration progressively
    for level_idx, level_points in enumerate(levels):
        for k in range(1, FRAMES_PER_SEGMENT + 1):
            frac = k / FRAMES_PER_SEGMENT
            partial = partial_polyline(level_points, frac)
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
    create_koch_gif_and_mp4()