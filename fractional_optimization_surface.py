#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimize_fractal_surface_with_fractional_optimizers.py

Optimize a fractal-like 2D surface with TensorFlow optimizers, including the
custom fractional optimizers already implemented in the local class files.

What this script does
---------------------
1. Defines a noisy / fractal-like objective surface:
       f(x, y) = sum_k a^k cos(b^k pi x) + sum_k a^k cos(b^k pi y)
   with an optional small quadratic regularizer for numerical containment.

2. Treats (x, y) as trainable TensorFlow variables.

3. Runs multiple optimizers on the same problem:
   - standard TensorFlow optimizers
   - Herrera-style fractional optimizers
   - explicit memory-based fractional optimizers
   - adaptive variable-order memory-based Adam
   - adaptive randomized variable-order memory-based FRMSprop

4. Records trajectories, losses, gradients, and optional order diagnostics.

5. Produces:
   - a contour plot with all trajectories,
   - a convergence plot,
   - an MP4 animation of the optimization process,
   - a JSON summary of results.

Requirements
------------
- TensorFlow
- NumPy
- Matplotlib
- ffmpeg available on the system path for MP4 writing

Local optimizer modules expected
--------------------------------
- class_b_tf_fractional_optimizers_OHA.py
- class_c_tf_fractional_optimizers_SR.py
- class_e_tf_gen_var_fadam.py
- class_tf_gen_var_frmsprop.py

Notes
-----
- The custom optimizers are used as TensorFlow optimizers through
  `apply_gradients`.
- The variable-order optimizers may expose `set_current_loss(...)`,
  `current_orders()`, and related helpers. This script uses them when present.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------------------------------------------------------------------
# LOCAL OPTIMIZER IMPORTS
# ---------------------------------------------------------------------
from class_b_tf_fractional_optimizers_OHA import (
    FSGD,
    FAdam,
    FAdagrad,
    FAdadelta,
    FRMSprop,
    FAdamW,
)
from class_c_tf_fractional_optimizers_SR import (
    MemoryFSGD,
    MemoryFRMSprop,
    MemoryFAdam,
    MemoryFAdadelta,
)
from class_e_tf_gen_var_fadam import AdaptiveModesVariableOrderMemoryFAdam
from class_tf_gen_var_frmsprop import AdaptiveRandomVariableOrderMemoryFRMSprop


# =============================================================================
# 0) GLOBAL SETTINGS
# =============================================================================

OUTPUT_DIR = Path("fractal_surface_optimizer_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42
DTYPE = tf.float32
tf.keras.backend.set_floatx("float32")

# Surface parameters
FRACTAL_A = 0.55
FRACTAL_B = 2.8
FRACTAL_K = 9
USE_QUADRATIC_REGULARIZER = True
QUADRATIC_WEIGHT = 0.01

# Domain for visualization
XMIN, XMAX = -2.5, 2.5
YMIN, YMAX = -2.5, 2.5
GRID_N = 300

# Optimization settings
INITIAL_POINT = np.array([1.75, -1.35], dtype=np.float32)
MAX_STEPS = 250
GRAD_CLIP_NORM = None  # set e.g. 10.0 if desired
STOP_IF_GRAD_NORM_BELOW = None  # set e.g. 1e-6 if desired

# Animation settings
FPS = 20
TRAIL_LENGTH = None  # set integer to restrict visible trail length
ANIMATION_FILENAME = OUTPUT_DIR / "fractal_surface_optimization.mp4"

# Plot settings
CONTOUR_LEVELS = 80
FIGSIZE = (10, 8)

# Fractional optimizer defaults
HERRERA_VDERIV = 0.8
MEMORY_VDERIV = 0.8
MEMORY_HISTORY_SIZE = 6
MEMORY_NORMALIZE_COEFFS = False

VAR_ORDER_INIT = 0.85
VAR_ORDER_MIN = 0.6
VAR_ORDER_MAX = 1.2
VAR_ORDER_HISTORY_SIZE = 6
VAR_ORDER_NORMALIZE_COEFFS = False
VAR_ORDER_ADAPT_RATE = 0.05
VAR_ORDER_DELTA = 1e-8
VAR_ORDER_EMA_GAMMA = 0.95
VAR_ORDER_TOTAL_SCHEDULE_STEPS = MAX_STEPS
VAR_ORDER_SCHEDULE_EXP_GAMMA = 4.0
VAR_ORDER_STABLE_NU_CAP = 1.0
VAR_ORDER_VARIABILITY_CLIP = 10.0
VAR_ORDER_ORDER_EMA_GAMMA = 0.90
VAR_ORDER_WARMUP_STEPS = 50
VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT = 1.0
VAR_ORDER_LOSS_EMA_BETA = 0.95
VAR_ORDER_LR_DECAY_PATIENCE = 50
VAR_ORDER_LR_DECAY_FACTOR = 0.7
VAR_ORDER_MIN_LR_FACTOR = 0.2
VAR_ORDER_LOSS_PLATEAU_TOLERANCE = 1e-4

ARVFRMSPROP_NU_MIN_OFFSET = 0.25
ARVFRMSPROP_NU_MAX_OFFSET = 1.5
ARVFRMSPROP_RANDOM_PERTURB_SCALE = 0.03
ARVFRMSPROP_RANDOM_DECAY = 0.999
ARVFRMSPROP_STABILITY_STEP_SIZE = 0.01
ARVFRMSPROP_STABILITY_EMA_GAMMA = 0.95
ARVFRMSPROP_STABLE_NU_CAP = 1.0
ARVFRMSPROP_WARMUP_STEPS = 50
ARVFRMSPROP_USE_SIGNED_RANDOM_SEARCH = True

# Standard optimizer defaults
LR_SGD = 1e-2
LR_ADAM = 1e-2
LR_RMSPROP = 5e-3
LR_ADAGRAD = 2e-2
LR_ADADELTA = 1.0
LR_ADAMW = 5e-3

SGD_MOMENTUM = 0.9
SGD_NESTEROV = False

RMSPROP_RHO = 0.9
RMSPROP_MOMENTUM = 0.0
RMSPROP_CENTERED = False

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-7
ADAM_AMSGRAD = False

ADAGRAD_INITIAL_ACCUMULATOR = 0.1
ADAGRAD_EPSILON = 1e-7

ADADELTA_RHO = 0.95
ADADELTA_EPSILON = 1e-7

ADAMW_WEIGHT_DECAY = 4e-3
HERRERA_FRAC_EPSILON = 1e-6


# Choose which optimizers to compare.
# You can comment out entries if you want a smaller animation.
OPTIMIZER_NAMES = [
    # Standard
    "SGD",
    "Adam",
    "RMSprop",
    # Herrera-style fractional
    "FSGD",
    "FAdam",
    "FRMSprop",
    "FAdagrad",
    "FAdadelta",
    "FAdamW",
    # Memory-based fractional
    "MemoryFSGD",
    "MemoryFRMSprop",
    "MemoryFAdam",
    "MemoryFAdadelta",
    # Adaptive variable-order fractional
    "AdaptiveModesVariableOrderMemoryFAdam__ema",
    "AdaptiveModesVariableOrderMemoryFAdam__schedule",
    "AdaptiveModesVariableOrderMemoryFAdam__hybrid",
    # Adaptive randomized variable-order fractional RMSprop
    "AdaptiveRandomVariableOrderMemoryFRMSprop",
]


# =============================================================================
# 1) REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_seed(GLOBAL_SEED)


# =============================================================================
# 2) OBJECTIVE FUNCTION
# =============================================================================

def fractal_surface_tf(
    xy: tf.Tensor,
    a: float = FRACTAL_A,
    b: float = FRACTAL_B,
    k_max: int = FRACTAL_K,
    use_quadratic: bool = USE_QUADRATIC_REGULARIZER,
    quadratic_weight: float = QUADRATIC_WEIGHT,
) -> tf.Tensor:
    """
    Fractal-like 2D surface:
        f(x, y) = sum_{k=0}^{K} a^k cos(b^k pi x) + sum_{k=0}^{K} a^k cos(b^k pi y)
    plus an optional small quadratic regularizer.
    """
    x = tf.cast(xy[0], DTYPE)
    y = tf.cast(xy[1], DTYPE)

    value = tf.constant(0.0, dtype=DTYPE)

    for k in range(k_max + 1):
        ak = tf.cast(a ** k, DTYPE)
        bk = tf.cast(b ** k, DTYPE)
        value += ak * tf.cos(bk * tf.constant(np.pi, dtype=DTYPE) * x)
        value += ak * tf.cos(bk * tf.constant(np.pi, dtype=DTYPE) * y)

    if use_quadratic:
        value += tf.cast(quadratic_weight, DTYPE) * (x * x + y * y)

    return value


def fractal_surface_numpy(
    x: np.ndarray,
    y: np.ndarray,
    a: float = FRACTAL_A,
    b: float = FRACTAL_B,
    k_max: int = FRACTAL_K,
    use_quadratic: bool = USE_QUADRATIC_REGULARIZER,
    quadratic_weight: float = QUADRATIC_WEIGHT,
) -> np.ndarray:
    value = np.zeros_like(x, dtype=np.float64)

    for k in range(k_max + 1):
        ak = a ** k
        bk = b ** k
        value += ak * np.cos(bk * np.pi * x)
        value += ak * np.cos(bk * np.pi * y)

    if use_quadratic:
        value += quadratic_weight * (x ** 2 + y ** 2)

    return value


# =============================================================================
# 3) OPTIMIZER FACTORY
# =============================================================================

def build_optimizer(optimizer_name: str) -> tf.keras.optimizers.Optimizer:
    # Standard
    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        )

    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    if optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
        )

    if optimizer_name == "Adagrad":
        return tf.keras.optimizers.Adagrad(
            lr=LR_ADAGRAD,
            initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=ADAGRAD_EPSILON,
        )

    if optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
        )

    if optimizer_name == "AdamW":
        return tf.keras.optimizers.AdamW(
            lr=LR_ADAMW,
            weight_decay=ADAMW_WEIGHT_DECAY,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    # Herrera-style fractional
    if optimizer_name == "FSGD":
        return FSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=HERRERA_VDERIV,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdam":
        return FAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=HERRERA_VDERIV,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FRMSprop":
        return FRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=HERRERA_VDERIV,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdagrad":
        return FAdagrad(
            lr=LR_ADAGRAD,
            initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=ADAGRAD_EPSILON,
            vderiv=HERRERA_VDERIV,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdadelta":
        return FAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=HERRERA_VDERIV,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdamW":
        return FAdamW(
            lr=LR_ADAMW,
            weight_decay=ADAMW_WEIGHT_DECAY,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=HERRERA_VDERIV,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    # Memory-based fractional
    if optimizer_name == "MemoryFSGD":
        return MemoryFSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=MEMORY_VDERIV,
            history_size=MEMORY_HISTORY_SIZE,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFRMSprop":
        return MemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=MEMORY_VDERIV,
            history_size=MEMORY_HISTORY_SIZE,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFAdam":
        return MemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=MEMORY_VDERIV,
            history_size=MEMORY_HISTORY_SIZE,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFAdadelta":
        return MemoryFAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=MEMORY_VDERIV,
            history_size=MEMORY_HISTORY_SIZE,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    # Adaptive variable-order fractional Adam
    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam__ema":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode="ema_smoothed_gradient_variability",
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type="cosine",
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam__schedule":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode="schedule",
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type="cosine",
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam__hybrid":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode="hybrid_transition",
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type="cosine",
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    # Adaptive randomized variable-order memory FRMSprop
    if optimizer_name == "AdaptiveRandomVariableOrderMemoryFRMSprop":
        return AdaptiveRandomVariableOrderMemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv_init=MEMORY_VDERIV,
            nu_min=ARVFRMSPROP_NU_MIN_OFFSET,
            nu_max=ARVFRMSPROP_NU_MAX_OFFSET,
            history_size=MEMORY_HISTORY_SIZE,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
            order_delta=VAR_ORDER_DELTA,
            random_perturb_scale=ARVFRMSPROP_RANDOM_PERTURB_SCALE,
            random_decay=ARVFRMSPROP_RANDOM_DECAY,
            stability_step_size=ARVFRMSPROP_STABILITY_STEP_SIZE,
            stability_ema_gamma=ARVFRMSPROP_STABILITY_EMA_GAMMA,
            stable_nu_cap=ARVFRMSPROP_STABLE_NU_CAP,
            warmup_steps=ARVFRMSPROP_WARMUP_STEPS,
            use_signed_random_search=ARVFRMSPROP_USE_SIGNED_RANDOM_SEARCH,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


# =============================================================================
# 4) OPTIMIZATION LOOP
# =============================================================================

def optimize_surface(
    optimizer_name: str,
    initial_point: np.ndarray,
    max_steps: int = MAX_STEPS,
) -> Dict[str, Any]:
    """
    Optimize the fractal surface with one optimizer.
    """
    xy = tf.Variable(initial_point.astype(np.float32), dtype=DTYPE, name=f"xy_{optimizer_name}")
    optimizer = build_optimizer(optimizer_name)

    trajectory = []
    losses = []
    grad_norms = []
    orders = []
    lr_factors = []

    for step in range(max_steps):
        with tf.GradientTape() as tape:
            loss = fractal_surface_tf(xy)

        grads = tape.gradient(loss, [xy])

        if GRAD_CLIP_NORM is not None:
            grads = [tf.clip_by_norm(g, GRAD_CLIP_NORM) if g is not None else g for g in grads]

        if hasattr(optimizer, "set_current_loss"):
            try:
                optimizer.set_current_loss(loss)
            except Exception:
                pass

        optimizer.apply_gradients(zip(grads, [xy]))

        xy_np = xy.numpy().copy()
        grad_np = grads[0].numpy().copy()
        grad_norm = float(np.linalg.norm(grad_np))

        trajectory.append(xy_np.tolist())
        losses.append(float(loss.numpy()))
        grad_norms.append(grad_norm)

        if hasattr(optimizer, "current_orders"):
            try:
                curr_orders = optimizer.current_orders()
                mean_order = float(np.mean([float(o.numpy()) for o in curr_orders]))
            except Exception:
                mean_order = None
        else:
            mean_order = None
        orders.append(mean_order)

        if hasattr(optimizer, "current_lr_factor"):
            try:
                lr_factor = float(optimizer.current_lr_factor())
            except Exception:
                lr_factor = None
        else:
            lr_factor = None
        lr_factors.append(lr_factor)

        if STOP_IF_GRAD_NORM_BELOW is not None and grad_norm < STOP_IF_GRAD_NORM_BELOW:
            break

    final_xy = np.array(trajectory[-1], dtype=np.float32)
    final_loss = float(fractal_surface_tf(tf.constant(final_xy, dtype=DTYPE)).numpy())

    return {
        "optimizer_name": optimizer_name,
        "trajectory": trajectory,
        "losses": losses,
        "grad_norms": grad_norms,
        "orders": orders,
        "lr_factors": lr_factors,
        "n_steps": len(trajectory),
        "initial_point": initial_point.tolist(),
        "final_point": final_xy.tolist(),
        "final_loss": final_loss,
    }


# =============================================================================
# 5) VISUALIZATION GRID
# =============================================================================

def compute_surface_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(XMIN, XMAX, GRID_N)
    ys = np.linspace(YMIN, YMAX, GRID_N)
    X, Y = np.meshgrid(xs, ys)
    Z = fractal_surface_numpy(X, Y)
    return X, Y, Z


# =============================================================================
# 6) PLOTS
# =============================================================================

def save_static_trajectory_plot(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    results: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    contour = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    plt.colorbar(contour, ax=ax, label="Objective value")

    for res in results:
        traj = np.array(res["trajectory"])
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=res["optimizer_name"])
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=35)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=50)

    ax.set_title("Optimizer trajectories on fractal-like surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.legend(fontsize=8, loc="upper right", ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_convergence_plot(results: List[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        ax.plot(res["losses"], label=res["optimizer_name"])

    ax.set_title("Objective value during optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("f(x, y)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_gradient_norm_plot(results: List[Dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        ax.plot(res["grad_norms"], label=res["optimizer_name"])

    ax.set_title("Gradient norm during optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel("||grad||")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# 7) ANIMATION
# =============================================================================

def save_animation(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    results: List[Dict[str, Any]],
    out_path: Path,
    fps: int = FPS,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    contour = ax.contourf(X, Y, Z, levels=CONTOUR_LEVELS)
    plt.colorbar(contour, ax=ax, label="Objective value")

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimization trajectories on fractal-like surface")

    max_len = max(res["n_steps"] for res in results)

    line_handles = {}
    point_handles = {}
    text_handles = {}

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for color, res in zip(colors, results):
        name = res["optimizer_name"]
        line, = ax.plot([], [], linewidth=1.8, color=color, label=name)
        point, = ax.plot([], [], marker="o", markersize=5, color=color)
        line_handles[name] = line
        point_handles[name] = point

    ax.legend(fontsize=8, loc="upper right")

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def init():
        for res in results:
            name = res["optimizer_name"]
            line_handles[name].set_data([], [])
            point_handles[name].set_data([], [])
        info_text.set_text("")
        artists = list(line_handles.values()) + list(point_handles.values()) + [info_text]
        return artists

    def update(frame: int):
        text_lines = [f"step = {frame + 1}"]

        for res in results:
            name = res["optimizer_name"]
            traj = np.array(res["trajectory"])
            last_idx = min(frame, len(traj) - 1)

            if TRAIL_LENGTH is None:
                start_idx = 0
            else:
                start_idx = max(0, last_idx - TRAIL_LENGTH + 1)

            segment = traj[start_idx:last_idx + 1]

            line_handles[name].set_data(segment[:, 0], segment[:, 1])
            point_handles[name].set_data(traj[last_idx, 0], traj[last_idx, 1])

            loss_now = res["losses"][last_idx]
            text_lines.append(f"{name}: loss={loss_now:.5f}")

        info_text.set_text("\n".join(text_lines))
        artists = list(line_handles.values()) + list(point_handles.values()) + [info_text]
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=max_len,
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out_path), writer=writer, dpi=160)
    plt.close(fig)


# =============================================================================
# 8) RESULT EXPORT
# =============================================================================

def save_results_json(results: List[Dict[str, Any]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("=" * 100)
    print("Fractal surface optimization summary")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda r: r["final_loss"])
    for rank, res in enumerate(sorted_results, start=1):
        final_point = res["final_point"]
        print(
            f"[{rank:02d}] {res['optimizer_name']:45s} "
            f"final_loss={res['final_loss']:.6f} "
            f"steps={res['n_steps']:3d} "
            f"final_point=({final_point[0]:.5f}, {final_point[1]:.5f})"
        )


# =============================================================================
# 9) MAIN
# =============================================================================

def main() -> None:
    print("Computing surface grid...")
    X, Y, Z = compute_surface_grid()

    results: List[Dict[str, Any]] = []

    print("\nRunning optimizers...")
    for optimizer_name in OPTIMIZER_NAMES:
        print(f"  -> {optimizer_name}")
        result = optimize_surface(
            optimizer_name=optimizer_name,
            initial_point=INITIAL_POINT,
            max_steps=MAX_STEPS,
        )
        results.append(result)

    print_summary(results)

    print("\nSaving outputs...")
    save_static_trajectory_plot(
        X=X,
        Y=Y,
        Z=Z,
        results=results,
        out_path=OUTPUT_DIR / "trajectories.png",
    )
    save_convergence_plot(
        results=results,
        out_path=OUTPUT_DIR / "convergence.png",
    )
    save_gradient_norm_plot(
        results=results,
        out_path=OUTPUT_DIR / "gradient_norms.png",
    )
    save_results_json(
        results=results,
        out_path=OUTPUT_DIR / "results.json",
    )
    save_animation(
        X=X,
        Y=Y,
        Z=Z,
        results=results,
        out_path=ANIMATION_FILENAME,
        fps=FPS,
    )

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"MP4 animation: {ANIMATION_FILENAME}")


if __name__ == "__main__":
    main()