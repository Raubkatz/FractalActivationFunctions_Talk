"""
adaptive_modes_variable_order_memory_fadam_legacy.py

Legacy TensorFlow / Keras implementation of a variable-order, memory-based
fractional Adam optimizer with three improved adaptation modes:

    1. ema_smoothed_gradient_variability
    2. schedule
    3. hybrid_transition

Design goals
------------
- Keep the implementation close to the previous legacy optimizer style.
- Preserve the explicit short-memory fractional gradient construction.
- Reduce sensitivity to minibatch noise.
- Improve stability by defaulting to a safe effective upper cap on the order.
- Improve speed by avoiding a fully fractional second moment in the two simpler
  modes, and by using a gradual second-moment transition in the hybrid mode.
- Keep compatibility with custom loops that call `set_current_loss(loss)`.

Compatibility notes
-------------------
- This file targets older tf.keras optimizer_v2 style APIs.
- It accepts both `learning_rate=` and `lr=`.
- It preserves the class name:
      AdaptiveModesVariableOrderMemoryFAdam
- It keeps `set_current_loss(...)` and `current_orders()`.

Mode aliases
------------
For convenience, old mode names are mapped as follows:
    "gradient_variability"              -> "ema_smoothed_gradient_variability"
    "ema_smoothed_gradient_variability" -> "ema_smoothed_gradient_variability"
    "schedule"                          -> "schedule"
    "loss_improvement"                  -> "hybrid_transition"

Main implementation changes
---------------------------
1. Order updates are smoothed, clipped, and warm-started from 1.0.
2. Effective order is capped by default at `stable_nu_cap=1.0`.
3. The second moment uses:
       - raw gradient in `ema_smoothed_gradient_variability`
       - raw gradient in `schedule`
       - scheduled mix of raw/fractional gradient in `hybrid_transition`
4. `set_current_loss(...)` updates an EMA of loss and a conservative internal
   learning-rate multiplier in hybrid mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import tensorflow as tf


class AdaptiveModesVariableOrderMemoryFAdam(tf.keras.optimizers.Optimizer):
    """
    Legacy-API variable-order memory-based fractional Adam.

    Supported adaptation modes:
        - "ema_smoothed_gradient_variability"
        - "schedule"
        - "hybrid_transition"
    """

    _VALID_ADAPTATION_MODES = {
        "ema_smoothed_gradient_variability",
        "schedule",
        "hybrid_transition",
    }

    _MODE_ALIASES = {
        "gradient_variability": "ema_smoothed_gradient_variability",
        "ema_smoothed_gradient_variability": "ema_smoothed_gradient_variability",
        "schedule": "schedule",
        "loss_improvement": "hybrid_transition",
        "hybrid_transition": "hybrid_transition",
    }

    _VALID_SCHEDULE_TYPES = {
        "linear",
        "cosine",
        "exponential",
    }

    def __init__(
        self,
        learning_rate: float = 1e-3,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv_init: float = 0.9,
        nu_min: float = 0.4,
        nu_max: float = 1.6,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        classical_fallback_tolerance: float = 1e-8,
        adaptation_mode: str = "ema_smoothed_gradient_variability",
        order_adapt_rate: float = 0.05,
        order_delta: float = 1e-8,
        ema_smoothing_gamma: float = 0.95,
        schedule_type: str = "cosine",
        total_schedule_steps: int = 10000,
        schedule_exponential_gamma: float = 4.0,
        stable_nu_cap: float = 1.0,
        variability_clip: float = 10.0,
        order_ema_gamma: float = 0.90,
        warmup_steps: int = 100,
        hybrid_final_fractional_second_moment: float = 1.0,
        loss_ema_beta: float = 0.95,
        lr_decay_patience: int = 100,
        lr_decay_factor: float = 0.7,
        min_lr_factor: float = 0.2,
        loss_plateau_tolerance: float = 1e-4,
        name: str = "AdaptiveModesVariableOrderMemoryFAdam",
        **kwargs: Any,
    ):
        super(AdaptiveModesVariableOrderMemoryFAdam, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        if not (0.0 < nu_min < nu_max < 2.0):
            raise ValueError("`nu_min` and `nu_max` must satisfy 0 < nu_min < nu_max < 2.")
        if not (nu_min <= vderiv_init <= nu_max):
            raise ValueError("`vderiv_init` must lie in [nu_min, nu_max].")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if classical_fallback_tolerance <= 0.0:
            raise ValueError("`classical_fallback_tolerance` must be > 0.")

        adaptation_mode = self._MODE_ALIASES.get(adaptation_mode, adaptation_mode)
        if adaptation_mode not in self._VALID_ADAPTATION_MODES:
            raise ValueError(
                "`adaptation_mode` must be one of {}.".format(
                    sorted(self._VALID_ADAPTATION_MODES)
                )
            )
        if schedule_type not in self._VALID_SCHEDULE_TYPES:
            raise ValueError(
                "`schedule_type` must be one of {}.".format(
                    sorted(self._VALID_SCHEDULE_TYPES)
                )
            )

        if order_adapt_rate < 0.0:
            raise ValueError("`order_adapt_rate` must be >= 0.")
        if order_delta <= 0.0:
            raise ValueError("`order_delta` must be > 0.")
        if not (0.0 <= ema_smoothing_gamma < 1.0):
            raise ValueError("`ema_smoothing_gamma` must satisfy 0 <= ema_smoothing_gamma < 1.")
        if total_schedule_steps < 1:
            raise ValueError("`total_schedule_steps` must be >= 1.")
        if schedule_exponential_gamma <= 0.0:
            raise ValueError("`schedule_exponential_gamma` must be > 0.")
        if stable_nu_cap <= 0.0:
            raise ValueError("`stable_nu_cap` must be > 0.")
        if variability_clip <= 0.0:
            raise ValueError("`variability_clip` must be > 0.")
        if not (0.0 <= order_ema_gamma < 1.0):
            raise ValueError("`order_ema_gamma` must satisfy 0 <= order_ema_gamma < 1.")
        if warmup_steps < 0:
            raise ValueError("`warmup_steps` must be >= 0.")
        if not (0.0 <= hybrid_final_fractional_second_moment <= 1.0):
            raise ValueError("`hybrid_final_fractional_second_moment` must lie in [0, 1].")
        if not (0.0 <= loss_ema_beta < 1.0):
            raise ValueError("`loss_ema_beta` must satisfy 0 <= loss_ema_beta < 1.")
        if lr_decay_patience < 1:
            raise ValueError("`lr_decay_patience` must be >= 1.")
        if not (0.0 < lr_decay_factor <= 1.0):
            raise ValueError("`lr_decay_factor` must satisfy 0 < lr_decay_factor <= 1.")
        if not (0.0 < min_lr_factor <= 1.0):
            raise ValueError("`min_lr_factor` must satisfy 0 < min_lr_factor <= 1.")
        if loss_plateau_tolerance < 0.0:
            raise ValueError("`loss_plateau_tolerance` must be >= 0.")

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)

        self.vderiv_init = float(vderiv_init)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.history_size = int(history_size)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.classical_fallback_tolerance = float(classical_fallback_tolerance)

        self.adaptation_mode = adaptation_mode
        self.order_adapt_rate = float(order_adapt_rate)
        self.order_delta = float(order_delta)
        self.ema_smoothing_gamma = float(ema_smoothing_gamma)

        self.schedule_type = schedule_type
        self.total_schedule_steps = int(total_schedule_steps)
        self.schedule_exponential_gamma = float(schedule_exponential_gamma)

        self.stable_nu_cap = float(stable_nu_cap)
        self.variability_clip = float(variability_clip)
        self.order_ema_gamma = float(order_ema_gamma)
        self.warmup_steps = int(warmup_steps)

        self.hybrid_final_fractional_second_moment = float(hybrid_final_fractional_second_moment)

        self.loss_ema_beta = float(loss_ema_beta)
        self.lr_decay_patience = int(lr_decay_patience)
        self.lr_decay_factor = float(lr_decay_factor)
        self.min_lr_factor = float(min_lr_factor)
        self.loss_plateau_tolerance = float(loss_plateau_tolerance)

        # Loss-related Python-side states used by hybrid mode
        self._current_loss_value = None
        self._prev_loss_value = None
        self._loss_state_initialized = False

        self._loss_ema_value = None
        self._best_loss_ema = None
        self._bad_loss_steps = 0
        self._lr_factor = 1.0

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _to_dense_gradient(self, grad, var):
        """
        Convert sparse gradients to dense tensors.
        """
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                indices=tf.expand_dims(grad.indices, axis=1),
                updates=grad.values,
                shape=tf.shape(var),
            )
        return grad

    def _scalar_from_slot_tensor(self, slot_tensor, dtype):
        """
        Extract a scalar state value from a slot tensor that has the same shape
        as the variable. We use the mean so the result is stable even if the
        tensor was filled by broadcasting.
        """
        return tf.cast(tf.reduce_mean(slot_tensor), dtype)

    def _filled_like_var(self, scalar_value, var):
        """
        Broadcast a scalar value to the full shape of `var`.
        """
        scalar_value = tf.cast(scalar_value, var.dtype.base_dtype)
        return tf.ones_like(var, dtype=var.dtype.base_dtype) * scalar_value

    def _effective_nu_bounds(self, dtype):
        """
        Return a conservative effective order interval.
        """
        nu_min = tf.cast(self.nu_min, dtype)
        nu_max = tf.cast(min(self.nu_max, self.stable_nu_cap), dtype)
        nu_max = tf.maximum(nu_max, nu_min)
        return nu_min, nu_max

    def _global_l2_norm(self, x, dtype):
        x = tf.cast(x, dtype)
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    def _compute_gl_coefficients_tensor(self, nu, dtype):
        """
        Compute short-memory Grünwald-Letnikov-style coefficients for scalar nu.
        Returns tensor of shape [history_size].
        """
        nu = tf.cast(nu, dtype)
        one = tf.cast(1.0, dtype)
        tol = tf.cast(self.classical_fallback_tolerance, dtype)

        def classical_case():
            first = tf.ones((1,), dtype=dtype)
            rest = tf.zeros((self.history_size - 1,), dtype=dtype)
            return tf.concat([first, rest], axis=0)

        def fractional_case():
            coeffs = [one]
            for k in range(1, self.history_size):
                k_t = tf.cast(k, dtype)
                coeff_k = (one - (nu + one) / k_t) * coeffs[-1]
                coeffs.append(coeff_k)

            coeffs_t = tf.stack(coeffs, axis=0)

            if self.normalize_coefficients:
                denom = tf.reduce_sum(tf.abs(coeffs_t))
                denom = tf.maximum(denom, tf.cast(1e-12, dtype))
                coeffs_t = coeffs_t / denom

            return coeffs_t

        return tf.cond(
            tf.abs(nu - one) <= tol,
            classical_case,
            fractional_case,
        )

    def _broadcast_coefficients(self, coeffs, var):
        reshape_shape = [1] * len(var.shape) + [self.history_size]
        return tf.reshape(coeffs, reshape_shape)

    def _history_slot_name(self, lag: int) -> str:
        return "grad_history_{}".format(lag)

    def _history_slot_names(self) -> List[str]:
        return [self._history_slot_name(i) for i in range(self.history_size)]

    def _get_history_tensor(self, var):
        """
        Reconstruct history tensor from per-lag slots.

        Output shape:
            var.shape + [history_size]

        Convention:
            history[..., 0] = g_t
            history[..., 1] = g_{t-1}
            ...
            history[..., K-1] = g_{t-K+1}
        """
        history_slots = [self.get_slot(var, name) for name in self._history_slot_names()]
        return tf.stack(history_slots, axis=-1)

    def _assign_history_tensor(self, var, new_history):
        """
        Assign stacked history tensor back into per-lag slots.
        """
        assign_ops = []
        for i, name in enumerate(self._history_slot_names()):
            slot = self.get_slot(var, name)
            assign_ops.append(slot.assign(new_history[..., i], use_locking=self._use_locking))
        return tf.group(*assign_ops)

    def _fractional_gradient_from_history(self, raw_grad, var, history_tensor, nu_t_scalar):
        """
        Update history and compute the variable-order fractional gradient.
        """
        new_history = tf.concat(
            [tf.expand_dims(raw_grad, axis=-1), history_tensor[..., :-1]],
            axis=-1,
        )

        coeffs = self._compute_gl_coefficients_tensor(nu_t_scalar, var.dtype.base_dtype)
        coeffs_bc = self._broadcast_coefficients(coeffs, var)
        frac_grad = tf.reduce_sum(new_history * coeffs_bc, axis=-1)

        return new_history, frac_grad

    def _schedule_progress(self, dtype):
        step = tf.cast(self.iterations + 1, dtype)
        total_steps = tf.cast(self.total_schedule_steps, dtype)
        zero = tf.cast(0.0, dtype)
        one = tf.cast(1.0, dtype)
        return tf.clip_by_value(step / total_steps, zero, one)

    def _schedule_curve(self, progress, dtype):
        one = tf.cast(1.0, dtype)

        if self.schedule_type == "linear":
            s = progress
        elif self.schedule_type == "cosine":
            pi_t = tf.constant(3.141592653589793, dtype=dtype)
            s = 0.5 * (one - tf.cos(pi_t * progress))
        elif self.schedule_type == "exponential":
            gamma = tf.cast(self.schedule_exponential_gamma, dtype)
            numer = one - tf.exp(-gamma * progress)
            denom = one - tf.exp(-gamma)
            denom = tf.maximum(denom, tf.cast(1e-12, dtype))
            s = numer / denom
        else:
            s = progress

        return s

    def _warmup_blend(self, target_nu, dtype):
        """
        Blend from 1.0 to target_nu during early iterations.
        """
        if self.warmup_steps <= 0:
            return target_nu

        one = tf.cast(1.0, dtype)
        warmup_steps_t = tf.cast(self.warmup_steps, dtype)
        step_t = tf.cast(self.iterations + 1, dtype)
        w = tf.clip_by_value(step_t / warmup_steps_t, 0.0, 1.0)
        return one + w * (target_nu - one)

    # ------------------------------------------------------------------
    # Loss injection and hybrid LR control
    # ------------------------------------------------------------------

    def set_current_loss(self, loss_value):
        """
        Store the current scalar loss value.
        In hybrid mode, also update a loss EMA and a conservative internal
        learning-rate multiplier.
        """
        try:
            loss_scalar = float(tf.keras.backend.get_value(tf.cast(loss_value, tf.float32)))
        except Exception:
            loss_scalar = float(loss_value)

        if not self._loss_state_initialized:
            self._current_loss_value = loss_scalar
            self._prev_loss_value = loss_scalar
            self._loss_state_initialized = True
        else:
            self._prev_loss_value = self._current_loss_value
            self._current_loss_value = loss_scalar

        # Hybrid-mode LR adaptation uses a smoothed loss signal, not raw batch differences.
        if self.adaptation_mode == "hybrid_transition":
            if self._loss_ema_value is None:
                self._loss_ema_value = loss_scalar
                self._best_loss_ema = loss_scalar
                self._bad_loss_steps = 0
                self._lr_factor = 1.0
            else:
                self._loss_ema_value = (
                    self.loss_ema_beta * self._loss_ema_value
                    + (1.0 - self.loss_ema_beta) * loss_scalar
                )

                improvement_threshold = self.loss_plateau_tolerance * max(1.0, abs(self._best_loss_ema))
                if self._loss_ema_value < self._best_loss_ema - improvement_threshold:
                    self._best_loss_ema = self._loss_ema_value
                    self._bad_loss_steps = 0
                else:
                    self._bad_loss_steps += 1
                    if self._bad_loss_steps >= self.lr_decay_patience:
                        self._lr_factor = max(
                            self.min_lr_factor,
                            self._lr_factor * self.lr_decay_factor,
                        )
                        self._bad_loss_steps = 0

    # ------------------------------------------------------------------
    # Adaptation rules
    # ------------------------------------------------------------------

    def _order_from_ema_smoothed_gradient_variability(
        self,
        raw_grad,
        prev_raw_grad,
        ema_variability_scalar,
        nu_state_scalar,
        var,
    ):
        dtype = var.dtype.base_dtype
        delta = tf.cast(self.order_delta, dtype)
        alpha = tf.cast(self.order_adapt_rate, dtype)
        gamma = tf.cast(self.ema_smoothing_gamma, dtype)
        order_ema_gamma = tf.cast(self.order_ema_gamma, dtype)
        nu_min, nu_max = self._effective_nu_bounds(dtype)

        diff_norm = self._global_l2_norm(raw_grad - prev_raw_grad, dtype)
        prev_norm = self._global_l2_norm(prev_raw_grad, dtype)
        ratio = diff_norm / (prev_norm + delta)
        ratio = tf.minimum(ratio, tf.cast(self.variability_clip, dtype))

        smoothed_ratio = gamma * ema_variability_scalar + (1.0 - gamma) * ratio

        target_nu = tf.clip_by_value(nu_max - alpha * smoothed_ratio, nu_min, nu_max)
        target_nu = self._warmup_blend(target_nu, dtype)

        new_nu = order_ema_gamma * nu_state_scalar + (1.0 - order_ema_gamma) * target_nu
        new_nu = tf.clip_by_value(new_nu, nu_min, nu_max)

        return new_nu, smoothed_ratio

    def _order_from_schedule(self, nu_state_scalar, var):
        dtype = var.dtype.base_dtype
        order_ema_gamma = tf.cast(self.order_ema_gamma, dtype)
        nu_min, nu_max = self._effective_nu_bounds(dtype)

        progress = self._schedule_progress(dtype)
        s = self._schedule_curve(progress, dtype)

        # Stable start near the classical optimizer, then move toward a safe target.
        start_nu = tf.cast(1.0, dtype)
        final_nu = tf.cast(min(max(self.vderiv_init, self.nu_min), min(self.nu_max, self.stable_nu_cap)), dtype)
        target_nu = start_nu + (final_nu - start_nu) * s
        target_nu = self._warmup_blend(target_nu, dtype)
        target_nu = tf.clip_by_value(target_nu, nu_min, nu_max)

        new_nu = order_ema_gamma * nu_state_scalar + (1.0 - order_ema_gamma) * target_nu
        new_nu = tf.clip_by_value(new_nu, nu_min, nu_max)
        return new_nu

    def _order_from_hybrid_transition(self, nu_state_scalar, var):
        dtype = var.dtype.base_dtype
        order_ema_gamma = tf.cast(self.order_ema_gamma, dtype)
        nu_min, nu_max = self._effective_nu_bounds(dtype)

        progress = self._schedule_progress(dtype)
        s = self._schedule_curve(progress, dtype)

        # Hybrid mode still uses a scheduled order, but allows a bit more range
        # within the stable cap to improve precision later in training.
        start_nu = tf.cast(1.0, dtype)
        final_nu = tf.cast(min(max(self.vderiv_init, self.nu_min), min(self.nu_max, self.stable_nu_cap)), dtype)
        target_nu = start_nu + (final_nu - start_nu) * s
        target_nu = self._warmup_blend(target_nu, dtype)
        target_nu = tf.clip_by_value(target_nu, nu_min, nu_max)

        new_nu = order_ema_gamma * nu_state_scalar + (1.0 - order_ema_gamma) * target_nu
        new_nu = tf.clip_by_value(new_nu, nu_min, nu_max)
        return new_nu

    def _hybrid_fractional_second_moment_mix(self, dtype):
        """
        Scheduled lambda in [0, hybrid_final_fractional_second_moment].
        Early: closer to first-order second moment for speed.
        Late: more fractional contribution for precision.
        """
        progress = self._schedule_progress(dtype)
        s = self._schedule_curve(progress, dtype)
        return tf.cast(self.hybrid_final_fractional_second_moment, dtype) * s

    # ------------------------------------------------------------------
    # Legacy optimizer API
    # ------------------------------------------------------------------

    def _create_slots(self, var_list):
        nu_init = tf.keras.initializers.Constant(self.vderiv_init)

        for var in var_list:
            for slot_name in self._history_slot_names():
                self.add_slot(var, slot_name)

            self.add_slot(var, "prev_raw_grad")
            self.add_slot(var, "nu_state", initializer=nu_init)
            self.add_slot(var, "m")
            self.add_slot(var, "v")

            if self.amsgrad:
                self.add_slot(var, "vhat")

            if self.adaptation_mode == "ema_smoothed_gradient_variability":
                self.add_slot(var, "ema_variability")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype

        base_lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
        lr_factor_t = tf.cast(self._lr_factor, var_dtype)
        lr_t = base_lr_t * lr_factor_t

        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)
        one = tf.cast(1.0, var_dtype)
        step_t = tf.cast(self.iterations + 1, var_dtype)

        raw_grad = self._to_dense_gradient(grad, var)

        history_tensor = self._get_history_tensor(var)
        prev_raw_grad = self.get_slot(var, "prev_raw_grad")
        nu_state = self.get_slot(var, "nu_state")
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        nu_state_scalar = self._scalar_from_slot_tensor(nu_state, var_dtype)

        # --------------------------------------------------------------
        # Step 1: compute the new scalar order
        # --------------------------------------------------------------
        if self.adaptation_mode == "ema_smoothed_gradient_variability":
            ema_state = self.get_slot(var, "ema_variability")
            ema_state_scalar = self._scalar_from_slot_tensor(ema_state, var_dtype)

            new_nu_scalar, new_ema_scalar = self._order_from_ema_smoothed_gradient_variability(
                raw_grad=raw_grad,
                prev_raw_grad=prev_raw_grad,
                ema_variability_scalar=ema_state_scalar,
                nu_state_scalar=nu_state_scalar,
                var=var,
            )
            ema_state_t = ema_state.assign(
                self._filled_like_var(new_ema_scalar, var),
                use_locking=self._use_locking,
            )

        elif self.adaptation_mode == "schedule":
            new_nu_scalar = self._order_from_schedule(
                nu_state_scalar=nu_state_scalar,
                var=var,
            )
            ema_state_t = tf.no_op()

        elif self.adaptation_mode == "hybrid_transition":
            new_nu_scalar = self._order_from_hybrid_transition(
                nu_state_scalar=nu_state_scalar,
                var=var,
            )
            ema_state_t = tf.no_op()

        else:
            raise ValueError("Unsupported adaptation_mode: {}".format(self.adaptation_mode))

        nu_t = nu_state.assign(
            self._filled_like_var(new_nu_scalar, var),
            use_locking=self._use_locking,
        )

        # Update previous raw gradient
        prev_grad_t = prev_raw_grad.assign(raw_grad, use_locking=self._use_locking)

        # --------------------------------------------------------------
        # Step 2: compute fractional-memory gradient
        # --------------------------------------------------------------
        new_history, frac_grad = self._fractional_gradient_from_history(
            raw_grad=raw_grad,
            var=var,
            history_tensor=history_tensor,
            nu_t_scalar=new_nu_scalar,
        )
        history_t = self._assign_history_tensor(var, new_history)

        # --------------------------------------------------------------
        # Step 3: choose second-moment input
        # --------------------------------------------------------------
        if self.adaptation_mode in {"ema_smoothed_gradient_variability", "schedule"}:
            # More conservative and usually faster/stabler:
            # use first-order gradient in the second moment.
            second_moment_grad = raw_grad
        elif self.adaptation_mode == "hybrid_transition":
            lam = self._hybrid_fractional_second_moment_mix(var_dtype)
            second_moment_grad = (
                (one - lam) * raw_grad + lam * frac_grad
            )
        else:
            second_moment_grad = frac_grad

        # --------------------------------------------------------------
        # Step 4: Adam update
        # --------------------------------------------------------------
        new_m = beta_1_t * m + (one - beta_1_t) * frac_grad
        new_v = beta_2_t * v + (one - beta_2_t) * tf.square(second_moment_grad)

        m_t = m.assign(new_m, use_locking=self._use_locking)
        v_t = v.assign(new_v, use_locking=self._use_locking)

        beta_1_power = tf.pow(beta_1_t, step_t)
        beta_2_power = tf.pow(beta_2_t, step_t)

        m_hat = new_m / (one - beta_1_power)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            new_vhat = tf.maximum(vhat, new_v)
            vhat_t = vhat.assign(new_vhat, use_locking=self._use_locking)
            v_hat = new_vhat / (one - beta_2_power)
        else:
            vhat_t = tf.no_op()
            v_hat = new_v / (one - beta_2_power)

        var_t = var.assign_sub(
            lr_t * m_hat / (tf.sqrt(v_hat) + eps_t),
            use_locking=self._use_locking,
        )

        return tf.group(
            var_t,
            m_t,
            v_t,
            vhat_t,
            history_t,
            prev_grad_t,
            nu_t,
            ema_state_t,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self) -> Dict[str, Any]:
        config = super(AdaptiveModesVariableOrderMemoryFAdam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "vderiv_init": self.vderiv_init,
                "nu_min": self.nu_min,
                "nu_max": self.nu_max,
                "history_size": self.history_size,
                "normalize_coefficients": self.normalize_coefficients,
                "classical_fallback_tolerance": self.classical_fallback_tolerance,
                "adaptation_mode": self.adaptation_mode,
                "order_adapt_rate": self.order_adapt_rate,
                "order_delta": self.order_delta,
                "ema_smoothing_gamma": self.ema_smoothing_gamma,
                "schedule_type": self.schedule_type,
                "total_schedule_steps": self.total_schedule_steps,
                "schedule_exponential_gamma": self.schedule_exponential_gamma,
                "stable_nu_cap": self.stable_nu_cap,
                "variability_clip": self.variability_clip,
                "order_ema_gamma": self.order_ema_gamma,
                "warmup_steps": self.warmup_steps,
                "hybrid_final_fractional_second_moment": self.hybrid_final_fractional_second_moment,
                "loss_ema_beta": self.loss_ema_beta,
                "lr_decay_patience": self.lr_decay_patience,
                "lr_decay_factor": self.lr_decay_factor,
                "min_lr_factor": self.min_lr_factor,
                "loss_plateau_tolerance": self.loss_plateau_tolerance,
            }
        )
        return config

    # ------------------------------------------------------------------
    # Convenience inspection helpers
    # ------------------------------------------------------------------

    def current_orders(self) -> List[tf.Tensor]:
        """
        Return current per-variable order states as scalar tensors.
        """
        orders = []
        for w in self.weights:
            if "nu_state" in w.name:
                orders.append(tf.reduce_mean(tf.identity(w)))
        return orders

    def current_mode(self) -> str:
        return self.adaptation_mode

    def current_lr_factor(self) -> float:
        return float(self._lr_factor)


__all__ = ["AdaptiveModesVariableOrderMemoryFAdam"]


if __name__ == "__main__":
    tf.random.set_seed(42)

    x = tf.random.normal((64, 10))
    y = tf.random.uniform((64,), minval=0, maxval=3, dtype=tf.int32)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    optimizer = AdaptiveModesVariableOrderMemoryFAdam(
        lr=1e-3,
        adaptation_mode="hybrid_transition",
        vderiv_init=0.85,
        nu_min=0.6,
        nu_max=1.2,
        history_size=6,
        order_adapt_rate=0.05,
        ema_smoothing_gamma=0.95,
        schedule_type="cosine",
        total_schedule_steps=1000,
        normalize_coefficients=False,
        stable_nu_cap=1.0,
        variability_clip=10.0,
        order_ema_gamma=0.90,
        warmup_steps=50,
        hybrid_final_fractional_second_moment=1.0,
        loss_ema_beta=0.95,
        lr_decay_patience=50,
        lr_decay_factor=0.7,
        min_lr_factor=0.2,
        loss_plateau_tolerance=1e-4,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x, y, epochs=1, batch_size=16, verbose=1)

    print("Current per-variable fractional orders:")
    for nu in optimizer.current_orders():
        try:
            print(nu.numpy())
        except Exception:
            print(nu)

    print("Current LR factor:", optimizer.current_lr_factor())