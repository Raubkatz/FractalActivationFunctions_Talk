"""
adaptive_modes_variable_order_memory_fadam_legacy.py

Legacy TensorFlow / Keras implementation of a variable-order, memory-based
fractional Adam optimizer with multiple order-adaptation modes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import tensorflow as tf


class AdaptiveModesVariableOrderMemoryFAdam(tf.keras.optimizers.Optimizer):
    """
    Legacy-API variable-order memory-based fractional Adam.

    Supported adaptation modes:
        - "gradient_variability"
        - "ema_smoothed_gradient_variability"
        - "schedule"
        - "loss_improvement"
    """

    _VALID_ADAPTATION_MODES = {
        "gradient_variability",
        "ema_smoothed_gradient_variability",
        "schedule",
        "loss_improvement",
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
        adaptation_mode: str = "gradient_variability",
        order_adapt_rate: float = 0.1,
        order_delta: float = 1e-8,
        ema_smoothing_gamma: float = 0.9,
        schedule_type: str = "linear",
        total_schedule_steps: int = 10000,
        schedule_exponential_gamma: float = 4.0,
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

        # Loss states for loss_improvement mode
        self._current_loss_value = None
        self._prev_loss_value = None
        self._loss_state_initialized = False

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

    # ------------------------------------------------------------------
    # Loss injection for loss_improvement mode
    # ------------------------------------------------------------------

    def set_current_loss(self, loss_value):
        """
        Store the current scalar loss value for `loss_improvement` mode.
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

    # ------------------------------------------------------------------
    # Adaptation rules
    # ------------------------------------------------------------------

    def _order_from_gradient_variability(self, raw_grad, prev_raw_grad, nu_state_scalar, var):
        dtype = var.dtype.base_dtype
        delta = tf.cast(self.order_delta, dtype)
        alpha = tf.cast(self.order_adapt_rate, dtype)
        nu_min = tf.cast(self.nu_min, dtype)
        nu_max = tf.cast(self.nu_max, dtype)

        diff_norm = tf.norm(raw_grad - prev_raw_grad, ord="euclidean")
        prev_norm = tf.norm(prev_raw_grad, ord="euclidean")
        ratio = diff_norm / (prev_norm + delta)

        new_nu = tf.clip_by_value(nu_max - alpha * ratio, nu_min, nu_max)
        return new_nu

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
        nu_min = tf.cast(self.nu_min, dtype)
        nu_max = tf.cast(self.nu_max, dtype)

        diff_norm = tf.norm(raw_grad - prev_raw_grad, ord="euclidean")
        prev_norm = tf.norm(prev_raw_grad, ord="euclidean")
        ratio = diff_norm / (prev_norm + delta)

        smoothed_ratio = gamma * ema_variability_scalar + (1.0 - gamma) * ratio
        new_nu = tf.clip_by_value(nu_max - alpha * smoothed_ratio, nu_min, nu_max)

        return new_nu, smoothed_ratio

    def _order_from_schedule(self, nu_state_scalar, var):
        dtype = var.dtype.base_dtype
        step = tf.cast(self.iterations + 1, dtype)
        total_steps = tf.cast(self.total_schedule_steps, dtype)
        zero = tf.cast(0.0, dtype)
        one = tf.cast(1.0, dtype)

        progress = tf.clip_by_value(step / total_steps, zero, one)

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

        nu_min = tf.cast(self.nu_min, dtype)
        nu_max = tf.cast(self.nu_max, dtype)
        new_nu = nu_min + (nu_max - nu_min) * s
        return new_nu

    def _order_from_loss_improvement(self, nu_state_scalar, var):
        dtype = var.dtype.base_dtype
        nu_min = tf.cast(self.nu_min, dtype)
        nu_max = tf.cast(self.nu_max, dtype)
        alpha = tf.cast(self.order_adapt_rate, dtype)
        delta = tf.cast(self.order_delta, dtype)

        if (self._current_loss_value is None) or (self._prev_loss_value is None) or (not self._loss_state_initialized):
            return tf.cast(nu_state_scalar, dtype)

        current_loss = tf.cast(self._current_loss_value, dtype)
        prev_loss = tf.cast(self._prev_loss_value, dtype)

        improvement = tf.abs(prev_loss - current_loss) / (tf.abs(prev_loss) + delta)
        new_nu = tf.clip_by_value(nu_min + alpha * improvement, nu_min, nu_max)
        return new_nu

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
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
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

        # Step 1: compute variable order as scalar, then broadcast to slot shape
        if self.adaptation_mode == "gradient_variability":
            new_nu_scalar = self._order_from_gradient_variability(
                raw_grad=raw_grad,
                prev_raw_grad=prev_raw_grad,
                nu_state_scalar=nu_state_scalar,
                var=var,
            )
            ema_state_t = tf.no_op()

        elif self.adaptation_mode == "ema_smoothed_gradient_variability":
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

        elif self.adaptation_mode == "loss_improvement":
            new_nu_scalar = self._order_from_loss_improvement(
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

        # Step 2: compute fractional-memory gradient using scalar order
        new_history, frac_grad = self._fractional_gradient_from_history(
            raw_grad=raw_grad,
            var=var,
            history_tensor=history_tensor,
            nu_t_scalar=new_nu_scalar,
        )
        history_t = self._assign_history_tensor(var, new_history)

        # Step 3: Adam update
        new_m = beta_1_t * m + (one - beta_1_t) * frac_grad
        new_v = beta_2_t * v + (one - beta_2_t) * tf.square(frac_grad)

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
            }
        )
        return config

    # ------------------------------------------------------------------
    # Convenience inspection helpers
    # ------------------------------------------------------------------

    def current_orders(self) -> List[tf.Tensor]:
        """
        Return current per-variable order states.

        This helper only works after slots have been created.
        """
        orders = []
        for w in self.weights:
            if "nu_state" in w.name:
                orders.append(tf.identity(w))
        return orders

    def current_mode(self) -> str:
        return self.adaptation_mode


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
        adaptation_mode="ema_smoothed_gradient_variability",
        vderiv_init=0.9,
        nu_min=0.4,
        nu_max=1.6,
        history_size=6,
        order_adapt_rate=0.1,
        ema_smoothing_gamma=0.9,
        normalize_coefficients=False,
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