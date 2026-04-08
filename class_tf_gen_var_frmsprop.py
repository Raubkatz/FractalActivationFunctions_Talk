"""
adaptive_randomized_variable_order_memory_frmsprop_legacy.py

Legacy TensorFlow / Keras implementation of a variable-order, memory-based
fractional RMSprop optimizer.

Main idea
---------
This optimizer keeps the explicit short-memory fractional gradient construction,
but uses RMSprop instead of Adam.

The fractional order is adapted conservatively:
1. Start near an initial fractional order vderiv_init.
2. At each step, sample a small random perturbation around the current order.
3. Evaluate whether this perturbed order improves gradient stability.
4. Move only a small step toward the more stable order.
5. Keep the order inside a safe interval.

This is intentionally simpler and more conservative than the previous adaptive
fractional Adam design.

Compatibility notes
-------------------
- This file targets older tf.keras optimizer_v2 style APIs.
- It accepts both `learning_rate=` and `lr=`.
- It preserves `set_current_loss(...)` and `current_orders()` for compatibility,
  although loss is not required for the main adaptation logic.

Important parameter interpretation
----------------------------------
In this version:
- `nu_min` is interpreted as an offset below `vderiv_init`
- `nu_max` is interpreted as an offset above `vderiv_init`

So the effective bounds are:
    effective_nu_min = max(0.05, vderiv_init - nu_min)
    effective_nu_max = vderiv_init + nu_max

There is no upper clipping by `stable_nu_cap` anymore. The parameter is kept only
for API compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List

import tensorflow as tf


class AdaptiveRandomVariableOrderMemoryFRMSprop(tf.keras.optimizers.Optimizer):
    """
    Legacy-API variable-order memory-based fractional RMSprop.

    Core design:
        - explicit fractional short-memory gradient
        - RMSprop variance normalization
        - small random exploration around current order
        - small update toward more stable gradient behavior
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        lr: float = None,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        vderiv_init: float = 0.9,
        nu_min: float = 0.6,
        nu_max: float = 1.5,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        classical_fallback_tolerance: float = 1e-8,
        order_delta: float = 1e-8,
        random_perturb_scale: float = 0.03,
        random_decay: float = 0.999,
        stability_step_size: float = 0.01,
        stability_ema_gamma: float = 0.95,
        stable_nu_cap: float = 1.0,
        warmup_steps: int = 50,
        use_signed_random_search: bool = True,
        name: str = "AdaptiveRandomVariableOrderMemoryFRMSprop",
        **kwargs: Any,
    ):
        super(AdaptiveRandomVariableOrderMemoryFRMSprop, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        if vderiv_init <= 0.0:
            raise ValueError("`vderiv_init` must be > 0.")
        if nu_min < 0.0:
            raise ValueError("`nu_min` must be >= 0. It is interpreted as an offset below `vderiv_init`.")
        if nu_max < 0.0:
            raise ValueError("`nu_max` must be >= 0. It is interpreted as an offset above `vderiv_init`.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if classical_fallback_tolerance <= 0.0:
            raise ValueError("`classical_fallback_tolerance` must be > 0.")
        if order_delta <= 0.0:
            raise ValueError("`order_delta` must be > 0.")
        if random_perturb_scale < 0.0:
            raise ValueError("`random_perturb_scale` must be >= 0.")
        if not (0.0 < random_decay <= 1.0):
            raise ValueError("`random_decay` must satisfy 0 < random_decay <= 1.")
        if stability_step_size < 0.0:
            raise ValueError("`stability_step_size` must be >= 0.")
        if not (0.0 <= stability_ema_gamma < 1.0):
            raise ValueError("`stability_ema_gamma` must satisfy 0 <= stability_ema_gamma < 1.")
        if stable_nu_cap <= 0.0:
            raise ValueError("`stable_nu_cap` must be > 0.")
        if warmup_steps < 0:
            raise ValueError("`warmup_steps` must be >= 0.")

        self.rho = float(rho)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.centered = bool(centered)

        self.vderiv_init = float(vderiv_init)
        self.nu_min = float(nu_min)  # offset below vderiv_init
        self.nu_max = float(nu_max)  # offset above vderiv_init
        self.history_size = int(history_size)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.classical_fallback_tolerance = float(classical_fallback_tolerance)

        self.order_delta = float(order_delta)
        self.random_perturb_scale = float(random_perturb_scale)
        self.random_decay = float(random_decay)
        self.stability_step_size = float(stability_step_size)
        self.stability_ema_gamma = float(stability_ema_gamma)
        self.stable_nu_cap = float(stable_nu_cap)  # kept only for API compatibility
        self.warmup_steps = int(warmup_steps)
        self.use_signed_random_search = bool(use_signed_random_search)

        # kept only for API compatibility if training loop calls this
        self._current_loss_value = None
        self._prev_loss_value = None
        self._loss_state_initialized = False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _to_dense_gradient(self, grad, var):
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                indices=tf.expand_dims(grad.indices, axis=1),
                updates=grad.values,
                shape=tf.shape(var),
            )
        return grad

    def _scalar_from_slot_tensor(self, slot_tensor, dtype):
        return tf.cast(tf.reduce_mean(slot_tensor), dtype)

    def _filled_like_var(self, scalar_value, var):
        scalar_value = tf.cast(scalar_value, var.dtype.base_dtype)
        return tf.ones_like(var, dtype=var.dtype.base_dtype) * scalar_value

    def _effective_nu_bounds(self, dtype):
        effective_nu_min = max(0.05, self.vderiv_init - self.nu_min)
        effective_nu_max = self.vderiv_init + self.nu_max

        nu_min = tf.cast(effective_nu_min, dtype)
        nu_max = tf.cast(effective_nu_max, dtype)
        nu_max = tf.maximum(nu_max, nu_min)
        return nu_min, nu_max

    def _global_l2_norm(self, x, dtype):
        x = tf.cast(x, dtype)
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    def _compute_gl_coefficients_tensor(self, nu, dtype):
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
        history_slots = [self.get_slot(var, name) for name in self._history_slot_names()]
        return tf.stack(history_slots, axis=-1)

    def _assign_history_tensor(self, var, new_history):
        assign_ops = []
        for i, name in enumerate(self._history_slot_names()):
            slot = self.get_slot(var, name)
            assign_ops.append(slot.assign(new_history[..., i], use_locking=self._use_locking))
        return tf.group(*assign_ops)

    def _fractional_gradient_from_history(self, raw_grad, var, history_tensor, nu_t_scalar):
        new_history = tf.concat(
            [tf.expand_dims(raw_grad, axis=-1), history_tensor[..., :-1]],
            axis=-1,
        )

        coeffs = self._compute_gl_coefficients_tensor(nu_t_scalar, var.dtype.base_dtype)
        coeffs_bc = self._broadcast_coefficients(coeffs, var)
        frac_grad = tf.reduce_sum(new_history * coeffs_bc, axis=-1)

        return new_history, frac_grad

    def _warmup_blend(self, target_nu, dtype):
        if self.warmup_steps <= 0:
            return target_nu

        one = tf.cast(1.0, dtype)
        warmup_steps_t = tf.cast(self.warmup_steps, dtype)
        step_t = tf.cast(self.iterations + 1, dtype)
        w = tf.clip_by_value(step_t / warmup_steps_t, 0.0, 1.0)
        return one + w * (target_nu - one)

    def _current_random_scale(self, dtype):
        step_t = tf.cast(self.iterations + 1, dtype)
        base = tf.cast(self.random_perturb_scale, dtype)
        decay = tf.cast(self.random_decay, dtype)
        return base * tf.pow(decay, step_t - 1.0)

    def _stability_score(self, raw_grad, prev_raw_grad, dtype):
        diff_norm = self._global_l2_norm(raw_grad - prev_raw_grad, dtype)
        prev_norm = self._global_l2_norm(prev_raw_grad, dtype)
        return diff_norm / (prev_norm + tf.cast(self.order_delta, dtype))

    def _sample_candidate_order(self, nu_state_scalar, dtype):
        nu_min, nu_max = self._effective_nu_bounds(dtype)

        scale = self._current_random_scale(dtype)
        if self.use_signed_random_search:
            sign = tf.where(
                tf.random.uniform([], 0.0, 1.0, dtype=dtype) < 0.5,
                tf.cast(-1.0, dtype),
                tf.cast(1.0, dtype),
            )
            perturb = sign * scale
        else:
            perturb = tf.random.normal([], mean=0.0, stddev=scale, dtype=dtype)

        candidate = nu_state_scalar + perturb
        candidate = self._warmup_blend(candidate, dtype)
        candidate = tf.clip_by_value(candidate, nu_min, nu_max)
        return candidate

    def _propose_new_order(
        self,
        raw_grad,
        prev_raw_grad,
        history_tensor,
        stability_state_scalar,
        nu_state_scalar,
        var,
    ):
        dtype = var.dtype.base_dtype
        gamma = tf.cast(self.stability_ema_gamma, dtype)
        step_size = tf.cast(self.stability_step_size, dtype)
        nu_min, nu_max = self._effective_nu_bounds(dtype)

        baseline_score = self._stability_score(raw_grad, prev_raw_grad, dtype)
        baseline_smoothed = gamma * stability_state_scalar + (1.0 - gamma) * baseline_score

        candidate_nu = self._sample_candidate_order(nu_state_scalar, dtype)

        _, frac_grad_current = self._fractional_gradient_from_history(
            raw_grad=raw_grad,
            var=var,
            history_tensor=history_tensor,
            nu_t_scalar=nu_state_scalar,
        )
        _, frac_grad_candidate = self._fractional_gradient_from_history(
            raw_grad=raw_grad,
            var=var,
            history_tensor=history_tensor,
            nu_t_scalar=candidate_nu,
        )

        current_frac_stability = self._global_l2_norm(frac_grad_current - prev_raw_grad, dtype) / (
            self._global_l2_norm(prev_raw_grad, dtype) + tf.cast(self.order_delta, dtype)
        )
        candidate_frac_stability = self._global_l2_norm(frac_grad_candidate - prev_raw_grad, dtype) / (
            self._global_l2_norm(prev_raw_grad, dtype) + tf.cast(self.order_delta, dtype)
        )

        better_direction = tf.where(
            candidate_frac_stability < current_frac_stability,
            tf.cast(1.0, dtype),
            tf.cast(-1.0, dtype),
        )

        target_nu = nu_state_scalar + step_size * better_direction * (candidate_nu - nu_state_scalar)
        target_nu = tf.clip_by_value(target_nu, nu_min, nu_max)

        new_nu = gamma * nu_state_scalar + (1.0 - gamma) * target_nu
        new_nu = tf.clip_by_value(new_nu, nu_min, nu_max)

        new_stability_state = baseline_smoothed

        return new_nu, new_stability_state, candidate_nu, current_frac_stability, candidate_frac_stability

    # ------------------------------------------------------------------
    # Compatibility helper
    # ------------------------------------------------------------------

    def set_current_loss(self, loss_value):
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
    # Legacy optimizer API
    # ------------------------------------------------------------------

    def _create_slots(self, var_list):
        nu_init = tf.keras.initializers.Constant(self.vderiv_init)
        stability_init = tf.keras.initializers.Constant(0.0)

        for var in var_list:
            for slot_name in self._history_slot_names():
                self.add_slot(var, slot_name)

            self.add_slot(var, "prev_raw_grad")
            self.add_slot(var, "nu_state", initializer=nu_init)
            self.add_slot(var, "stability_state", initializer=stability_init)
            self.add_slot(var, "velocity")

            if self.momentum > 0.0:
                self.add_slot(var, "momentum")

            if self.centered:
                self.add_slot(var, "avg_grad")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
        rho_t = tf.cast(self.rho, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)
        one = tf.cast(1.0, var_dtype)

        raw_grad = self._to_dense_gradient(grad, var)

        history_tensor = self._get_history_tensor(var)
        prev_raw_grad = self.get_slot(var, "prev_raw_grad")
        nu_state = self.get_slot(var, "nu_state")
        stability_state = self.get_slot(var, "stability_state")
        velocity = self.get_slot(var, "velocity")

        nu_state_scalar = self._scalar_from_slot_tensor(nu_state, var_dtype)
        stability_state_scalar = self._scalar_from_slot_tensor(stability_state, var_dtype)

        new_nu_scalar, new_stability_scalar, _, _, _ = self._propose_new_order(
            raw_grad=raw_grad,
            prev_raw_grad=prev_raw_grad,
            history_tensor=history_tensor,
            stability_state_scalar=stability_state_scalar,
            nu_state_scalar=nu_state_scalar,
            var=var,
        )

        nu_t = nu_state.assign(
            self._filled_like_var(new_nu_scalar, var),
            use_locking=self._use_locking,
        )

        stability_t = stability_state.assign(
            self._filled_like_var(new_stability_scalar, var),
            use_locking=self._use_locking,
        )

        prev_grad_t = prev_raw_grad.assign(raw_grad, use_locking=self._use_locking)

        new_history, frac_grad = self._fractional_gradient_from_history(
            raw_grad=raw_grad,
            var=var,
            history_tensor=history_tensor,
            nu_t_scalar=new_nu_scalar,
        )
        history_t = self._assign_history_tensor(var, new_history)

        new_velocity = rho_t * velocity + (one - rho_t) * tf.square(frac_grad)
        velocity_t = velocity.assign(new_velocity, use_locking=self._use_locking)

        if self.centered:
            avg_grad = self.get_slot(var, "avg_grad")
            new_avg_grad = rho_t * avg_grad + (one - rho_t) * frac_grad
            avg_grad_t = avg_grad.assign(new_avg_grad, use_locking=self._use_locking)
            denom = tf.sqrt(new_velocity - tf.square(new_avg_grad) + eps_t)
        else:
            avg_grad_t = tf.no_op()
            denom = tf.sqrt(new_velocity + eps_t)

        if self.momentum > 0.0:
            momentum_coeff = tf.cast(self.momentum, var_dtype)
            mom = self.get_slot(var, "momentum")
            new_mom = momentum_coeff * mom + lr_t * frac_grad / denom
            mom_t = mom.assign(new_mom, use_locking=self._use_locking)
            var_t = var.assign_sub(new_mom, use_locking=self._use_locking)
            return tf.group(
                var_t,
                nu_t,
                stability_t,
                prev_grad_t,
                history_t,
                velocity_t,
                avg_grad_t,
                mom_t,
            )

        var_t = var.assign_sub(lr_t * frac_grad / denom, use_locking=self._use_locking)
        return tf.group(
            var_t,
            nu_t,
            stability_t,
            prev_grad_t,
            history_t,
            velocity_t,
            avg_grad_t,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self) -> Dict[str, Any]:
        config = super(AdaptiveRandomVariableOrderMemoryFRMSprop, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                "vderiv_init": self.vderiv_init,
                "nu_min": self.nu_min,
                "nu_max": self.nu_max,
                "history_size": self.history_size,
                "normalize_coefficients": self.normalize_coefficients,
                "classical_fallback_tolerance": self.classical_fallback_tolerance,
                "order_delta": self.order_delta,
                "random_perturb_scale": self.random_perturb_scale,
                "random_decay": self.random_decay,
                "stability_step_size": self.stability_step_size,
                "stability_ema_gamma": self.stability_ema_gamma,
                "stable_nu_cap": self.stable_nu_cap,
                "warmup_steps": self.warmup_steps,
                "use_signed_random_search": self.use_signed_random_search,
            }
        )
        return config

    # ------------------------------------------------------------------
    # Convenience inspection helpers
    # ------------------------------------------------------------------

    def current_orders(self) -> List[tf.Tensor]:
        orders = []
        for w in self.weights:
            if "nu_state" in w.name:
                orders.append(tf.reduce_mean(tf.identity(w)))
        return orders

    def current_stability_states(self) -> List[tf.Tensor]:
        states = []
        for w in self.weights:
            if "stability_state" in w.name:
                states.append(tf.reduce_mean(tf.identity(w)))
        return states


__all__ = ["AdaptiveRandomVariableOrderMemoryFRMSprop"]


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

    optimizer = AdaptiveRandomVariableOrderMemoryFRMSprop(
        lr=1e-3,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-7,
        centered=False,
        vderiv_init=0.85,
        nu_min=0.25,
        nu_max=1.5,
        history_size=6,
        normalize_coefficients=False,
        order_delta=1e-8,
        random_perturb_scale=0.03,
        random_decay=0.999,
        stability_step_size=0.01,
        stability_ema_gamma=0.95,
        stable_nu_cap=1.0,
        warmup_steps=50,
        use_signed_random_search=True,
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

    print("Current per-variable stability states:")
    for st in optimizer.current_stability_states():
        try:
            print(st.numpy())
        except Exception:
            print(st)