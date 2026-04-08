"""
memory_fractional_optimizers_legacy.py

Legacy TensorFlow / Keras implementations of explicit memory-based fractional
optimizers.

Purpose
-------
This file is adapted for older TensorFlow / Keras environments that still use
the `optimizer_v2` API style, where custom optimizers typically:
    - subclass `tf.keras.optimizers.Optimizer`,
    - use `_set_hyper(...)` / `_get_hyper(...)`,
    - implement `_create_slots(...)`,
    - implement `_resource_apply_dense(...)` and `_resource_apply_sparse(...)`,
    - accept `lr` as the legacy learning-rate argument.

This file provides the same conceptual optimizer family as the modern version,
but rewritten to match the older API expected by environments such as the one
shown in the traceback:

    Allowed kwargs are {'clipvalue', 'clipnorm', 'global_clipnorm', 'decay', 'lr'}.

Main idea
---------
These optimizers implement an explicit short-memory fractional-gradient
construction. Unlike Herrera-style fractional optimizers, which rescale only the
current gradient, the optimizers in this file store a history of past gradients
and combine them with fractional coefficients.

For a parameter tensor w and gradient sequence {g_t}, the effective
fractional-memory gradient is

    g_t^(nu) = sum_{k=0}^{K-1} c_k^(nu) * g_{t-k},

where
    - nu is the fractional order (`vderiv`),
    - K is the history length (`history_size`),
    - c_k^(nu) are short-memory Grünwald-Letnikov-style coefficients.

Implemented optimizers
----------------------
    - MemoryFSGD
    - MemoryFRMSprop
    - MemoryFAdam
    - MemoryFAdadelta
"""

from __future__ import annotations

from typing import Any, Dict, List

import tensorflow as tf


# =============================================================================
# Shared helper base
# =============================================================================


class _MemoryFractionalBase:
    """
    Shared helper functionality for explicit memory-based fractional optimizers.
    """

    def _init_memory_fractional_args(
        self,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
    ) -> None:
        if not isinstance(vderiv, (float, int)):
            raise TypeError("`vderiv` must be a float.")
        if not (0.0 < float(vderiv) < 2.0):
            raise ValueError("`vderiv` must satisfy 0 < vderiv < 2.")
        if not isinstance(history_size, int) or history_size < 1:
            raise ValueError("`history_size` must be an integer >= 1.")
        if coeff_epsilon <= 0.0:
            raise ValueError("`coeff_epsilon` must be > 0.")

        self.vderiv = float(vderiv)
        self.history_size = int(history_size)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.coeff_epsilon = float(coeff_epsilon)

        self._fractional_coefficients = self._compute_fractional_coefficients(
            self.vderiv,
            self.history_size,
            self.normalize_coefficients,
        )

    def _compute_fractional_coefficients(
        self,
        alpha: float,
        n_terms: int,
        normalize: bool = False,
    ) -> List[float]:
        """
        Compute short-memory Grünwald-Letnikov-style coefficients.

        c_0 = 1
        c_k = (1 - (alpha + 1)/k) * c_{k-1},  k >= 1

        Practical classical-limit convention:
        If alpha is numerically equal to 1, use [1, 0, ..., 0].
        """
        if abs(alpha - 1.0) < 1e-12:
            coeffs = [1.0] + [0.0] * (n_terms - 1)
            return coeffs

        coeffs = [1.0]
        for k in range(1, n_terms):
            coeff = (1.0 - (alpha + 1.0) / k) * coeffs[k - 1]
            coeffs.append(coeff)

        if normalize:
            denom = sum(abs(c) for c in coeffs)
            if denom < self.coeff_epsilon:
                denom = 1.0
            coeffs = [c / denom for c in coeffs]

        return coeffs

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

    def _coefficients_tensor(self, dtype):
        """
        Return coefficient vector as a tensor of shape [history_size].
        """
        return tf.constant(self._fractional_coefficients, dtype=dtype)

    def _make_history_slot_name(self, idx: int) -> str:
        return f"grad_history_{idx}"

    def _history_slot_names(self) -> List[str]:
        return [self._make_history_slot_name(i) for i in range(self.history_size)]

    def _get_history_tensor(self, var):
        """
        Reconstruct stacked history tensor from per-lag slots.

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

    def _fractional_gradient_from_history(self, grad, var, history_tensor):
        """
        Update history and compute the effective fractional-memory gradient.
        """
        grad = self._to_dense_gradient(grad, var)
        var_dtype = var.dtype.base_dtype

        new_history = tf.concat(
            [tf.expand_dims(grad, axis=-1), history_tensor[..., :-1]],
            axis=-1,
        )

        if abs(self.vderiv - 1.0) < 1e-12 or self.history_size == 1:
            frac_grad = grad
        else:
            coeffs = self._coefficients_tensor(var_dtype)
            reshape_shape = [1] * len(var.shape) + [self.history_size]
            coeffs = tf.reshape(coeffs, reshape_shape)
            frac_grad = tf.reduce_sum(new_history * coeffs, axis=-1)

        return new_history, frac_grad

    def _memory_fractional_config(self) -> Dict[str, Any]:
        return {
            "vderiv": self.vderiv,
            "history_size": self.history_size,
            "normalize_coefficients": self.normalize_coefficients,
            "coeff_epsilon": self.coeff_epsilon,
        }


# =============================================================================
# MemoryFSGD
# =============================================================================


class MemoryFSGD(_MemoryFractionalBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API explicit memory-based fractional SGD.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        name: str = "MemoryFSGD",
        **kwargs: Any,
    ):
        super(MemoryFSGD, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
        )

    def _create_slots(self, var_list):
        for var in var_list:
            for slot_name in self._history_slot_names():
                self.add_slot(var, slot_name)
            if self.momentum > 0.0:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)

        history_tensor = self._get_history_tensor(var)
        new_history, frac_grad = self._fractional_gradient_from_history(grad, var, history_tensor)
        history_t = self._assign_history_tensor(var, new_history)

        if self.momentum > 0.0:
            momentum_coeff = tf.cast(self.momentum, var_dtype)
            m = self.get_slot(var, "momentum")

            new_m = momentum_coeff * m - lr_t * frac_grad
            m_t = m.assign(new_m, use_locking=self._use_locking)

            if self.nesterov:
                var_t = var.assign_add(
                    momentum_coeff * new_m - lr_t * frac_grad,
                    use_locking=self._use_locking,
                )
            else:
                var_t = var.assign_add(new_m, use_locking=self._use_locking)

            return tf.group(var_t, m_t, history_t)

        var_t = var.assign_sub(lr_t * frac_grad, use_locking=self._use_locking)
        return tf.group(var_t, history_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(MemoryFSGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFRMSprop
# =============================================================================


class MemoryFRMSprop(_MemoryFractionalBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API explicit memory-based fractional RMSprop.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        name: str = "MemoryFRMSprop",
        **kwargs: Any,
    ):
        super(MemoryFRMSprop, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.centered = bool(centered)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
        )

    def _create_slots(self, var_list):
        for var in var_list:
            for slot_name in self._history_slot_names():
                self.add_slot(var, slot_name)
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

        history_tensor = self._get_history_tensor(var)
        velocity = self.get_slot(var, "velocity")

        new_history, frac_grad = self._fractional_gradient_from_history(grad, var, history_tensor)
        history_t = self._assign_history_tensor(var, new_history)

        new_velocity = rho_t * velocity + (1.0 - rho_t) * tf.square(frac_grad)
        velocity_t = velocity.assign(new_velocity, use_locking=self._use_locking)

        if self.centered:
            avg_grad = self.get_slot(var, "avg_grad")
            new_avg_grad = rho_t * avg_grad + (1.0 - rho_t) * frac_grad
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
            return tf.group(var_t, history_t, velocity_t, avg_grad_t, mom_t)

        var_t = var.assign_sub(lr_t * frac_grad / denom, use_locking=self._use_locking)
        return tf.group(var_t, history_t, velocity_t, avg_grad_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(MemoryFRMSprop, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdam
# =============================================================================


class MemoryFAdam(_MemoryFractionalBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API explicit memory-based fractional Adam.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        name: str = "MemoryFAdam",
        **kwargs: Any,
    ):
        super(MemoryFAdam, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
        )

    def _create_slots(self, var_list):
        for var in var_list:
            for slot_name in self._history_slot_names():
                self.add_slot(var, slot_name)
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)
        step_t = tf.cast(self.iterations + 1, var_dtype)

        history_tensor = self._get_history_tensor(var)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        new_history, frac_grad = self._fractional_gradient_from_history(grad, var, history_tensor)
        history_t = self._assign_history_tensor(var, new_history)

        new_m = beta_1_t * m + (1.0 - beta_1_t) * frac_grad
        new_v = beta_2_t * v + (1.0 - beta_2_t) * tf.square(frac_grad)

        m_t = m.assign(new_m, use_locking=self._use_locking)
        v_t = v.assign(new_v, use_locking=self._use_locking)

        beta_1_power = tf.pow(beta_1_t, step_t)
        beta_2_power = tf.pow(beta_2_t, step_t)

        m_hat = new_m / (1.0 - beta_1_power)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            new_vhat = tf.maximum(vhat, new_v)
            vhat_t = vhat.assign(new_vhat, use_locking=self._use_locking)
            v_hat = new_vhat / (1.0 - beta_2_power)
        else:
            vhat_t = tf.no_op()
            v_hat = new_v / (1.0 - beta_2_power)

        var_t = var.assign_sub(
            lr_t * m_hat / (tf.sqrt(v_hat) + eps_t),
            use_locking=self._use_locking,
        )

        return tf.group(var_t, history_t, m_t, v_t, vhat_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(MemoryFAdam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# MemoryFAdadelta
# =============================================================================


class MemoryFAdadelta(_MemoryFractionalBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API explicit memory-based fractional Adadelta.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        lr: float = None,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        history_size: int = 8,
        normalize_coefficients: bool = False,
        coeff_epsilon: float = 1e-12,
        name: str = "MemoryFAdadelta",
        **kwargs: Any,
    ):
        super(MemoryFAdadelta, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.epsilon = float(epsilon)

        self._init_memory_fractional_args(
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=normalize_coefficients,
            coeff_epsilon=coeff_epsilon,
        )

    def _create_slots(self, var_list):
        for var in var_list:
            for slot_name in self._history_slot_names():
                self.add_slot(var, slot_name)
            self.add_slot(var, "accumulated_grad")
            self.add_slot(var, "accumulated_delta")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
        rho_t = tf.cast(self.rho, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)

        history_tensor = self._get_history_tensor(var)
        acc_g = self.get_slot(var, "accumulated_grad")
        acc_d = self.get_slot(var, "accumulated_delta")

        new_history, frac_grad = self._fractional_gradient_from_history(grad, var, history_tensor)
        history_t = self._assign_history_tensor(var, new_history)

        new_acc_g = rho_t * acc_g + (1.0 - rho_t) * tf.square(frac_grad)
        acc_g_t = acc_g.assign(new_acc_g, use_locking=self._use_locking)

        rms_delta = tf.sqrt(acc_d + eps_t)
        rms_grad = tf.sqrt(new_acc_g + eps_t)
        delta = -(rms_delta / rms_grad) * frac_grad

        new_acc_d = rho_t * acc_d + (1.0 - rho_t) * tf.square(delta)
        acc_d_t = acc_d.assign(new_acc_d, use_locking=self._use_locking)

        var_t = var.assign_add(lr_t * delta, use_locking=self._use_locking)

        return tf.group(var_t, history_t, acc_g_t, acc_d_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(MemoryFAdadelta, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "epsilon": self.epsilon,
                **self._memory_fractional_config(),
            }
        )
        return config


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    "MemoryFSGD",
    "MemoryFRMSprop",
    "MemoryFAdam",
    "MemoryFAdadelta",
]


# =============================================================================
# Minimal smoke test
# =============================================================================

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

    optimizer = MemoryFAdam(
        lr=1e-3,
        vderiv=0.8,
        history_size=6,
        normalize_coefficients=False,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x, y, epochs=1, batch_size=16, verbose=1)