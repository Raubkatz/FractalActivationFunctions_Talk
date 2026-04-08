"""
fractional_herrera_optimizers_legacy.py

Legacy TensorFlow / Keras implementations of Herrera-style fractional optimizers.

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

    Allowed kwargs are {'decay', 'clipvalue', 'clipnorm', 'global_clipnorm', 'lr'}.

Main idea
---------
These optimizers follow the construction introduced by Oscar Herrera-Alcántara
and collaborators for "fractional derivative gradient-based optimizers".

For a parameter tensor w and ordinary gradient g = dL/dw, the modified gradient is

    g^(nu) = g * f_nu(w),

with

    f_nu(w) = (|w| + epsilon)^(1 - nu) / Gamma(2 - nu),

where
    - nu is the fractional order (`vderiv` below),
    - epsilon > 0 prevents singularities at w = 0,
    - Gamma(.) is the Gamma function.

This means:
    - the base optimizer structure is preserved,
    - only the gradient input is modified,
    - the fractional effect is local in optimization time,
    - no additional fractional memory/history is introduced.

Implemented optimizers
----------------------
    - FSGD
    - FAdam
    - FAdagrad
    - FAdadelta
    - FRMSprop
    - FAdamW

Important conceptual note
-------------------------
These are Herrera-style fractional gradient-scaled optimizers.
They are not explicit fractional-memory optimizers.

Compatibility
-------------
This version is intended for older `tf.keras.optimizers.Optimizer` APIs.
It accepts both:
    - learning_rate=...
    - lr=...
but internally stores the hyperparameter in the legacy style.

References
----------
1. Herrera-Alcántara, O.
   Fractional Derivative Gradient-Based Optimizers for Neural Networks and
   Human Activity Recognition.
   Applied Sciences, 12(18), 9264, 2022.
   https://doi.org/10.3390/app12189264

2. Herrera-Alcántara, O., Castelán-Aguilar, J. R.
   Fractional Gradient Optimizers for PyTorch: Enhancing GAN and BERT.
   Fractal and Fractional, 7(7), 500, 2023.
   https://doi.org/10.3390/fractalfract7070500

Usage example
-------------
    optimizer = FAdam(
        lr=1e-3,
        vderiv=0.9,
        frac_epsilon=1e-6
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

Notes
-----
- Sparse gradients are densified for simplicity.
- This is acceptable for most dense MLP/CNN experiments.
- For very large sparse models, specialized sparse updates may be preferable.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import tensorflow as tf


# =============================================================================
# Shared helper base
# =============================================================================


class _FractionalHerreraBase:
    """
    Shared helper functionality for Herrera-style fractional gradient scaling.
    """

    def _init_fractional_args(self, vderiv: float = 1.0, frac_epsilon: float = 1e-6) -> None:
        if not isinstance(vderiv, (float, int)):
            raise TypeError("`vderiv` must be a float.")
        if not (0.0 < float(vderiv) < 2.0):
            raise ValueError("`vderiv` must satisfy 0 < vderiv < 2.")
        if frac_epsilon <= 0.0:
            raise ValueError("`frac_epsilon` must be > 0.")

        self.vderiv = float(vderiv)
        self.frac_epsilon = float(frac_epsilon)
        self._gamma_factor = float(math.gamma(2.0 - self.vderiv))

    def _to_dense_gradient(self, grad, var):
        """
        Convert sparse gradients to dense tensors.

        For most dense-network experiments this is sufficient and keeps the
        implementation simpler and more robust across TF versions.
        """
        if isinstance(grad, tf.IndexedSlices):
            return tf.scatter_nd(
                indices=tf.expand_dims(grad.indices, axis=1),
                updates=grad.values,
                shape=tf.shape(var),
            )
        return grad

    def _fractional_factor(self, var):
        """
        Compute the Herrera fractional scaling factor for the current variable.
        """
        dtype = var.dtype.base_dtype
        eps = tf.cast(self.frac_epsilon, dtype)
        exponent = tf.cast(1.0 - self.vderiv, dtype)
        gamma_val = tf.cast(self._gamma_factor, dtype)

        return tf.pow(tf.abs(var) + eps, exponent) / gamma_val

    def _fractional_gradient(self, grad, var):
        """
        Apply the Herrera scaling factor to the current gradient.
        """
        grad = self._to_dense_gradient(grad, var)
        return grad * self._fractional_factor(var)

    def _fractional_config(self) -> Dict[str, Any]:
        return {
            "vderiv": self.vderiv,
            "frac_epsilon": self.frac_epsilon,
        }


# =============================================================================
# FSGD
# =============================================================================


class FSGD(_FractionalHerreraBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API Herrera-style fractional SGD.

    Update without momentum:
        g_t^(nu) = g_t * f_nu(w_t)
        w_{t+1} = w_t - lr * g_t^(nu)

    Update with momentum:
        m_t = momentum * m_{t-1} - lr * g_t^(nu)
        w_{t+1} = w_t + m_t
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        lr: float = None,
        momentum: float = 0.0,
        nesterov: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FSGD",
        **kwargs: Any,
    ):
        super(FSGD, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= momentum < 1.0):
            raise ValueError("`momentum` must satisfy 0 <= momentum < 1.")

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _create_slots(self, var_list):
        if self.momentum > 0.0:
            for var in var_list:
                self.add_slot(var, "momentum")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)

        grad = self._fractional_gradient(grad, var)

        if self.momentum > 0.0:
            momentum_coeff = tf.cast(self.momentum, var_dtype)
            m = self.get_slot(var, "momentum")

            new_m = momentum_coeff * m - lr_t * grad
            m_t = m.assign(new_m, use_locking=self._use_locking)

            if self.nesterov:
                var_t = var.assign_add(
                    momentum_coeff * new_m - lr_t * grad,
                    use_locking=self._use_locking,
                )
            else:
                var_t = var.assign_add(new_m, use_locking=self._use_locking)

            return tf.group(var_t, m_t)

        return var.assign_sub(lr_t * grad, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(FSGD, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdam
# =============================================================================


class FAdam(_FractionalHerreraBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API Herrera-style fractional Adam.
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
        frac_epsilon: float = 1e-6,
        name: str = "FAdam",
        **kwargs: Any,
    ):
        super(FAdam, self).__init__(name, **kwargs)

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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _create_slots(self, var_list):
        for var in var_list:
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

        grad = self._fractional_gradient(grad, var)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        new_m = beta_1_t * m + (1.0 - beta_1_t) * grad
        new_v = beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)

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

        return tf.group(var_t, m_t, v_t, vhat_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(FAdam, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdagrad
# =============================================================================


class FAdagrad(_FractionalHerreraBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API Herrera-style fractional Adagrad.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FAdagrad",
        **kwargs: Any,
    ):
        super(FAdagrad, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if initial_accumulator_value < 0.0:
            raise ValueError("`initial_accumulator_value` must be >= 0.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.initial_accumulator_value = float(initial_accumulator_value)
        self.epsilon = float(epsilon)
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _create_slots(self, var_list):
        initializer = tf.keras.initializers.Constant(self.initial_accumulator_value)
        for var in var_list:
            self.add_slot(var, "accumulator", initializer=initializer)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)

        grad = self._fractional_gradient(grad, var)
        acc = self.get_slot(var, "accumulator")

        new_acc = acc + tf.square(grad)
        acc_t = acc.assign(new_acc, use_locking=self._use_locking)

        var_t = var.assign_sub(
            lr_t * grad / (tf.sqrt(new_acc) + eps_t),
            use_locking=self._use_locking,
        )

        return tf.group(var_t, acc_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(FAdagrad, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "initial_accumulator_value": self.initial_accumulator_value,
                "epsilon": self.epsilon,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdadelta
# =============================================================================


class FAdadelta(_FractionalHerreraBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API Herrera-style fractional Adadelta.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        lr: float = None,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FAdadelta",
        **kwargs: Any,
    ):
        super(FAdadelta, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must satisfy 0 <= rho < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.rho = float(rho)
        self.epsilon = float(epsilon)
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "accumulated_grad")
            self.add_slot(var, "accumulated_delta")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
        rho_t = tf.cast(self.rho, var_dtype)
        eps_t = tf.cast(self.epsilon, var_dtype)

        grad = self._fractional_gradient(grad, var)

        acc_g = self.get_slot(var, "accumulated_grad")
        acc_d = self.get_slot(var, "accumulated_delta")

        new_acc_g = rho_t * acc_g + (1.0 - rho_t) * tf.square(grad)
        acc_g_t = acc_g.assign(new_acc_g, use_locking=self._use_locking)

        rms_delta = tf.sqrt(acc_d + eps_t)
        rms_grad = tf.sqrt(new_acc_g + eps_t)
        delta = - (rms_delta / rms_grad) * grad

        new_acc_d = rho_t * acc_d + (1.0 - rho_t) * tf.square(delta)
        acc_d_t = acc_d.assign(new_acc_d, use_locking=self._use_locking)

        var_t = var.assign_add(lr_t * delta, use_locking=self._use_locking)

        return tf.group(var_t, acc_g_t, acc_d_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(FAdadelta, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "epsilon": self.epsilon,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FRMSprop
# =============================================================================


class FRMSprop(_FractionalHerreraBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API Herrera-style fractional RMSprop.
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
        frac_epsilon: float = 1e-6,
        name: str = "FRMSprop",
        **kwargs: Any,
    ):
        super(FRMSprop, self).__init__(name, **kwargs)

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
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _create_slots(self, var_list):
        for var in var_list:
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

        grad = self._fractional_gradient(grad, var)

        velocity = self.get_slot(var, "velocity")
        new_velocity = rho_t * velocity + (1.0 - rho_t) * tf.square(grad)
        velocity_t = velocity.assign(new_velocity, use_locking=self._use_locking)

        if self.centered:
            avg_grad = self.get_slot(var, "avg_grad")
            new_avg_grad = rho_t * avg_grad + (1.0 - rho_t) * grad
            avg_grad_t = avg_grad.assign(new_avg_grad, use_locking=self._use_locking)
            denom = tf.sqrt(new_velocity - tf.square(new_avg_grad) + eps_t)
        else:
            avg_grad_t = tf.no_op()
            denom = tf.sqrt(new_velocity + eps_t)

        if self.momentum > 0.0:
            momentum_coeff = tf.cast(self.momentum, var_dtype)
            mom = self.get_slot(var, "momentum")
            new_mom = momentum_coeff * mom + lr_t * grad / denom
            mom_t = mom.assign(new_mom, use_locking=self._use_locking)
            var_t = var.assign_sub(new_mom, use_locking=self._use_locking)
            return tf.group(var_t, velocity_t, avg_grad_t, mom_t)

        var_t = var.assign_sub(lr_t * grad / denom, use_locking=self._use_locking)
        return tf.group(var_t, velocity_t, avg_grad_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(FRMSprop, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "rho": self.rho,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "centered": self.centered,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# FAdamW
# =============================================================================


class FAdamW(_FractionalHerreraBase, tf.keras.optimizers.Optimizer):
    """
    Legacy-API Herrera-style fractional AdamW with decoupled weight decay.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lr: float = None,
        weight_decay: float = 0.004,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        vderiv: float = 1.0,
        frac_epsilon: float = 1e-6,
        name: str = "FAdamW",
        **kwargs: Any,
    ):
        super(FAdamW, self).__init__(name, **kwargs)

        actual_lr = learning_rate if lr is None else lr
        self._set_hyper("learning_rate", actual_lr)

        if weight_decay < 0.0:
            raise ValueError("`weight_decay` must be >= 0.")
        if not (0.0 <= beta_1 < 1.0):
            raise ValueError("`beta_1` must satisfy 0 <= beta_1 < 1.")
        if not (0.0 <= beta_2 < 1.0):
            raise ValueError("`beta_2` must satisfy 0 <= beta_2 < 1.")
        if epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0.")

        self.weight_decay = float(weight_decay)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = bool(amsgrad)
        self._init_fractional_args(vderiv=vderiv, frac_epsilon=frac_epsilon)

    def _create_slots(self, var_list):
        for var in var_list:
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
        wd_t = tf.cast(self.weight_decay, var_dtype)
        step_t = tf.cast(self.iterations + 1, var_dtype)

        grad = self._fractional_gradient(grad, var)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        new_m = beta_1_t * m + (1.0 - beta_1_t) * grad
        new_v = beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)

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

        # Adam step
        var_t = var.assign_sub(
            lr_t * m_hat / (tf.sqrt(v_hat) + eps_t),
            use_locking=self._use_locking,
        )

        # Decoupled weight decay
        if self.weight_decay > 0.0:
            wd_op = var.assign_sub(lr_t * wd_t * var, use_locking=self._use_locking)
        else:
            wd_op = tf.no_op()

        return tf.group(var_t, wd_op, m_t, v_t, vhat_t)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        sparse_grad = tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var))
        return self._resource_apply_dense(sparse_grad, var, apply_state=apply_state)

    def get_config(self):
        config = super(FAdamW, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                **self._fractional_config(),
            }
        )
        return config


# =============================================================================
# Export list
# =============================================================================

__all__ = [
    "FSGD",
    "FAdam",
    "FAdagrad",
    "FAdadelta",
    "FRMSprop",
    "FAdamW",
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

    optimizer = FAdam(lr=1e-3, vderiv=0.9, frac_epsilon=1e-6)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x, y, epochs=1, batch_size=16, verbose=1)