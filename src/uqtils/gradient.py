"""Module for vectorized finite-difference gradient approximations.

Includes:

- `approx_jac` - vectorized Jacobian approximation
- `approx_hess` - vectorized Hessian approximation
"""
import numpy as np

from .uq_types import Array

__all__ = ['approx_jac', 'approx_hess']


def approx_jac(func, theta: Array, pert=0.01) -> np.ndarray:
    """Approximate Jacobian of `func` at a specified `theta` location using finite difference approximation.

    :param func: expects to be called as `func(theta) -> (..., y_dim)`
    :param theta: `(..., theta_dim)`, points to linearize model about
    :param pert: perturbation percent for approximate partial derivatives
    :returns J: `(..., y_dim, theta_dim)`, the approximate Jacobian `(y_dim, theta_dim)` at all locations `(...)`
    """
    theta = np.atleast_1d(theta)
    shape = theta.shape[:-1]                # (...,)
    theta_dim = theta.shape[-1]             # Number of parameters
    dtheta = pert * np.abs(theta)

    # Make sure dtheta is not 0 anywhere
    for i in range(theta_dim):
        zero_idx = np.isclose(dtheta[..., i], 0)
        if np.any(zero_idx):
            subs_dtheta = pert * np.abs(np.mean(theta[..., i]))
            if np.isclose(subs_dtheta, 0):
                subs_dtheta = pert
            dtheta[zero_idx, i] = subs_dtheta

    # Return the Jacobians (..., y_dim, theta_dim)
    J, y_dim = None, None

    for i in range(theta_dim):
        theta_n1 = np.copy(theta)
        theta_p1 = np.copy(theta)

        # Perturbations to theta
        theta_n1[..., i] -= dtheta[..., i]
        theta_p1[..., i] += dtheta[..., i]
        f_n1 = func(theta_n1)
        f_p1 = func(theta_p1)

        if J is None:
            y_dim = f_p1.shape[-1]
            J = np.empty(shape + (y_dim, theta_dim))

        J[..., i] = (f_p1 - f_n1) / np.expand_dims(2 * dtheta[..., i], axis=-1)

    if y_dim == 1:
        J = np.squeeze(J, axis=-2)
        if theta_dim == 1:
            J = np.squeeze(J, axis=-1)
    return np.atleast_1d(J)


def approx_hess(func, theta: Array, pert=0.01) -> np.ndarray:
    """Approximate Hessian of `func` at a specified `theta` location using finite difference approximation.

    :param func: expects to be called as `func(theta) -> (..., y_dim)`
    :param theta: `(..., theta_dim)`, points to linearize model about
    :param pert: perturbation percent for approximate partial derivatives
    :returns H: `(..., y_dim, theta_dim, theta_dim)`, the approximate Hessian `(theta_dim, theta_dim)` at all locations
                `(...,)` for vector-valued function of dimension `y_dim`
    """
    theta = np.atleast_1d(theta)
    shape = theta.shape[:-1]                # (...,)
    theta_dim = theta.shape[-1]             # Number of parameters
    dtheta = pert * np.abs(theta)

    # Make sure dtheta is not 0 anywhere
    for i in range(theta_dim):
        zero_idx = np.isclose(dtheta[..., i], 0)
        if np.any(zero_idx):
            subs_dtheta = pert * np.abs(np.mean(theta[..., i]))
            if np.isclose(subs_dtheta, 0):
                subs_dtheta = pert
            dtheta[zero_idx, i] = subs_dtheta

    # Return the Hessians (..., y_dim, theta_dim, theta_dim)
    y_dim, H = None, None

    for i in range(theta_dim):
        for j in range(i, theta_dim):
            # Allocate space at 4 grid points (n1=-1, p1=+1)
            theta_n1_n1 = np.copy(theta)
            theta_p1_p1 = np.copy(theta)
            theta_n1_p1 = np.copy(theta)
            theta_p1_n1 = np.copy(theta)

            # Perturbations to theta in each direction
            theta_n1_n1[..., i] -= dtheta[..., i]
            theta_n1_n1[..., j] -= dtheta[..., j]
            f_n1_n1 = func(theta_n1_n1)

            theta_p1_p1[..., i] += dtheta[..., i]
            theta_p1_p1[..., j] += dtheta[..., j]
            f_p1_p1 = func(theta_p1_p1)

            theta_n1_p1[..., i] -= dtheta[..., i]
            theta_n1_p1[..., j] += dtheta[..., j]
            f_n1_p1 = func(theta_n1_p1)

            theta_p1_n1[..., i] += dtheta[..., i]
            theta_p1_n1[..., j] -= dtheta[..., j]
            f_p1_n1 = func(theta_p1_n1)

            if H is None:
                y_dim = f_p1_n1.shape[-1]
                H = np.empty(shape + (y_dim, theta_dim, theta_dim))

            res = (f_n1_n1 + f_p1_p1 - f_n1_p1 - f_p1_n1) / np.expand_dims(4 * dtheta[..., i] * dtheta[..., j],
                                                                           axis=-1)
            H[..., i, j] = res
            H[..., j, i] = res

    if y_dim == 1:
        H = np.squeeze(H, axis=-3)
        if theta_dim == 1:
            H = np.squeeze(H, axis=(-1, -2))
    return np.atleast_1d(H)
