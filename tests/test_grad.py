import numpy as np

from uqtils.gradient import approx_hess, approx_jac


def test_jacobian():
    """Test the Jacobian of $f_1(x, y) = 2x^2 + 3xy^3$, $f_2(x, y) = 4y^2 + 2x^3y$"""
    shape = (11, 12)
    dim = 2
    theta0 = np.random.rand(*shape, dim)
    f1 = lambda theta: 2 * theta[..., 0:1] ** 2 + 3 * theta[..., 0:1] * theta[..., 1:2] ** 3
    f2 = lambda theta: 4 * theta[..., 1:2] ** 2 + 2 * theta[..., 0:1] ** 3 * theta[..., 1:2]
    fun = lambda theta: np.concatenate((f1(theta), f2(theta)), axis=-1)

    J_tilde = approx_jac(fun, theta0)
    J_exact = np.empty(shape + (2, dim))
    J_exact[..., 0, 0] = 4 * theta0[..., 0] + 3 * theta0[..., 1]**3
    J_exact[..., 0, 1] = 9 * theta0[..., 0] * theta0[..., 1]**2
    J_exact[..., 1, 0] = 6 * theta0[..., 0]**2 * theta0[..., 1]
    J_exact[..., 1, 1] = 8 * theta0[..., 1] + 2 * theta0[..., 0]**3

    assert np.allclose(J_tilde, J_exact, rtol=1e-3, atol=1e-3)


def test_hessian():
    """Test the Hessian of $f_1(x, y) = 2x^2 + 3xy^3$, $f_2(x, y) = 4y^2 + 2x^3y$"""
    shape = (11, 12)
    dim = 2
    theta0 = np.random.rand(*shape, dim)
    f1 = lambda theta: 2 * theta[..., 0:1]**2 + 3 * theta[..., 0:1] * theta[..., 1:2]**3
    f2 = lambda theta: 4 * theta[..., 1:2]**2 + 2 * theta[..., 0:1]**3 * theta[..., 1:2]
    fun = lambda theta: np.concatenate((f1(theta), f2(theta)), axis=-1)

    H_tilde = approx_hess(fun, theta0)
    H_exact = np.empty(shape + (2, dim, dim))
    H_exact[..., 0, 0, 0] = 4
    H_exact[..., 0, 0, 1] = 9 * theta0[..., 1]**2
    H_exact[..., 0, 1, 0] = 9 * theta0[..., 1]**2
    H_exact[..., 0, 1, 1] = 18 * theta0[..., 0] * theta0[..., 1]
    H_exact[..., 1, 0, 0] = 12 * theta0[..., 0] * theta0[..., 1]
    H_exact[..., 1, 0, 1] = 6 * theta0[..., 0] ** 2
    H_exact[..., 1, 1, 0] = 6 * theta0[..., 0] ** 2
    H_exact[..., 1, 1, 1] = 8

    assert np.allclose(H_tilde, H_exact, rtol=1e-3, atol=1e-3)
