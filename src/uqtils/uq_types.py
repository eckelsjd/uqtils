"""Custom types"""
import numpy as np

Array = list | float | np.ndarray

__all__ = ['Array', 'format_input']


def format_input(x: Array, ndim: int) -> tuple[bool, np.ndarray]:
    """Helper function to make sure input `x` is an `ndarray` of shape `(..., ndim)`.

    :param x: if 1d-like as `(n,)`, then converted to 2d as `(1, n) if n==ndim or (n, 1) if ndim==1`
    :param ndim: the dimension of the inputs
    :returns: `x` as at least a 2d array `(..., ndim)`, and whether `x` was originally 1d-like
    """
    x = np.atleast_1d(x)
    is_1d = len(x.shape) == 1
    if is_1d:
        if x.shape[0] != ndim and ndim > 1:
            raise ValueError(f'Input x shape {x.shape} is incompatible with ndim of {ndim}')
        x = np.expand_dims(x, axis=0 if x.shape[0] == ndim else 1)

    return is_1d, x
