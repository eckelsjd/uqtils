import numpy as np

from uqtils.example import gradient_example, mcmc_example, normal_example, sobol_example
from uqtils.uq_types import format_input


def test_all_examples():
    normal_example()
    gradient_example()
    mcmc_example()
    sobol_example()


def test_format_inputs():
    test_cases = [1.1, np.array(2.1), np.random.rand(2, 1), np.random.rand(1, 2), np.random.rand(3, 3)]

    for array in test_cases:
        ndim = np.atleast_1d(array).shape[-1]
        _, x = format_input(array, ndim)
        assert len(x.shape) == 2
        assert x.shape[-1] == ndim
