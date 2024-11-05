"""Assorted utilities for uncertainty quantification and scientific computing.

- Author - Joshua Eckels (eckelsjd.@umich.edu)
- License - GPL-3.0

Includes:

- **MCMC** - A standard DRAM MCMC sampler.
- **Gradients** - Vectorized finite-difference implementation of Jacobian and Hessians.
- **Plotting** - Some plotting utilities for `matplotlib`.
- **Sobol'** - Sobol' global, variance-based sensitivity analysis.
"""
from .gradient import *  # noqa: F403
from .mcmc import *  # noqa: F403
from .plots import *  # noqa: F403
from .sobol import *  # noqa: F403
from .uq_types import *  # noqa: F403

__version__ = "0.4.2"
__all__ = gradient.__all__ + mcmc.__all__ + plots.__all__ + sobol.__all__ + uq_types.__all__  # noqa: F405
