"""A package that provides basic utilities for uncertainty quantification and scientific computing.

- Author - Joshua Eckels (eckelsjd.@umich.edu)
- License - GNU GPLv3
"""
from .uq_types import Array
from .grad import *
from .mcmc import *
from .plots import *
from .sobol import *

__version__ = "0.3.1"
__all__ = [Array] + grad.__all__ + mcmc.__all__ + plots.__all__ + sobol.__all__
