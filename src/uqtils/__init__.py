"""Assorted utilities for uncertainty quantification and scientific computing..

***Includes:***

- **MCMC** - A standard DRAM MCMC sampler. (1) And some vectorized normal samplers.
- **Gradients** - Vectorized finite-difference implementation of Jacobian and Hessians.
- **Plotting** - Some plotting utilities for `matplotlib`.
- **Sobol'** - Sobol' global, variance-based sensitivity analysis.
{ .annotate }

1.
```python title="DRAM MCMC sampler"
--8<-- "uqtils/mcmc.py:dram"
```

- Author - Joshua Eckels (eckelsjd.@umich.edu)
- License - GPL-3.0
"""
from .uq_types import Array
from .grad import *
from .mcmc import *
from .plots import *
from .sobol import *

__version__ = "0.3.1"
__all__ = [Array] + grad.__all__ + mcmc.__all__ + plots.__all__ + sobol.__all__
