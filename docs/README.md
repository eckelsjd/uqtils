![Logo](assets/logo.svg)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![PyPI](https://img.shields.io/pypi/v/uqtils?logo=python&logoColor=%23cccccc)](https://pypi.org/project/uqtils)
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)

Assorted utilities for uncertainty quantification and scientific computing.

## Installation
You can install normally with:
```shell
pip install uqtils
```
If you are using [pdm](https://github.com/pdm-project/pdm) in your own project, then you can use:
```shell
cd <your-pdm-project>
pdm add uqtils
```
You can also quickly set up a development environment with:
```shell
# After forking this project on Github...
git clone https://github.com/<your-username>/uqtils.git
cd uqtils
pdm install  # reads pdm.lock and sets up an identical venv
```

## Quickstart
```python
import numpy as np
import uqtils as uq

ndim, nsamples = 3, 1000

mu = np.random.rand(ndim)
cov = np.eye(ndim)

samples = uq.normal_sample(mu, cov, nsamples)
fig, ax = uq.ndscatter(samples)
```

## Contributing
See the [contribution](CONTRIBUTING.md) guidelines.

<sup><sub>Made with the [UQ pdm template](https://github.com/eckelsjd/pdm-template-uq.git).</sub></sup>

