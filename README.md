![Logo](https://raw.githubusercontent.com/eckelsjd/uqtils/main/docs/assets/logo.svg)

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
[![PyPI](https://img.shields.io/pypi/v/uqtils?logo=python&logoColor=%23cccccc)](https://pypi.org/project/uqtils)
![build](https://img.shields.io/github/actions/workflow/status/eckelsjd/uqtils/deploy.yml?logo=github)
![docs](https://img.shields.io/github/actions/workflow/status/eckelsjd/uqtils/docs.yml?logo=materialformkdocs&logoColor=%2523cccccc&label=docs)
![tests](https://img.shields.io/github/actions/workflow/status/eckelsjd/uqtils/tests.yml?logo=github&logoColor=%2523cccccc&label=tests)
![Code Coverage](https://img.shields.io/badge/coverage-88%25-yellowgreen?logo=codecov)

Assorted utilities for uncertainty quantification and scientific computing.

## ⚙️ Installation
```shell
pip install uqtils
```
If you are using [pdm](https://github.com/pdm-project/pdm) in your own project, then you can use:
```shell
pdm add uqtils

# Or in editable mode from a local clone...
pdm add -e ./uqtils --dev
```

## 📍 Quickstart
```python
import numpy as np
import uqtils as uq

ndim, nsamples = 3, 1000

mu = np.random.rand(ndim)
cov = np.eye(ndim)

samples = uq.normal_sample(mu, cov, nsamples)
fig, ax = uq.ndscatter(samples)
```

## 🏗️ Contributing
See the [contribution](https://github.com/eckelsjd/uqtils/blob/main/CONTRIBUTING.md) guidelines.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
