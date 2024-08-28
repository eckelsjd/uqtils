import numpy as np
from scipy.stats import norm

from uqtils.mcmc import (
    autocorrelation,
    dram,
    is_positive_definite,
    nearest_positive_definite,
    normal_pdf,
    normal_sample,
)


def test_sample():
    """Test 1d and 2d batch normal sampling"""
    dim = 2
    shape = (4, 5)
    N = 100000
    mean = np.random.rand(*shape, dim)
    cov = np.eye(dim) * 0.01
    samples = normal_sample(mean, cov, N)
    assert samples.shape == (N, *shape, dim)
    assert np.allclose(np.mean(samples, axis=0), mean, rtol=1e-3, atol=1e-3)

    mean = np.random.rand()
    cov = 0.01
    samples = normal_sample(mean, cov, N)
    assert np.isclose(mean, np.mean(samples, axis=0), rtol=1e-3, atol=1e-3)


def test_pdf():
    mean = np.linspace(0, 3, 1000)
    x = np.random.rand(1000) * 3
    pdf = norm.pdf(x, loc=mean, scale=1)
    pdf2 = normal_pdf(x[..., np.newaxis], mean[..., np.newaxis], 1)
    assert np.allclose(pdf, pdf2, rtol=1e-4, atol=1e-8)


def test_mcmc():
    burnin = 0.1
    niter, nwalk, ndim = 3000, 4, 2
    cov = np.random.rand(ndim, ndim)
    if not is_positive_definite(cov):
        cov = nearest_positive_definite(cov)
    cov += np.eye(ndim) * (np.random.rand(ndim) + 1)
    cov = cov.T @ cov
    x0 = np.random.randn(nwalk, ndim)
    mu = np.random.rand(2)
    def fun(x):
        return normal_pdf(x, mu, cov, logpdf=True)
    samples, logpdf, accepted = dram(fun, x0, niter, cov0=cov, progress=False)
    samples = samples[int(burnin*niter):, ...]
    lags, autos, tau, ess = autocorrelation(samples)
    mean_pred = np.mean(samples, axis=(0, 1))
    assert np.all(np.abs(mean_pred) - np.abs(mu) < 0.1)
    assert np.mean(ess) > 100
