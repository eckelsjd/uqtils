"""Examples for using the package."""
# ruff: noqa: F841
# ruff: noqa: I001

def normal_example():
    """Sample and plot a normal pdf."""
    # --8<-- [start:normal]
    import numpy as np
    import uqtils as uq

    ndim = 3
    shape = (5, ndim)

    means = np.random.randint(0, 10, size=shape).astype(np.float64)
    cov = np.eye(ndim) * 0.1

    samples = uq.normal_sample(means, cov, size=1000)     # (1000, 5, 3)
    pdfs = uq.normal_pdf(samples, means, cov)             # (1000, 5)

    fig, ax = uq.ndscatter(samples[:, 0, :])
    # --8<-- [end:normal]


def gradient_example():
    """Evaluate 1d and multivariate gradients."""
    # --8<-- [start:gradient]
    import numpy as np
    import uqtils as uq

    # 1d example
    def f(x):
        return np.sin(x)

    x0 = 1.0
    df_dx = uq.approx_jac(f, x0)
    d2f_dx2 = uq.approx_hess(f, x0)

    # Multivariate example
    n_in, n_out = 3, 2
    def f(x):
        x0, x1, x2 = [x[..., i] for i in range(n_in)]
        f0 = x0 * x1 + x2
        f1 = np.sin(x0)**2 + x2**3
        return np.concatenate((f0[..., np.newaxis], f1[..., np.newaxis]), axis=-1)

    shape = (100, 5, n_in)
    x0 = np.random.rand(*shape)
    jac  = uq.approx_jac(f, x0)      # (100, 5, n_out, n_in)
    hess = uq.approx_hess(f, x0)     # (100, 5, n_out, n_in, n_in)
    # --8<-- [end:gradient]


def mcmc_example():
    """Sample from a logpdf distribution using MCMC."""
    # --8<-- [start:mcmc]
    import numpy as np
    import uqtils as uq

    def fun(x):
        mu = [1, 1]
        cov = [[0.5, -0.1], [-0.1, 0.5]]
        return uq.normal_pdf(x, mu, cov, logpdf=True)

    nsamples, nwalkers, ndim = 1000, 4, 2
    x0 = np.random.randn(nwalkers, ndim)
    cov0 = np.eye(ndim)

    samples, log_pdf, accepted = uq.dram(fun, x0, nsamples, cov0=cov0)

    burn_in = int(0.1 * nsamples)
    samples = samples[burn_in:, ...].reshape((-1, ndim))
    fig, ax = uq.ndscatter(samples, plot2d='hist')
    # --8<-- [end:mcmc]

def sobol_example():
    """Do Sobol' analysis on the Ishigami test function."""
    # --8<-- [start:sobol]
    import numpy as np
    import uqtils as uq

    model = lambda x: uq.ishigami(x)['y']
    sampler = lambda shape: np.random.rand(*shape, 3) * (2 * np.pi) - np.pi
    n_samples = 1000

    S1, ST = uq.sobol_sa(model, sampler, n_samples)
    # --8<-- [end:sobol]
