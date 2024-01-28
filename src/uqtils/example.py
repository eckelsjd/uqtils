"""Examples for using the package."""


def normal_example():
    # --8<-- [start:normal]
    import numpy as np
    import uqtils as uq
    import matplotlib.pyplot as plt

    ndim = 3
    shape = (5, ndim)

    means = np.random.randint(0, 10, size=shape).astype(np.float64)
    cov = np.eye(ndim) * 0.1

    samples = uq.normal_sample(means, cov, size=1000)     # (1000, 5, 3)
    pdfs = uq.normal_pdf(samples, means, cov)             # (1000, 5)

    fig, ax = uq.ndscatter(samples[:, 0, :])
    plt.show()
    # --8<-- [end:normal]


def gradient_example():
    # --8<-- [start:gradient]
    import numpy as np
    import uqtils as uq

    # 1d limiting case

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
    # --8<-- [start:mcmc]
    import numpy as np
    import uqtils as uq
    import matplotlib.pyplot as plt

    def fun(x):
        mu = [1, 1]
        cov = [[0.5, -0.1], [-0.1, 0.5]]
        return uq.normal_pdf(x, mu, cov, logpdf=True)

    nsamples, nwalkers, ndim = 1000, 16, 2
    x0 = np.random.randn(nwalkers, ndim)
    cov0 = np.eye(ndim)

    samples, log_pdf, accepted = uq.dram(fun, x0, cov0, nsamples)

    burn_in = int(0.1 * nsamples)
    samples = samples[burn_in:, ...].reshape((-1, ndim))
    fig, ax = uq.ndscatter(samples, plot='hist')
    plt.show()
    # --8<-- [end:mcmc]
