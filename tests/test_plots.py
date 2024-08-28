import numpy as np

from uqtils.plots import ndscatter, plot_slice


def test_plotting():
    """Test plotting utils"""
    mean = np.array([5, -2])
    cov = np.array([[2, 0.4], [0.2, 1]])
    samples = np.random.multivariate_normal(mean, cov.T @ cov, size=100)
    yt = samples[:, 0] + samples[:, 1] ** 2
    ysurr = yt + np.random.randn(*yt.shape)
    err = np.abs(ysurr - yt) / np.abs(yt)
    ndscatter(samples, labels=[r'$\alpha$', r'$\beta$', r'$\gamma$'], plot2d='scatter', cmap='plasma',
              cb_norm='log', z=err)

    funs = [lambda x: np.cos(x[..., 0]) + x[..., 1], lambda x: np.sin(x[..., 0]) + x[..., 1]]
    bds = [(-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]
    plot_slice(funs, bds, random_walk=True)
