import numpy as np
import scipy.stats as st

from uqtils.sobol import ishigami, sobol_sa


def test_sobol_sa():
    model = lambda x: ishigami(x)['y']
    n_samples = 2000

    def lhc_sampler(shape):
        N, dim = np.prod(shape), 3
        x = st.qmc.LatinHypercube(d=dim).random(n=N)
        x = st.qmc.scale(x, [-np.pi] * dim, [np.pi] * dim).reshape(shape + (dim,))
        return x

    # rand_sampler = lambda shape: np.random.rand(*shape, 3) * (2 * np.pi) - np.pi
    S1, S2, ST, bar_plot, pie_plot = sobol_sa(model, lhc_sampler, n_samples, cmap='summer', plot=True, compute_s2=True)

    S1_est = np.mean(S1, axis=0)[:, 0]
    S1_se = np.sqrt(np.var(S1, axis=0) / n_samples)[:, 0]
    S1_truth = np.array([0.3139, 0.4424, 0])

    S1_lb, S1_ub = S1_est - 2.5*S1_se, S1_est + 2.5*S1_se
    assert np.all(np.logical_and(S1_truth <= S1_ub, S1_truth >= S1_lb))
