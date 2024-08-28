"""Module for Markov-Chain Monte Carlo routines.

Includes:

- `normal_pdf` - vectorized Gaussian pdf evaluation
- `normal_sample` - vectorized Gaussian sampling
- `is_positive_definite` - whether a matrix is positive semi-definite
- `nearest_positive_definite` - finds the nearest PSD matrix
- `dram` - Delayed rejection adaptive Metropolis-Hastings MCMC
- `autocorrelation` - computes the autocorrelation of a set of samples
"""
import warnings

import h5py
import numpy as np
import tqdm

from .uq_types import Array

__all__ = ['normal_pdf', 'normal_sample', 'is_positive_definite', 'nearest_positive_definite', 'dram',
           'autocorrelation']


def normal_sample(mean: Array, cov: Array, size: tuple | int = (), sqrt=False) -> np.ndarray:
    """Generic batch sample multivariate normal distributions (pretty much however you want).

    !!! Note
        The provided `mean` and `cov` should match along the last dimension, that is the dimension of the random
        variables to sample. If you want to sample a 1d Gaussian, then you can specify both the mean and covariance
        as scalars. However, as long as the mean and covariance are broadcastable in size, then you can use this
        function however you want, (i.e. sample many multivariate distributions at once, all with different means
        and covariances, etc., just get creative)

    :param mean: `(..., dim)`, expected values, where `dim` is the random variable dimension
    :param cov: `(..., dim, dim)`, covariance matrices (or the sqrt(cov) if `sqrt=True`)
    :param size: shape of additional samples
    :param sqrt: whether `cov` was passed in already as the `sqrt(cov)` via cholesky decomposition
    :returns samples: `(*size, ..., dim)`, samples from multivariate distributions
    """
    mean = np.atleast_1d(mean)
    cov = np.atleast_2d(cov)
    sqrt_cov = cov if sqrt else np.linalg.cholesky(cov)

    if isinstance(size, int):
        size = (size, )
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    x_normal = np.random.standard_normal((*shape, 1)).astype(mean.dtype)
    samples = np.squeeze(sqrt_cov @ x_normal, axis=-1) + mean
    return samples


def normal_pdf(x: Array, mean: Array, cov: Array, logpdf: bool = False) -> np.ndarray:
    """Compute the Gaussian pdf at each `x` location (pretty much however you want).

    :param x: `(..., dim)`, the locations to evaluate the pdf at
    :param mean: `(..., dim)`, expected values, where dim is the random variable dimension
    :param cov: `(..., dim, dim)`, covariance matrices
    :param logpdf: whether to return the logpdf instead
    :returns: `(...,)` the pdf values at `x`
    """
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)
    cov = np.atleast_2d(cov)
    dim = cov.shape[-1]

    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))
    diff = x - mean
    diff_col = np.expand_dims(diff, axis=-1)    # (..., dim, 1)
    diff_row = np.expand_dims(diff, axis=-2)    # (..., 1, dim)
    inexp = np.squeeze(diff_row @ np.linalg.inv(cov) @ diff_col, axis=(-1, -2))

    pdf = np.log(preexp) - 0.5 * inexp if logpdf else preexp * np.exp(-0.5 * inexp)

    return pdf


def is_positive_definite(A):
    """Returns true when input is positive-definite, via Cholesky."""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input.

    A Python port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix", 1988
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    eye = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += eye * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def dram(logpdf, x0, niter, cov0=None, gamma=0.5, eps=1e-6, adapt_after=100, adapt_interval=10,
         delayed=True, progress=True, filename=None):
    """Delayed adaptive metropolis-hastings MCMC with a Gaussian proposal.

    :param logpdf: log PDF function of target distribution
    :param x0: `(nwalkers, ndim)` initial parameter samples, ignored if samples exist in `filename`
    :param cov0: `(ndim, ndim)` the initial proposal covariance, defaults to identity or `cov` value in filename
    :param niter: number of iterations
    :param gamma: scale factor for the covariance matrix for delayed rejection step
    :param eps: small constant for making sure covariance is well-conditioned
    :param adapt_after: the number of iterations before covariance adaptation begins (ignored if <=0)
    :param adapt_interval: the number of iterations between each covariance adaptation (ignored if `adapt_after<=0`)
    :param delayed: whether to try to sample again after first rejection
    :param progress: whether to display progress of the sampler
    :param filename: if specified, an hdf5 file to save results to. If the file already has dram results, the new
                     samples will be appended. Follows the same format as the `emcee` library
    :returns: `samples, log_pdf, acceptance` - `(niter, nwalkers, ndim)` samples of the target distribution, the logpdf
              values at these locations, and the cumulative number of accepted samples per walker
    """
    # Override x0, cov0 if filename already has samples
    try:
        if filename is not None:
            with h5py.File(filename, 'a') as fd:
                group = fd.get('mcmc', None)
                if group is not None:
                    x0 = group['chain'][-1, ...]
                    if cov0 is None:
                        cov0 = np.array(group['cov'])  # only override if cov0 is not passed in
                    niter += 1
    except Exception as e:
        warnings.warn(str(e))

    # Initialize
    x0 = np.atleast_2d(x0)
    nwalk, ndim = x0.shape
    cov0 = np.eye(ndim) if cov0 is None else cov0
    sd = (2.4**2/ndim)
    curr_cov = np.broadcast_to(cov0, (nwalk, ndim, ndim)).copy().astype(x0.dtype)
    curr_chol = np.linalg.cholesky(curr_cov)
    adapt_cov = curr_cov.copy()  # adaptive covariance
    curr_mean = x0
    curr_loc_logpdf = logpdf(x0)
    samples = np.empty((niter, nwalk, ndim), dtype=x0.dtype)
    log_pdf = np.empty((niter, nwalk), dtype=x0.dtype)
    accepted = np.zeros((nwalk,), dtype=x0.dtype)
    samples[0, ...] = x0
    log_pdf[0, ...] = curr_loc_logpdf

    def accept_first(curr_log, prop_log):
        with np.errstate(over='ignore'):
            # Overflow values go to -> infty, so they will always get accepted
            ret = np.minimum(1.0, np.exp(prop_log - curr_log))
        return ret

    # Main sample loop
    iterable = tqdm.tqdm(range(niter-1)) if progress else range(niter-1)
    # --8<-- [start:dram]
    for i in iterable:
        # Propose sample
        x1 = samples[i, ...]
        y1 = normal_sample(x1, curr_chol, sqrt=True)    # (nwalkers, ndim)
        x1_log = curr_loc_logpdf
        y1_log = logpdf(y1)

        # Compute first acceptance
        with np.errstate(invalid='ignore'):
            a1 = y1_log - x1_log                        # (nwalkers,)
        a1_idx = a1 > 0
        a1_idx |= np.log(np.random.rand(nwalk)) < a1
        samples[i + 1, a1_idx, :] = y1[a1_idx, :]
        samples[i + 1, ~a1_idx, :] = x1[~a1_idx, :]
        curr_loc_logpdf[a1_idx] = y1_log[a1_idx]
        accepted[a1_idx] += 1

        # Second level proposal
        if delayed and np.any(~a1_idx):
            y2 = normal_sample(x1[~a1_idx, :], curr_chol[~a1_idx, ...] * np.sqrt(gamma), sqrt=True)
            y2_log = logpdf(y2)
            with ((np.errstate(divide='ignore', invalid='ignore'))):
                # If a(y2, y1)=1, then log(1-a(y2,y1)) -> -infty and a2 -> 0
                frac_1 = y2_log - x1_log[~a1_idx]
                frac_2 = (normal_pdf(y1[~a1_idx, :], y2, curr_cov[~a1_idx, ...], logpdf=True) -
                          normal_pdf(y1[~a1_idx, :], x1[~a1_idx, :], curr_cov[~a1_idx, ...], logpdf=True))
                frac_3 = (np.log(1 - accept_first(y2_log, y1_log[~a1_idx])) -
                          np.log(1 - np.minimum(1.0, np.exp(a1[~a1_idx]))))
                a2 = frac_1 + frac_2 + frac_3
            a2_idx = a2 > 0
            a2_idx |= np.log(np.random.rand(a2.shape[0])) < a2

            sample_a2_idx = np.where(~a1_idx)[0][a2_idx]  # Indices that were False the 1st time, then true the 2nd
            samples[i + 1, sample_a2_idx, :] = y2[a2_idx, :]
            curr_loc_logpdf[sample_a2_idx] = y2_log[a2_idx]
            accepted[sample_a2_idx] += 1

        log_pdf[i+1, ...] = curr_loc_logpdf

        # Update the sample mean and cov every iteration
        if adapt_after > 0:
            k = i + 1
            last_mean = curr_mean.copy()
            curr_mean = (1/(k+1)) * samples[k, ...] + (k/(k+1))*last_mean
            mult = (np.eye(ndim) * eps + k * last_mean[..., np.newaxis] @ last_mean[..., np.newaxis, :] -
                    (k + 1) * curr_mean[..., np.newaxis] @ curr_mean[..., np.newaxis, :] +
                    samples[k, ..., np.newaxis] @ samples[k, ..., np.newaxis, :])
            adapt_cov = ((k - 1) / k) * adapt_cov + (sd / k) * mult

            if k > adapt_after and k % adapt_interval == 0:
                try:
                    curr_chol[:] = np.linalg.cholesky(adapt_cov)
                    curr_cov[:] = adapt_cov[:]
                except np.linalg.LinAlgError:
                    warnings.warn(f"Non-PSD matrix at k={k}. Ignoring...")
    # --8<-- [end:dram]

    try:
        if filename is not None:
            with h5py.File(filename, 'a') as fd:
                group = fd.get('mcmc', None)
                if group is not None:
                    samples = np.concatenate((group['chain'], samples[1:, ...]), axis=0)
                    log_pdf = np.concatenate((group['log_pdf'], log_pdf[1:, ...]), axis=0)
                    accepted += group['accepted']
                    del group['chain']
                    del group['log_pdf']
                    del group['accepted']
                    del group['cov']
                fd.create_dataset('mcmc/chain', data=samples)
                fd.create_dataset('mcmc/log_pdf', data=log_pdf)
                fd.create_dataset('mcmc/accepted', data=accepted)
                fd.create_dataset('mcmc/cov', data=curr_cov)
    except Exception as e:
        warnings.warn(str(e))

    return samples, log_pdf, accepted


def autocorrelation(samples, maxlag=100, step=1):
    """Compute the auto-correlation of a set of samples.

    :param samples: `(niter, nwalk, ndim)` samples returned from `dram` or a similar MCMC routine
    :param maxlag: maximum distance to compute the correlation for
    :param step: step between distances from 0 to `maxlag` for which to compute the correlations
    :returns: lags, autos, tau, ess - the lag times, auto-correlations, integrated auto-correlation,
              and effective sample sizes
    """
    niter, nwalk, ndim = samples.shape
    mean = np.mean(samples, axis=0)
    var = np.sum((samples - mean[np.newaxis, ...]) ** 2, axis=0)

    lags = np.arange(0, maxlag, step)
    autos = np.zeros((len(lags), nwalk, ndim))
    for zz, lag in enumerate(lags):
        # compute the covariance between all samples *lag apart*
        for ii in range(niter - lag):
            autos[zz, ...] += (samples[ii, ...] - mean) * (samples[ii + lag, ...] - mean)
        autos[zz, ...] /= var
    tau = 1 + 2 * np.sum(autos, axis=0)     # Integrated auto-correlation
    ess = niter / tau                       # Effective sample size
    return lags, autos, tau, ess
