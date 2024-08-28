## Normal distribution
It is a frequent task to either sample from or evaluate the normal distribution. There are many ways to do this in
existing libraries, like `np.random.randn()` or `scipy.stats.norm`. However, sometimes you deal with multivariate data
of arbitrary shapes. What if you wanted to sample from `N` distinct multivariate Gaussians, each with their own
covariances and means? Or perhaps you want to use the same covariance for all `N` distributions and then evaluate
the pdf at `M` completely different locations?

The code snippet here shows how we generalize normal sampling and pdfs using a vectorized implementation.

```python title="Vectorized sample/pdf"
--8<-- "uqtils/example.py:normal"
```

## Gradients
It is another frequent task to obtain first and second derivatives of objective functions, say for maximum likelihood
estimation. The generalizations of 1st and 2nd derivatives for multivariate, vector-valued functions
(i.e. multiple inputs and multiple outputs) are the Jacobian and Hessian matrices. It is convenient to have functions
that evaluate the Jacobian/Hessian for arbitrary numbers of inputs or outputs, including the limiting case of a single
input and single output (which gives the normal 1st and 2nd derivatives that we all know and love).

The code snippet here shows generalized finite-difference approximations of the Jacobian/Hessian for arbitrary
input/output dimensions. The implementation is also vectorized over any number of extra axes, allowing you to evaluate
_multiple_ Jacobians/Hessians in one go.

```python title="Vectorized Jacobians/Hessians"
--8<-- "uqtils/example.py:gradient"
```

## Markov-Chain Monte Carlo
Here is an example of the delayed rejection adaptive Metropolis-Hastings (DRAM) algorithm for MCMC sampling. The
implementation is vectorized over `nwalkers`, allowing multiple sampling chains to proceed in parallel.

```python title="MCMC sampling"
--8<-- "uqtils/example.py:mcmc"
```

## Sobol' sensitivity analysis
Here is an example of getting the Sobol' indices for the [Ishigami](https://ieeexplore.ieee.org/document/151285) test function.

```python title="Sobol' sensitivity analysis"
--8<-- "uqtils/example.py:sobol"
```
