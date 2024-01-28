"""Module for plotting utilities.

Includes
--------
- `ax_default` - Nice default plt formatting for x-y data
- `plot_slice` - Plots a grid of 1d slices of a multivariate function
- `ndscatter` - Plots a grid of 1d and 2d marginals in a "corner plot" for n-dimensional data
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import StrMethodFormatter, AutoLocator, FuncFormatter

from .uq_types import Array

__all__ = ['ax_default', 'plot_slice', 'ndscatter']


def ax_default(ax: plt.Axes, xlabel='', ylabel='', legend=False, cmap='tab10'):
    """Nice default formatting for plotting X-Y data.

    :param ax: the axes to apply these settings to
    :param xlabel: the xlabel to set for `ax`
    :param ylabel: the ylabel to set for `ax`
    :param legend: whether to show a legend
    :param cmap: colormap to use for cycling
    """
    plt.rcParams["axes.prop_cycle"] = _get_cycle(cmap)
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    if legend:
        leg = ax.legend(fancybox=True)
        frame = leg.get_frame()
        frame.set_edgecolor('k')


def _get_cycle(cmap: str | matplotlib.colors.Colormap, num_colors: int = None):
    """Get a color cycler for plotting.

    :param cmap: a string specifier of a matplotlib colormap (or a colormap instance)
    :param num_colors: the number of colors to cycle through
    """
    use_index = False
    if isinstance(cmap, str):
        use_index = cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
                             'tab10', 'tab20', 'tab20b', 'tab20c']
        cmap = matplotlib.cm.get_cmap(cmap)
    if num_colors is None:
        num_colors = cmap.N
    if cmap.N > 100:
        use_index = False
    elif isinstance(cmap, LinearSegmentedColormap):
        use_index = False
    elif isinstance(cmap, ListedColormap):
        use_index = True
    if use_index:
        ind = np.arange(int(num_colors)) % cmap.N
        return cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, num_colors))
        return cycler("color", colors)


def plot_slice(funs, bds: list[tuple], x0: Array = None, x_idx: list[int] = None,
               y_idx: list[int] = None, N: int = 50, random_walk: bool = False, xlabels: list[str] = None,
               ylabels: list[str] = None, cmap='viridis', fun_labels=None):
    """Helper function to plot 1d slices of a function(s) over inputs.

    :param funs: function callable as `y=f(x)`, with `x` as `(..., xdim)` and `y` as `(..., ydim)`, can also be a list
                of functions to evaluate and plot together.
    :param bds: list of tuples of `(min, max)` specifying the bounds of the inputs
    :param x0: the default values for all inputs; defaults to middle of `bds`
    :param x_idx: list of input indices to take 1d slices of
    :param y_idx: list of output indices to plot 1d slices of
    :param N: the number of points to take in each 1d slice
    :param random_walk: whether to slice in a random d-dimensional direction or hold all params const while slicing
    :param xlabels: list of labels for the inputs
    :param ylabels: list of labels for the outputs
    :param cmap: the name of the matplotlib colormap to use
    :param fun_labels: the legend labels if plotting multiple functions on each plot
    :returns: `fig, ax` with `num_inputs` by `num_outputs` subplots
    """
    funs = funs if isinstance(funs, list) else [funs]
    x_idx = list(np.arange(0, min(3, len(bds)))) if x_idx is None else x_idx
    y_idx = [0] if y_idx is None else y_idx
    xlabels = [f'x{i}' for i in range(len(x_idx))] if xlabels is None else xlabels
    ylabels = [f'QoI {i}' for i in range(len(y_idx))] if ylabels is None else ylabels
    fun_labels = [f'fun {i}' for i in range(len(funs))] if fun_labels is None else fun_labels
    x0 = [(b[0] + b[1]) / 2 for b in bds] if x0 is None else x0
    x0 = np.atleast_1d(x0)
    xdim = x0.shape[0]
    lb = np.atleast_1d([b[0] for b in bds])
    ub = np.atleast_1d([b[1] for b in bds])
    cmap = plt.get_cmap(cmap)

    # Construct sliced inputs
    xs = np.zeros((N, len(x_idx), xdim))
    for i in range(len(x_idx)):
        if random_walk:
            # Make a random straight-line walk across d-cube
            r0 = np.random.rand(xdim) * (ub - lb) + lb
            r0[x_idx[i]] = lb[x_idx[i]]                     # Start slice at this lower bound
            rf = np.random.rand(xdim) * (ub - lb) + lb
            rf[x_idx[i]] = ub[x_idx[i]]                     # Slice up to this upper bound
            xs[0, i, :] = r0
            for k in range(1, N):
                xs[k, i, :] = xs[k-1, i, :] + (rf-r0)/(N-1)
        else:
            # Otherwise, only slice one variable
            for j in range(xdim):
                if j == x_idx[i]:
                    xs[:, i, j] = np.linspace(lb[x_idx[i]], ub[x_idx[i]], N)
                else:
                    xs[:, i, j] = x0[j]

    # Compute function values and show ydim by xdim grid of subplots
    ys = []
    for func in funs:
        y = func(xs)
        if y.shape == (N, len(x_idx)):
            y = y[..., np.newaxis]
        ys.append(y)
    c_intervals = np.linspace(0, 1, len(ys))

    fig, axs = plt.subplots(len(y_idx), len(x_idx), sharex='col', sharey='row')
    for i in range(len(y_idx)):
        for j in range(len(x_idx)):
            if len(y_idx) == 1:
                ax = axs if len(x_idx) == 1 else axs[j]
            elif len(x_idx) == 1:
                ax = axs if len(y_idx) == 1 else axs[i]
            else:
                ax = axs[i, j]
            x = xs[:, j, x_idx[j]]
            for k in range(len(ys)):
                y = ys[k][:, j, y_idx[i]]
                ax.plot(x, y, ls='-', color=cmap(c_intervals[k]), label=fun_labels[k])
            ylabel = ylabels[i] if j == 0 else ''
            xlabel = xlabels[j] if i == len(y_idx) - 1 else ''
            legend = (i == 0 and j == len(x_idx) - 1 and len(ys) > 1)
            ax_default(ax, xlabel, ylabel, legend=legend)
    fig.set_size_inches(3 * len(x_idx), 3 * len(y_idx))
    fig.tight_layout()

    return fig, axs


def ndscatter(samples: np.ndarray, labels: list[str] = None, tick_fmts: list[str] = None, plot='scatter',
              cmap='viridis', bins=20, z: np.ndarray = None, cb_label=None, cb_norm='linear', subplot_size=3):
    """Triangle scatter plots of n-dimensional samples.

    !!! Warning
        Best for `dim < 10`. You can shrink the `subplot_size` to assist graphics loading time.

    :param samples: `(N, dim)` samples to plot
    :param labels: list of axis labels of length `dim`
    :param tick_fmts: list of str.format() specifiers for ticks, e.g `['{x: ^10.2f}', ...]`, of length `dim`
    :param plot: 'hist' for 2d hist plot, 'kde' for kernel density estimation, or 'scatter' (default)
    :param cmap: the matplotlib string specifier of a colormap
    :param bins: number of bins in each dimension for histogram marginals
    :param z: `(N,)` a performance metric corresponding to `samples`, used to color code the scatter plot if provided
    :param cb_label: label for color bar (if `z` is provided)
    :param cb_norm: `str` or `plt.colors.Normalize`, normalization method for plotting `z` on scatter plot
    :param subplot_size: size in inches of a single 2d marginal subplot
    :returns fig, axs: the `plt` Figure and Axes objects, (returns an additional `cb_fig, cb_ax` if `z` is specified)
    """
    N, dim = samples.shape
    x_min = np.min(samples, axis=0)
    x_max = np.max(samples, axis=0)
    if labels is None:
        labels = [f"x{i}" for i in range(dim)]
    if z is None:
        z = np.zeros(N)
    if cb_label is None:
        cb_label = 'Performance metric'

    def tick_format_func(value, pos):
        if np.isclose(value, 0):
            return f'{value:.2f}'
        if abs(value) > 1:
            return f'{value:.2f}'
        if abs(value) > 0.01:
            return f'{value:.4f}'
        if abs(value) < 0.01:
            return f'{value:.2E}'
    default_ticks = FuncFormatter(tick_format_func)
    # if tick_fmts is None:
    #     tick_fmts = ['{x:.2G}' for i in range(dim)]

    # Set up triangle plot formatting
    fig, axs = plt.subplots(dim, dim, sharex='col', sharey='row')
    for i in range(dim):
        for j in range(dim):
            ax = axs[i, j]
            if i == j:                      # 1d marginals on diagonal
                # ax.get_shared_y_axes().remove(ax)
                ax._shared_axes['y'].remove(ax)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if i == 0:
                    ax.get_yaxis().set_ticks([])
            if j > i:                       # Clear the upper triangle
                ax.axis('off')
            if i == dim - 1:                # Bottom row
                ax.set_xlabel(labels[j])
                ax.xaxis.set_major_locator(AutoLocator())
                formatter = StrMethodFormatter(tick_fmts[j]) if tick_fmts is not None else default_ticks
                ax.xaxis.set_major_formatter(formatter)
            if j == 0 and i > 0:            # Left column
                ax.set_ylabel(labels[i])
                ax.yaxis.set_major_locator(AutoLocator())
                formatter = StrMethodFormatter(tick_fmts[i]) if tick_fmts is not None else default_ticks
                ax.yaxis.set_major_formatter(formatter)

    # Plot marginals
    for i in range(dim):
        for j in range(dim):
            ax = axs[i, j]
            if i == j:                      # 1d marginals (on diagonal)
                c = plt.get_cmap(cmap)(0)
                if plot == 'kde':
                    kernel = st.gaussian_kde(samples[:, i])
                    x = np.linspace(x_min[i], x_max[i], 1000)
                    ax.fill_between(x, y1=kernel(x), y2=0, lw=0, alpha=0.3, facecolor=c)
                    ax.plot(x, kernel(x), ls='-', c=c, lw=1.5)
                else:
                    ax.hist(samples[:, i], edgecolor='black', color=c, density=True, alpha=0.5,
                            linewidth=1.2, bins='auto', histtype='step')
            if j < i:                       # 2d marginals (lower triangle)
                ax.set_xlim([x_min[j], x_max[j]])
                ax.set_ylim([x_min[i], x_max[i]])
                if plot == 'scatter':
                    sc = ax.scatter(samples[:, j], samples[:, i], s=1.5, c=z, cmap=cmap, norm=cb_norm)
                elif plot == 'hist':
                    ax.hist2d(samples[:, j], samples[:, i], bins=bins, density=True, cmap=cmap)
                elif plot == 'kde':
                    kernel = st.gaussian_kde(samples[:, [j, i]].T)
                    xg, yg = np.meshgrid(np.linspace(x_min[j], x_max[j], 60), np.linspace(x_min[i], x_max[i], 60))
                    x = np.vstack([xg.ravel(), yg.ravel()])
                    zg = np.reshape(kernel(x), xg.shape)
                    ax.contourf(xg, yg, zg, 5, cmap=cmap, alpha=0.9)
                    ax.contour(xg, yg, zg, 5, colors='k', linewidths=1.5)
                else:
                    raise NotImplementedError('This plot type is not known. plot=["hist", "kde", "scatter"]')

    fig.set_size_inches(subplot_size * dim, subplot_size * dim)
    fig.tight_layout()

    # Plot colorbar in standalone figure
    if np.max(z) > 0 and plot == 'scatter':
        cb_fig, cb_ax = plt.subplots(figsize=(1.5, 6))
        cb_fig.subplots_adjust(right=0.7)
        cb_fig.colorbar(sc, cax=cb_ax, orientation='vertical', label=cb_label)
        cb_fig.tight_layout()
        return fig, axs, cb_fig, cb_ax

    return fig, axs
