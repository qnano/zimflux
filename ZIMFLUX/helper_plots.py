# import matplotlib

import numpy as np
import scipy
from scipy.optimize import curve_fit
# font = {'size': 14}
# matplotlib.rc('font', **font)
#
# fig, ax = plt.subplots(figsize=(6, 4))
# fig.tight_layout()
# ax.legend(edgecolor=(1, 1, 1, 1.), facecolor=(1, 1, 1, 1))
#
# plt.savefig('sinewave.png', bbox_inches='tight', pad_inches=0, dpi=900)
def histo_gaussian(ax, x, xlabel, ylabel,xlim=0,ylim=0, color='blue',binwidth=0.1, init_mean = -2):

    def gauss(x, a, x0, sigma, c):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) + c)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    """Customized line plot with error bars."""

    init_fit = scipy.stats.norm.fit(x)
    n, bins= np.histogram(x,
                                     bins=np.arange(min(x), max(x) + binwidth,
                                                    binwidth))

    popt, pcov= curve_fit(gauss, (bins[1:] + bins[:-1]) / 2, n,
                                     p0=[np.sum(n),init_mean, np.std(x)*2, 0],
                                     maxfev=2000, method='trf')
    ax.hist(x,
             bins=np.arange(np.min(x), max(x) + binwidth, binwidth))
    ax.plot(np.linspace(xlim[0],
                         xlim[1], 1000),   gauss(np.linspace(xlim[0],xlim[1], 1000), *popt))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim !=0:
        ax.set_xlim(xlim)
    if ylim!=0:
        ax.set_ylim(ylim)
    return ax, popt[1], popt[0],popt[2]

def custom_lineplot(ax, x, y, error, xlims, ylims, color='blue'):
    """Customized line plot with error bars."""

    ax.errorbar(x, y, yerr=error,  ls='none', marker='o', capsize=2, capthick=1, ecolor='black')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax

def custom_lineplot_err(ax, x, y, line ,sampling, error, xlims, ylims, color='blue', label=''):
    """Customized line plot with error bars."""
    import matplotlib.pyplot as plt
    ax.errorbar(x, y, yerr=error, ls='none', marker='o', ecolor='black', capsize=2,markersize=2)

    ax.plot(sampling,line,label=label)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax



def custom_scatterplot(ax, x, y, error, xlims, ylims, color='green', markerscale=100):
    """Customized scatter plot where marker size is proportional to error measure."""

    markersize = error * markerscale

    ax.scatter(x, y, color=color, marker='o', s=markersize, alpha=0.5)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax


def custom_barchart(ax, x, y, error, xlims, ylims, error_kw, color='lightblue', width=0.75):
    """Customized bar chart with positive error bars only."""

    error = [np.zeros(len(error)), error]

    ax.bar(x, y, color=color, width=width, yerr=error, error_kw=error_kw, align='center')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax


def custom_boxplot(ax, x, y, error, xlims, ylims, mediancolor='magenta'):
    """Customized boxplot with solid black lines for box, whiskers, caps, and outliers."""

    medianprops = {'color': mediancolor, 'linewidth': 2}
    boxprops = {'color': 'black', 'linestyle': '-'}
    whiskerprops = {'color': 'black', 'linestyle': '-'}
    capprops = {'color': 'black', 'linestyle': '-'}
    flierprops = {'color': 'black', 'marker': 'x'}

    ax.boxplot(y,
               positions=x,
               medianprops=medianprops,
               boxprops=boxprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               flierprops=flierprops)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    return ax


def stylize_axes(ax,xlabel, ylabel):
    """Customize axes spines, title, labels, ticks, and ticklabels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)

    #ax.set_title(title)


