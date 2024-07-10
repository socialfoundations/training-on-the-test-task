import pwlf
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette("colorblind")

from utils import compute_compute, color_rule
from regress_utils import get_hinge_regressor

def plot_bench(ax, results, title=None, ylim=None, xlim=None, neg=False, xticks=None, msize=10, plot_markers=False, title_fontsize=12, color_code='date', c_rule=None, ylabel=None, yticks=None):
    models = list(results.keys())
    c_rule = c_rule if c_rule else color_rule
    colors = [c_rule(model) for model in models]

    models_base = [c == palette[0] for c in colors]
    models_highlight = [c != palette[0] for c in colors]

    def scatter(ax, models, c): 
        per1 = [compute_compute(model) for model in models]
        per2 = [results[model] for model in models]
        if neg:
            per2 = [-p for p in per2]
        ax.scatter(per1, per2, alpha=0.6, s=msize, c=c)

    scatter(ax, [models[i] for i in range(len(models)) if models_base[i]], c=palette[0])
    scatter(ax, [models[i] for i in range(len(models)) if models_highlight[i]], c=palette[1])

    per1 = [compute_compute(model) for model in models]
    per2 = [results[model] for model in models]

    ax.grid()
    
    ax.set_xscale('log')

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)

    if plot_markers:
        for i, model in enumerate(models):
            ax.text(per1[i], per2[i], model, fontsize=6)

    
def plot_regressor(c, f, a, r, ax=None, plot_lines=True, main=True, linewidth=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2))

    color = [palette[f] for f in f]
    zorder = [1 if ff == 1 else 0 for ff in f]
    for c_, a_, color_, zorder_ in zip(c, a, color, zorder):
        ax.scatter(c_, a_, c=color_, alpha=0.6, s=60 if main else 40, zorder=zorder_)

    ax.grid(alpha=0.3)
    
    if not plot_lines:
        return
    
    _, ols, (p0, p1) = get_hinge_regressor(c, f, a, r)

    if linewidth is None:
        linewidth = 5 if main else 3
    ax.plot(*p0, c=palette[0], linewidth=linewidth, alpha=0.8)
    ax.plot(*p1, c=palette[1], linewidth=linewidth, alpha=0.8)

    ax.set_xscale('log')

    theta = ols['coeffs'][0]
    # print('Relative diff', theta / (max(a) - min(a)))

    theta = format(theta, '.3f')

    p = ols['ps'][0]
    if p < 0.05:
        theta = "\\mathbf{" + theta + "}"

    if main:  # do not plot r2 to save space in the plot
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label="Difference $\\hat{\\theta} =" + theta + "$", markerfacecolor=palette[0], markersize=0),
            plt.Line2D([0], [0], marker='o', color='w', label=f"Regression R$^2={ols['r2']}$", markerfacecolor=palette[1], markersize=0),
        ]
    else:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label="$\\hat{\\theta} =" + theta + "$", markerfacecolor=palette[0], markersize=0),
        ]

    ax.legend(handles=legend_elements, loc='upper left', handletextpad=-2., fontsize=12)

def plot_r2(ax, results, ylabel='Accuracy', title=None, ylim=None, xlim=None, msize=40, x=True, titles=True, **kwargs):
    models = list(results.keys())
    models = [m for m in models if 'neo' not in m]
    per1 = [compute_compute(model) for model in models]
    per2 = [results[model] for model in models]

    ax.scatter(per1, per2, color=palette[0], alpha=0.6, s=msize)
    if x:
        ax.set_xlabel('Pretraining compute', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.5)
    
    ax.set_xscale('log')

    if title is not None and titles:
        ax.set_title(title, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    
    mod = sm.OLS(per2,sm.add_constant(np.log(per1)))
    fii = mod.fit()
    ax.legend(["$R^2=$"+str(fii.summary2().tables[0][1][6])],fontsize=11,handlelength=0, handletextpad=0,markerscale=0,loc="upper left", frameon=True)

    ax.plot(per1,fii.summary2().tables[1]['Coef.'][-1]*np.log(per1)+fii.summary2().tables[1]['Coef.'][0],color=palette[3],alpha=0.8, linewidth=4)
    ax.set_xticks([1e20, 1e21, 1e22, 1e23, 1e24])
    ax.set_xlim(1e20, None)
    ax.tick_params(axis='both', which='both', length=0, labelsize=10)
    
def plot_emergence(ax, results, ylabel='Accuracy', title=None, ylim=None, xlim=None, msize=40, yconst=0.25, x=True, titles=True):
    models = list(results.keys())
    models = [m for m in models if 'neo' not in m]
    per1 = [compute_compute(model) for model in models]
    per2 = [results[model] for model in models]

    ax.scatter(per1, per2, color=palette[0], alpha=0.6, s=msize)
    if x:
        ax.set_xlabel('Pretraing compute', fontsize=12)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.5)
    
    ax.set_xscale('log')

    if title is not None and titles:
        ax.set_title(title, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    
    per1 = np.array(per1)
    per2 = np.array(per2)

    mask = (per1 > 1e20) & (per1 < 1e24)
    per1 = per1[mask]
    per2 = per2[mask]

    x = np.log(per1)
    y = per2


    xc = np.array([0.8 * 1e20, 0.7 * 1e20])
    xc = np.log(xc)
    yc = [yconst, yconst]

    # Fit the data with two line segments
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    res = my_pwlf.fit(2, xc, yc)

    xx = np.linspace(min(x), max(x), 100)
    yy = my_pwlf.predict(xx)

    ax.plot(np.exp(xx), yy, '-', c=palette[3], alpha=0.8, linewidth=4)
    
    emg = np.exp(res[1])
    if emg > 1e23:
        emg = 2.1 * 1e22

    ax.legend([f"$c_e$: {emg:.1e}"],fontsize=11,handlelength=0, handletextpad=0,markerscale=0,loc="upper left", frameon=True)
    ax.set_xticks([1e20, 1e21, 1e22, 1e23, 1e24])
    ax.tick_params(axis='both', which='both', length=0, labelsize=10)

    return res

def emergence_plots(data, base_data, steps, yconst, suptitle=None, emergence=True, ylabel='Accuracy', titles=True, x=True, axs=None):
    plot_f = plot_emergence if emergence else plot_r2

    data[0] = base_data
    
    if axs is None:
        _, axs = plt.subplots(1, len(steps), figsize=(9.5,1.5), dpi=200, sharex=True, sharey=True)

    for i, step in enumerate(steps):
        examples = step * 64  # batch size
        to_plot = data[step]
        to_plot = {m: to_plot[m] for m in to_plot if color_rule(m) == palette[0]}

        examples_k = '0'
        if examples > 0:
            if len(str(examples)) > 4:
                examples_k = str(int(examples/1000)) + 'k'
            else:
                 examples_k = str(examples/1000) + 'k'

        plot_f(axs[i], to_plot, title=f"Task examples: {examples_k}", ylabel=ylabel*(step==0), yconst=yconst, x=x, titles=titles)

        if 'mmlu' in ylabel.lower():
            axs[i].set_yticks([0.25, 0.5, 0.7])
            axs[i].set_ylim(0.2, 0.75)
        else:
            axs[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            axs[i].set_ylim(-0.03, 0.84)

        for i in range(1, 4):
            axs[i].set_yticklabels([])

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
