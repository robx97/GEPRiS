import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from parameters import get_param, FitParams
from matplotlib.offsetbox import AnchoredText

# ── low-level primitives ────────────────────────────────────────────────────

def plot_band(ax, x, y, sigma, color, zorder=1):
    ax.fill_between(x, y - sigma,   y + sigma,   color=color, alpha=0.25, zorder=zorder)
    ax.fill_between(x, y - 3*sigma, y + 3*sigma, color=color, alpha=0.15, zorder=zorder)


def plot_residual(ax, x, data, model, err, color='#444444'):
    res = (data - model) / err
    ax.errorbar(x, res, yerr=np.ones_like(res), fmt='o', color='#444444', ms=3)
    ax.fill_between(x, -1,  1,  color=color, alpha=0.25)
    ax.fill_between(x, -3,  3,  color=color, alpha=0.15)
    ax.axhline(0, ls='--', color=color, lw=0.8)
    ax.set_ylabel(r"Residual ($\sigma$)")
    ax.set_ylim(-5, 5)


# ── annotation helpers ───────────────────────────────────────────────────────

# Parameters worth displaying per panel type (nuisance norms excluded)
_DISPLAY_PARAMS = {
    'gamma':  ['A', 'kB', 'fC', 'kI'],
    'b12':    ["N_b12", "N_n12"],
    'c11':    ["N_c11"],
    'resol.': ['resol_a', 'resol_b', 'resol_bp', 'resol_c'],
}

_PARAM_LABELS = {
    'A':        r'$A$',
    'kB':       r'$k_B$',
    'fC':       r'$f_C$',
    'kI':       r'$k_I$',
    'N_b12':    r'$N_{^{12}B}$',
    'N_n12':    r'$N_{^{12}N}$',
    'N_c11':    r'$N_{^{11}C}$',
    'resol_a':  r'$a$',
    'resol_b':  r'$b$',
    'resol_bp': r"$b'$",
    'resol_c':  r'$c$',
}


def _format_val(v, err):
    """Auto-scale significant figures to match the error magnitude."""
    if err > 0:
        mag = int(np.floor(np.log10(err))) - 1      # 2 sig figs on error
        fmt = f'.{max(0, -mag)}f'
        return f'{v:{fmt}} ± {err:{fmt}}'
    return f'{v:.4g}'


def annotate_panel(ax, dataset, params, errors, panel_key):
    lines = []

    chi2_val = dataset.chi2(params)
    ndf = getattr(dataset, 'ndf', None)
    if ndf and ndf > 0:
        lines.append(rf'$\chi^2$/ndf = {chi2_val:.1f} / {ndf}')
    else:
        lines.append(rf'$\chi^2$ = {chi2_val:.2f}')

    for name in _DISPLAY_PARAMS.get(panel_key, []):
        try:
            v = get_param(params, name)
            err = errors.get(name, 0.0)
            label = _PARAM_LABELS.get(name, name)
            lines.append(rf'{label} = {_format_val(v, err)}')
        except KeyError:
            pass

    text = '\n'.join(lines)

    if panel_key == 'gamma':
        loc='lower center'
    elif panel_key == 'resol.':
        loc='center right'
    else:
        loc='lower left'
    
    anchored_box = AnchoredText(
        text, 
        loc=loc, 
        prop=dict(fontsize=9, family='monospace'),
        frameon=True,
        pad=0.3
    )
    
    anchored_box.patch.set_boxstyle("round,pad=0.3")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_alpha(0.75)
    anchored_box.patch.set_edgecolor("none")
    
    ax.add_artist(anchored_box)

# ── beta uncertainty bands ───────────────────────────────────────

def _beta_band_mc(model, e_grid, params, errors, is_pos, n_samples=200, random_seed=None, cov=None):
    """
    Monte Carlo 1-sigma band for electron or positron NL over e_grid.
    Varies A, kB, fC, kI jointly using the errors dict (diagonal cov).
    Returns (mean, sigma) arrays shaped like e_grid.
    """
    names  = ['A', 'kB', 'fC', 'kI']
    rng = np.random.default_rng(random_seed)
    means  = np.array([get_param(params, n) for n in names])

    if cov is not None:
        all_names = list(params.keys())
        idx = [all_names.index(n) for n in names]
        draw_cov = cov[np.ix_(idx, idx)]
    else:
        sigmas   = np.array([errors.get(n, 0.0) for n in names])
        draw_cov = np.diag(sigmas ** 2)

    draws  = rng.multivariate_normal(means, draw_cov, size=n_samples)

    curves = []
    for A_i, kB_i, fC_i, kI_i in draws:
        try:
            nl = model.beta_scint(e_grid, A_i, kB_i, fC_i, kI_i, is_pos=is_pos)
            if np.all(np.isfinite(nl)):
                curves.append(nl)
        except Exception:
            continue

    if len(curves) < 2:
        nominal = model.beta_scint(e_grid, *means, is_pos=is_pos)
        return nominal, np.zeros_like(nominal)

    stack = np.stack(curves)
    return stack.mean(axis=0), stack.std(axis=0, ddof=1)


# ──  plotters ───────────────────────────────────────────────────────

def plot_gamma(ax, ax_res, dataset, model, params, errors, bands=True, cov=None, color='mediumslateblue'):
    E    = dataset.E
    data = dataset.data
    err  = dataset.err
    pred = dataset.prediction(params)

    ax.errorbar(E, data, yerr=err, fmt='o', color='#444444', label="Data", zorder=3)
    ax.plot(E, pred, color=color, label=r'$\gamma$', zorder=2)

    # electron / positron overlay with optional MC bands
    e_grid = np.linspace(0.1, 10, 300)
    kI = get_param(params, "kI")

    if bands:
        e_mean,   e_sig = _beta_band_mc(model, e_grid, params, errors, is_pos=False, cov=cov)
        pos_mean, p_sig = _beta_band_mc(model, e_grid, params, errors, is_pos=True, cov=cov)

        ax.plot(e_grid,          e_mean,   '--',  label=r"$e^-$",  color='#E69F00', alpha=0.85)
        ax.plot(e_grid + 1.022,  pos_mean, '-.',  label=r"$e^+$",  color='deepskyblue', alpha=0.85)
        plot_band(ax, e_grid,          e_mean,   e_sig,  '#E69F00', zorder=1)
        plot_band(ax, e_grid + 1.022,  pos_mean, p_sig,  '#0077BB', zorder=1)

        # gamma data band
        g_sigma = dataset.uncertainty(params, errors, cov=cov)
        plot_band(ax, E, pred, g_sigma, color)
    else:
        elec = model.beta_scint(e_grid, get_param(params,'A'), get_param(params,'kB'),
                                get_param(params,'fC'), kI, is_pos=False)
        pos  = model.beta_scint(e_grid, get_param(params,'A'), get_param(params,'kB'),
                                get_param(params,'fC'), kI, is_pos=True)
        ax.plot(e_grid,         elec, '--', label=r"$e^-$", color='#AA3377', alpha=0.7)
        ax.plot(e_grid + 1.022, pos,  '-.', label=r"$e^+$", color='deepskyblue', alpha=0.7)

    ax.set_xlim(0, np.max(E) + 0.5)
    ax.set_ylabel("Non-linearity")
    #ax.set_title(r"$\gamma$ NL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    annotate_panel(ax, dataset, params, errors, 'gamma')
    plot_residual(ax_res, E, data, pred, err, color=color)


def plot_spectrum(ax, ax_res, dataset, params, errors,
                  label="Prediction", color='C1', bands=True, cov=None):
    x    = dataset.centers
    data = dataset.data
    err  = dataset.err
    pred = dataset.prediction(params)

    ax.plot(x, pred, color=color, label=label, zorder=2)
    ax.errorbar(x, data, yerr=err, fmt='.', color='#444444', label="Data", zorder=3)

    if dataset.name in ('c11', 'b12'):
        ax.set_yscale('log')

    if bands:
        sigma = dataset.uncertainty(params, errors, cov=cov)
        plot_band(ax, x, pred, sigma, color)

    ax.set_ylabel("Counts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    annotate_panel(ax, dataset, params, errors, dataset.name)
    plot_residual(ax_res, x, data, pred, err, color=color)


def plot_resolution(ax, ax_res, dataset, params, errors, bands=True, color='#228833'):
    x    = dataset.centers
    data = dataset.data
    err  = dataset.err
    pred = dataset.prediction(params)

    ax.errorbar(x, data, yerr=err, fmt='o', color='#444444', label="Data", zorder=3)
    ax.scatter(x, pred, color=color, label="Resolution", zorder=2, s=20)

    if bands:
        sigma = dataset.uncertainty(params, errors)
        plot_band(ax, x, pred, sigma, color)

    ax.set_ylabel(r"$\sigma/E$")
    #ax.set_title("Resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    annotate_panel(ax, dataset, params, errors, 'resol.')
    plot_residual(ax_res, x, data, pred, err, color=color)


# ── final plot ────────────────────────────────────────────────────────────

def make_full_plot(fitter, model, params, errors, bands=True, cov=None, save_path='first_fit.pdf'):
    """
    params : dict  {name: central_value}
    errors : dict  {name: 1-sigma}  — only free parameters need entries
    """
    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.15, wspace=0.3)

    ax_gamma = fig.add_subplot(gs[0, 0:2])
    ax_rg    = fig.add_subplot(gs[1, 0:2], sharex=ax_gamma)
    ax_reso  = fig.add_subplot(gs[0, 2])
    ax_rr    = fig.add_subplot(gs[1, 2],   sharex=ax_reso)
    ax_b12   = fig.add_subplot(gs[2, 1])
    ax_rb12  = fig.add_subplot(gs[3, 1],   sharex=ax_b12)
    ax_c11   = fig.add_subplot(gs[2, 2])
    ax_rc11  = fig.add_subplot(gs[3, 2],   sharex=ax_c11)
    ax_logo  = fig.add_subplot(gs[2:4, 0])

    try:
        logo = mpimg.imread('./inputs/GEPRiS_vector.png')
        ax_logo.imshow(logo)
    except FileNotFoundError:
        ax_logo.text(0.5, 0.5, 'GEPRiS', ha='center', va='center',
                     fontsize=24, transform=ax_logo.transAxes)
    ax_logo.axis('off')

    for ax in (ax_gamma, ax_reso, ax_b12, ax_c11):
        plt.setp(ax.get_xticklabels(), visible=False)

    dispatchers = {
        'gamma':  lambda ds: plot_gamma(ax_gamma, ax_rg,  ds, model, params, errors, bands, cov=cov),
        'b12':    lambda ds: plot_spectrum(ax_b12,  ax_rb12, ds, params, errors,
                                           label="B12 + N12", color='coral',    bands=bands, cov=cov),
        'c11':    lambda ds: plot_spectrum(ax_c11,  ax_rc11, ds, params, errors,
                                           label="C11",       color='firebrick', bands=bands, cov=cov),
        'resol.': lambda ds: plot_resolution(ax_reso, ax_rr, ds, params, errors, bands=bands),
    }

    for ds in fitter.datasets:
        name = getattr(ds, 'name', None)
        if name in dispatchers:
            dispatchers[name](ds)

    ax_rg.set_xlabel(r"$Energy$ (MeV)")
    ax_rr.set_xlabel(r"$Energy$ (MeV)")
    ax_rb12.set_xlabel(r"$Energy$ (MeV)")
    ax_rc11.set_xlabel(r"$Energy$ (MeV)")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    return fig