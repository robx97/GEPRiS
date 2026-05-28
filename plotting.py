import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_gamma(ax, ax_res, dataset, model, params, errors, bands=True, color='C0'):
    E = dataset.E
    data = dataset.data
    err = dataset.err

    pred = dataset.prediction(params)

    ax.errorbar(E, data, yerr=err, fmt='o', label="Data")
    ax.plot(E, pred, color=color, label="Model")

    #electron/positron curves
    e_grid = np.linspace(0.1, 10, 200)
    elec = model.beta_scint(e_grid, params["A"], params["kB"], params["fC"])
    pos  = model.beta_scint(e_grid, params["A"], params["kB"], params["fC"], is_pos=True)

    ax.plot(e_grid, elec, '--', label=r"$e^-$", alpha=0.7)
    ax.plot(e_grid+1.022, pos,  '-.', label=r"$e^+$", alpha=0.7)
    ax.set_xlim(0, np.max(E) + 1.0)

    # uncertainty bands
    if bands:
        mean, sigma = model.gamma_mc_uncertainty(
            E,
            params["A"], params["kB"], params["fC"], params.get("alpha", 0),
            errors["A"], errors["kB"], errors["fC"], errors.get("alpha", 0),
        )
        plot_band(ax, E, mean, sigma, color)

    ax.set_ylabel("Non-linearity")
    ax.set_title(r"$\gamma$ Fit")
    ax.legend()
    ax.grid()

    # residuals
    plot_residual(ax_res, E, data, pred, err)
    
def plot_residual(ax, x, data, model, err, sigma=None, color='k'):
    if sigma is None:
        sigma = err

    res = (data - model) / sigma
    res_err = err / sigma

    ax.errorbar(x, res, yerr=res_err, fmt='o', color=color)

    # bands
    ax.fill_between(x, -1, 1, color=color, alpha=0.3)
    ax.fill_between(x, -3, 3, color=color, alpha=0.15)

    ax.axhline(0, ls='--', color=color)
    ax.set_ylabel("Residual ($\\sigma$)")
    ax.set_ylim(-6, 6)
    
def plot_band(ax, x, y, sigma, color):
    ax.fill_between(x, y - 3*sigma, y + 3*sigma, color=color, alpha=0.2)
    ax.fill_between(x, y - sigma,  y + sigma,  color=color, alpha=0.4)
    
def plot_spectrum(ax, ax_res, dataset, params, errors,
                  label="Prediction", color='C1', bands=True, scale_param=None):

    x = dataset.centers
    data = dataset.data
    err = dataset.err

    pred = dataset.prediction(params)

    # apply normalization if needed
    if scale_param:
        pred = pred * params[scale_param]

    ax.plot(x, pred, color=color, label=label)
    if dataset.name in ['c11', 'b12']:
        ax.semilogy()
    ax.errorbar(x, data, yerr=err, fmt='.', label="Data")

    if bands and hasattr(dataset, "uncertainty"):
        sigma = dataset.uncertainty(params, errors)
        plot_band(ax, x, pred, sigma, color)

    ax.set_ylabel("Counts")
    ax.legend()

    plot_residual(ax_res, x, data, pred, err)
    
def plot_resolution(ax, ax_res, E, sigma_data, sigma_err, model, params, errors, bands=True):
    pred = model.juno_resolution(E,
                                params["a"], params["b"], params["c"],
                                fit=True)

    ax.errorbar(E, sigma_data, yerr=sigma_err, fmt='o', label="Data")
    ax.scatter(E, pred, label="Model")

    if bands:
        sigma = model.juno_reso_error(
            E,
            params["a"], params["b"], params["c"],
            errors["a"], errors["b"], errors["c"]
        )
        plot_band(ax, E, pred, sigma, 'C2')

    ax.set_ylabel(r"$\sigma/E$")
    ax.set_title("Resolution")
    ax.legend()

    plot_residual(ax_res, E, sigma_data, pred, sigma_err, sigma=sigma)
    
def make_full_plot(fitter, model, params, errors, bands=True):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 3, height_ratios=[3,1,3,1])

    ax_gamma = fig.add_subplot(gs[0, 0:2])
    ax_rg    = fig.add_subplot(gs[1, 0:2], sharex=ax_gamma)

    ax_reso  = fig.add_subplot(gs[0, 2])
    ax_rr    = fig.add_subplot(gs[1, 2], sharex=ax_reso)

    ax_b12   = fig.add_subplot(gs[2, 1])
    ax_rb12  = fig.add_subplot(gs[3, 1], sharex=ax_b12)

    ax_c11   = fig.add_subplot(gs[2, 2])
    ax_rc11  = fig.add_subplot(gs[3, 2], sharex=ax_c11)

    logo = mpimg.imread('./inputs/GEPRiS_vector.png')
    ax_logo   = fig.add_subplot(gs[2:4, 0])
    ax_logo.imshow(logo)
    ax_logo.axis('off')
    

    for ds in fitter.datasets:
        if not hasattr(ds, 'name'):
            continue
        if ds.name == "gamma":
            plot_gamma(ax_gamma, ax_rg, ds, model, params, errors, bands)

        elif ds.name == "b12":
            plot_spectrum(ax_b12, ax_rb12, ds, params, errors,
                          label="B12+N12", color='coral', bands=bands)

        elif ds.name == "c11":
            plot_spectrum(ax_c11, ax_rc11, ds, params, errors,
                          label="C11", color='firebrick', bands=bands)
                          
        elif ds.name == "resol.":
            plot_spectrum(ax_reso, ax_rr, ds, params, errors,
                          label="Resolution", bands=bands)

    plt.tight_layout()
    fig.savefig('first_fit.pdf')
    return fig
