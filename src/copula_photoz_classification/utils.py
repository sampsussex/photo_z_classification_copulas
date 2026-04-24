import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flux2mag(flux):
    return 8.9 - 2.5 * np.log10(flux)

def load_g09(region = 'all', spec_zs = 'paus',
             path_to_g09_photom= '/Users/sp624AA/Downloads/qmost_spv/G09_d1m3p1f1.parquet', 
             path_to_g09_photos= '/Users/sp624AA/Downloads/qmost_spv/G09_Photoz_Combined_Sabine Bellstedt_v3.parquet'):
    if region not in ['all', 'wide', 'deep']:
        raise ValueError(f"Unknown region {region} (must be 'all', 'wide' or 'deep')")
    
    if spec_zs not in ['paus']:
        raise ValueError(f"Unknown spec_zs {spec_zs} (must be 'paus'- mixed not implemented yet)")
    
    photom_cols_to_load = ['uberID','RAmax', 'Decmax', 'mag_Zt', 'flux_it', 'flux_gt', 'flux_Yt', 'flux_rt', 
                           'flux_Kt', 'flux_Jt','starmask', 'ghostmask', 'mask', 'class', 'duplicate']
    photom_g09 = pd.read_parquet(path_to_g09_photom, columns=photom_cols_to_load)
    photom_g09['mag_it'] = flux2mag(photom_g09['flux_it'])
    photom_g09['mag_gt'] = flux2mag(photom_g09['flux_gt'])
    photom_g09['mag_Yt'] = flux2mag(photom_g09['flux_Yt'])
    photom_g09['mag_Kt'] = flux2mag(photom_g09['flux_Kt'])
    photom_g09['mag_Jt'] = flux2mag(photom_g09['flux_Jt'])
    photom_g09['mag_rt'] = flux2mag(photom_g09['flux_rt'])
    photom_g09['g-i'] = photom_g09['mag_gt'] - photom_g09['mag_it']
    photom_g09['uberID'] = photom_g09['uberID'].astype("int64")
    mask = (
        (photom_g09['mask'] == 0) * 
        (photom_g09['starmask'] == 0) * 
        (photom_g09['ghostmask'] == 0) * 
        (photom_g09['class'] !=  'star') * 
        (photom_g09['class'] != 'artefact') *
        (photom_g09['duplicate'] == 0) *
        (photom_g09['mag_Zt'] < 21.25)
    )
    photom_g09 = photom_g09[mask]

    photo_z_cols_to_load = ['uberID', 'P020_comb', 'P080_comb' ,'zphot_invar', 
                            'zphot_err', 'z_paus', 'qz_paus', 'chi2_paus', 'z_spec', 
                            'z_NQ', 'z_source']
    photo_z_g09 = pd.read_parquet(path_to_g09_photos, columns=photo_z_cols_to_load)

    photo_z_g09.rename(columns={'zphot_invar': 'photoZ', 'zphot_err': 'photoZ_err'}, inplace=True)
    photo_z_g09['uberID'] = photo_z_g09['uberID'].astype("int64")

    photo_z_g09 = photo_z_g09[photo_z_g09['z_paus'].notna()]

    photo_z_g09['z_best'] = photo_z_g09['z_paus']

    # where z_spec is not Nan, replace z_best with z_spec
    photo_z_g09.loc[photo_z_g09['z_spec'].notna(), 'z_best'] = photo_z_g09.loc[photo_z_g09['z_spec'].notna(), 'z_spec']

    #if spec_zs == 'paus':
    # git rid of this strange value 
    #photo_z_g09 = photo_z_g09[photo_z_g09['z_best'].notna()]
    photo_z_g09 = photo_z_g09[photo_z_g09['z_best'] != 0.6729999780654907226562]


    # inner merge on uberID
    photom_g09 = photom_g09.merge(photo_z_g09, on='uberID', how='inner')
    del photo_z_g09
           
    return photom_g09


def load_g09_waveswide_xys(region = 'all', spec_zs = 'paus',
             path_to_g09_photom= '/Users/sp624AA/Downloads/qmost_spv/G09_d1m3p1f1.parquet', 
             path_to_g09_photos= '/Users/sp624AA/Downloads/qmost_spv/G09_Photoz_Combined_Sabine Bellstedt_v3.parquet'):
    g09 = load_g09(region, spec_zs, path_to_g09_photom, path_to_g09_photos)
    sel = (g09['g-i'] > 0.) * (g09['g-i'] < 2) * (g09['mag_Zt'] < 21.1)

    g09_valid = g09[sel]
    g09_valid = g09_valid[g09_valid['z_best'] < 0.2]
    g09_valid = g09_valid.dropna(subset=['P020_comb', 'g-i', 'z_best', 'mag_Zt'])

    g09_valid['TP'] = g09_valid['P020_comb'] > 0.14

    x_all = -g09_valid['mag_Zt'].to_numpy() + 21.1
    y_all = g09_valid['g-i'].to_numpy()
    xy_all = np.column_stack((x_all, y_all))

    xy_all_tp_mask = g09_valid['TP'].to_numpy()

    xy_fn = xy_all[~xy_all_tp_mask]

    return xy_all, xy_fn, xy_all_tp_mask



def plot_marginal_fits(results, data_map, save_path="all_fits_marginals.png"):
    """
    Plot fit diagnostics for each variable in `results`.

    Parameters
    ----------
    results  : dict returned by fit_all_marginals()
    data_map : dict mapping the same keys to the original raw data arrays,
               e.g. {"x_all": x_all, "x_fn": x_fn, "y_all": y_all, "y_fn": y_fn}
    save_path: file path for the saved figure
    """
    keys   = list(results.keys())
    n_rows = len(keys)

    fig, axs = plt.subplots(n_rows, 4, figsize=(18, 3.5 * n_rows))
    fig.suptitle("Distribution Fits → CDF → Uniform Marginals", fontsize=13)

    for row, key in enumerate(keys):
        res  = results[key]
        data = data_map[key]
        xr, pdf_vals, cdf_vals, u = res['xr'], res['pdf_vals'], res['cdf_vals'], res['u']

        _plot_fit        (axs[row, 0], data, xr, pdf_vals, key)
        _plot_cdf        (axs[row, 1], data, xr, cdf_vals)
        _plot_uniform    (axs[row, 2], u)
        _plot_qq_uniform (axs[row, 3], u)

    for ax in axs.flat:
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def _plot_fit(ax, data, xr, pdf_vals, title):
    """Col 0: histogram of data overlaid with fitted PDF."""
    ax.hist(data, bins=30, density=True, alpha=0.5, color="steelblue")
    ax.plot(xr, pdf_vals, "r-", lw=2)
    ax.set(title=title, ylabel="density")


def _plot_cdf(ax, data, xr, cdf_vals):
    """Col 1: empirical step CDF vs fitted CDF."""
    xs = np.sort(data)
    empirical = np.arange(1, len(data) + 1) / len(data)
    ax.step(xs, empirical, color="steelblue", lw=1.5, label="empirical")
    ax.plot(xr, cdf_vals, "r-", lw=2, label="fitted")
    ax.set(title="CDF", ylabel="F(x)")
    ax.legend(fontsize=7)


def _plot_uniform(ax, u):
    """Col 2: histogram of uniform marginals — should be flat at 1."""
    ax.hist(u, bins=30, density=True, alpha=0.5, color="mediumseagreen")
    ax.axhline(1.0, color="r", lw=2, ls="--", label="U[0,1]")
    ax.set(title="Uniform marginals", ylabel="density")
    ax.legend(fontsize=7)


def _plot_qq_uniform(ax, u):
    """Col 3: Q-Q plot of uniform marginals vs theoretical U[0,1]."""
    ax.scatter(np.sort(u), np.linspace(0, 1, len(u)), s=6, alpha=0.4, color="steelblue")
    ax.plot([0, 1], [0, 1], "r-", lw=2)
    ax.set(title="Q-Q (vs uniform)", xlabel="F(x)", ylabel="empirical")


def plot_uniform_marginals(uv, save_path=None, title="Uniform Marginals Scatter Plot"):
    """Plot uniform marginals against each other, in a classic copula diagnostic."""
    plt.figure(figsize=(6, 6))
    plt.scatter(uv[:, 0], uv[:, 1], s=10, alpha=0.5, color="black")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
