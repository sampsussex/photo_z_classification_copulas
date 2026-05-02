import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from scipy.stats import distributions as dist

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
    sel = (g09['mag_Zt'] < 21.1) * (g09['g-i'] > 0.) * (g09['g-i'] < 2)

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
    #ax.plot(xr, pdf_vals, "r-", lw=2)
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


def beta_max_error(k, n, c=0.68):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    p_lower = dist.beta.ppf((1 - c) / 2.0, k + 1, n - k + 1)
    p_upper = dist.beta.ppf(1 - (1 - c) / 2.0, k + 1, n - k + 1)
    return p, max(p - p_lower, p_upper - p)


def plot_binned_completeness_comparison(
    z,
    g_minus_i,
    actual_complete,
    estimated_complete,
    x_bins,
    y_bins,
    x_label="Z",
    y_label="g-i",
    label="Sample",
    conf_level=0.68,
    save_path=None,
):
    x = np.asarray(z)
    y = np.asarray(g_minus_i)
    actual_complete = np.asarray(actual_complete).astype(bool)
    estimated_complete = np.asarray(estimated_complete)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(estimated_complete)
        & (estimated_complete >= 0)
        & (estimated_complete <= 1)
    )

    x = x[mask]
    y = y[mask]
    actual_complete = actual_complete[mask]
    estimated_complete = estimated_complete[mask]

    xbin = np.digitize(x, x_bins) - 1
    ybin = np.digitize(y, y_bins) - 1

    nx, ny = len(x_bins) - 1, len(y_bins) - 1

    n_grid = np.full((nx, ny), 0)
    actual_grid = np.full((nx, ny), np.nan)
    estimated_grid = np.full((nx, ny), np.nan)
    residual_grid = np.full((nx, ny), np.nan)

    actual_err_grid = np.full((nx, ny), np.nan)
    estimated_err_grid = np.full((nx, ny), np.nan)
    residual_err_grid = np.full((nx, ny), np.nan)

    for i in range(nx):
        for j in range(ny):
            in_cell = (xbin == i) & (ybin == j)
            n = in_cell.sum()
            n_grid[i, j] = n

            if n == 0:
                continue

            actual_vals = actual_complete[in_cell]
            estimated_vals = estimated_complete[in_cell]

            actual = actual_vals.sum() / len(actual_vals)
            estimated = estimated_vals.mean()
            residual = actual - estimated

            actual_err = beta_max_error(actual_vals.sum(), n, c=conf_level)[1]
            estimated_err = beta_max_error(estimated_vals.sum(), n, c=conf_level)[1]
            residual_err = estimated_err#np.sqrt(actual_err**2 + estimated_err**2)


            actual_grid[i, j] = actual
            estimated_grid[i, j] = estimated
            residual_grid[i, j] = residual / residual_err if residual_err > 0 else np.nan

            actual_err_grid[i, j] = actual_err
            estimated_err_grid[i, j] = estimated_err
            residual_err_grid[i, j] = residual_err

    all_errs = np.concatenate([
        actual_err_grid[np.isfinite(actual_err_grid)],
        estimated_err_grid[np.isfinite(estimated_err_grid)],
        residual_err_grid[np.isfinite(residual_err_grid)],
    ])

    global_max_err = np.nanmax(all_errs) if len(all_errs) else np.nan

    def draw_grid(ax, val_grid, err_grid, vmin=0, vmax=1, cmap="inferno"):
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        dx_arr = np.diff(x_bins)
        dy_arr = np.diff(y_bins)
        for i in range(nx):
            xc, dx = x_bins[i] + dx_arr[i] / 2, dx_arr[i]
            for j in range(ny):
                yc, dy = y_bins[j] + dy_arr[j] / 2, dy_arr[j]
                val, err = val_grid[i, j], err_grid[i, j]
                if not np.isfinite(val):
                    continue
                frac = np.clip(1.0 - 0.75 * (err / global_max_err if np.isfinite(err) and global_max_err > 0 else 0), 0.25, 1.0)
                ax.add_patch(Rectangle((xc - dx * frac / 2, yc - dy * frac / 2), dx * frac, dy * frac,
                                        facecolor=cmap_obj(norm(val)), edgecolor="black", linewidth=0.35))
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        return sm

    def add_size_legend(ax):
        if not (np.isfinite(global_max_err) and global_max_err > 0):
            return
        dx, dy = x_bins[1] - x_bins[0], y_bins[1] - y_bins[0]
        legend_x = x_bins[-1] + dx * 1.7
        y_positions = np.linspace(y_bins[0] + dy / 2, y_bins[-1] - dy / 2, 5)
        example_errs = np.linspace(0, global_max_err, 5)
        sizes = np.clip(1.0 - 0.75 * (example_errs / global_max_err), 0.25, 1.0)
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        cell_px = min(bbox.width * dx / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                      bbox.height * dy / (ax.get_ylim()[1] - ax.get_ylim()[0]))
        max_s = (cell_px / (fig.dpi / 72)) ** 2
        for yp, s, err in zip(y_positions, sizes, example_errs):
            ax.scatter(legend_x, yp, s=max_s * s ** 2, marker='s',
                       facecolor='lightgray', edgecolor='black', linewidth=0.8, clip_on=False, zorder=5)
            ax.text(legend_x + dx * 0.65, yp, f"{err:.3f}", ha="left", va="center", fontsize=8, clip_on=False)
        ax.text(legend_x - dx * 0.9, np.mean(y_positions), "Binomial 1$\\sigma$ error",
                ha="center", va="center", fontsize=9, clip_on=False, rotation=90)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.87,
        bottom=0.14,
        top=0.82,
        wspace=0.08,
    )

    residual_norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3) # TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    panels = [
        (
            actual_grid,
            actual_err_grid,
            "G09 completeness",
            "inferno",
            Normalize(vmin=0, vmax=1),
        ),
        (
            estimated_grid,
            estimated_err_grid,
            "Estimated completeness",
            "inferno",
            Normalize(vmin=0, vmax=1),
        ),
        (
            residual_grid,
            residual_err_grid,
            "Normalized Residual $(\Delta / \sigma)$",
            "coolwarm",
            residual_norm,
        ),
    ]

    for ax, (grid, err_grid, label_str, cmap, norm) in zip(axes, panels):
        sm = draw_grid(
            ax,
            grid,
            err_grid,
            vmin = 0 if cmap == "inferno" else -3,
            vmax = 1 if cmap == "inferno" else 3,
            cmap = cmap,
        )

        fig.colorbar(
            sm,
            ax=ax,
            location="top",
            pad=0.02,
        ).set_label(label_str)

        ax.set_xlabel(x_label)
        ax.set_xlim(x_bins[0], x_bins[-1])
        ax.set_ylim(y_bins[0], y_bins[-1])

    axes[0].set_ylabel(y_label)

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    add_size_legend(axes[2])

    plt.suptitle(label)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        'n_grid': n_grid,
        "actual": actual_grid,
        "estimated": estimated_grid,
        "residual": residual_grid,
        "actual_err": actual_err_grid,
        "estimated_err": estimated_err_grid,
        "residual_err": residual_err_grid,
    }