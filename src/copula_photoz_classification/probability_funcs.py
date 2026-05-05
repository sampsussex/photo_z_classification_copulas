import numpy as np
from scipy import stats
from scipy.stats import rankdata
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid


# ════════════════════════════════════════════════════════════════════════════════
# MIXTURE MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

def linear_gauss(x, w, slope, intercept, mu, sigma):
    """Normalised linear-component + Gaussian mixture PDF."""
    linear = np.clip(slope * x + intercept, 0, None)
    norm_factor = np.trapezoid(linear, x)
    if norm_factor > 0:
        linear /= norm_factor
    gauss = stats.norm.pdf(x, mu, sigma)
    return w * linear + (1 - w) * gauss


def double_gauss(x, w, mu1, s1, mu2, s2):
    """Two-component Gaussian mixture PDF."""
    return w * stats.norm.pdf(x, mu1, s1) + (1 - w) * stats.norm.pdf(x, mu2, s2)


# ════════════════════════════════════════════════════════════════════════════════
# CDF UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def pdf_to_cdf(xr, pdf_vals):
    """Numerically integrate a PDF array to a normalised CDF."""
    cdf = cumulative_trapezoid(pdf_vals, xr, initial=0)
    cdf /= cdf[-1]
    return cdf


def cdf_transform(data, xr, cdf_vals):
    """Map data to uniform [0,1] via interpolation onto a fitted CDF."""
    return np.interp(data, xr, cdf_vals)


# ════════════════════════════════════════════════════════════════════════════════
# FITTERS  (one per distribution family)
# ════════════════════════════════════════════════════════════════════════════════


def fit_pareto(data, n_bins=30):
    """
    Fit a Pareto PDF to `data` via MLE.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with b, loc, scale
    """
    b, loc, scale = stats.pareto.fit(data)

    xr       = np.linspace(0, data.max() + 0.5, 10000)
    pdf_vals = stats.pareto.pdf(xr, b, loc=loc, scale=scale)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(b=b, loc=loc, scale=scale)
    return u, xr, pdf_vals, cdf_vals, params


def fit_laplace(data):
    """
    Fit a Laplace PDF to `data` via MLE.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with loc, scale
    """
    loc, scale = stats.laplace.fit(data)

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10000)
    pdf_vals = stats.laplace.pdf(xr, loc=loc, scale=scale)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(loc=loc, scale=scale)
    return u, xr, pdf_vals, cdf_vals, params


def gauss_gennorm(x, w, mu, sigma, beta, gnorm_loc, gnorm_scale):
    """Gaussian + generalised-normal mixture PDF."""
    return (w       * stats.norm.pdf(x, mu, sigma) +
            (1 - w) * stats.gennorm.pdf(x, beta, loc=gnorm_loc, scale=gnorm_scale))

def double_gennorm(x, w, beta1, loc1, scale1, beta2, loc2, scale2):
    """Two-component generalised-normal mixture PDF."""
    return (w       * stats.gennorm.pdf(x, beta1, loc=loc1, scale=scale1) +
            (1 - w) * stats.gennorm.pdf(x, beta2, loc=loc2, scale=scale2))


def fit_gauss_gennorm(data, n_bins=30):
    """
    Fit a Gaussian + generalised-normal mixture PDF to `data`.

    The generalised-normal component is initialised to the right of the
    Gaussian and with a smaller scale. Its shape parameter beta is free:
      beta < 1  →  heavier than Laplace
      beta = 1  →  Laplace
      beta = 2  →  Gaussian

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with w, mu, sigma, beta, gnorm_loc, gnorm_scale
    """
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    bc = (edges[:-1] + edges[1:]) / 2

    mu0         = data.mean() - 0.3
    gnorm_loc0  = data.mean() + 0.5      # right of the Gaussian
    sig0        = data.std() * 0.8
    gnorm_scl0  = data.std() * 0.4       # narrower
    beta0       = 1.0                    # start at Laplace, let it roam

    p0     = [0.7,  mu0,   sig0,  beta0,  gnorm_loc0,  gnorm_scl0]
    bounds = ([0,   data.min(), 0.01,  0.1,  data.min(),  0.01],
              [1,   data.max(),  10,   10,   data.max(),   10 ])

    popt, _ = curve_fit(gauss_gennorm, bc, counts, p0=p0,
                        bounds=bounds, maxfev=20_000)
    w, mu, sigma, beta, gnorm_loc, gnorm_scale = popt

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10000)
    pdf_vals = gauss_gennorm(xr, *popt)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(w=w, mu=mu, sigma=sigma, beta=beta,
                  gnorm_loc=gnorm_loc, gnorm_scale=gnorm_scale)
    return u, xr, pdf_vals, cdf_vals, params


def fit_linear(data, n_bins=30):
    """
    Fit a decreasing linear PDF  f(x) = slope·x + intercept  to `data`.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with slope, intercept, x_min, x_max
    """
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    bc = (edges[:-1] + edges[1:]) / 2
    slope, intercept, *_ = stats.linregress(bc, counts)

    x_min, x_max = data.min(), data.max()
    C = -(0.5 * slope * x_min**2 + intercept * x_min)

    xr       = np.linspace(x_min, x_max, 10000)
    pdf_vals = np.clip(slope * xr + intercept, 0, None)
    cdf_vals = np.clip(0.5 * slope * xr**2 + intercept * xr + C, 0, 1)
    u        = np.interp(data, xr, cdf_vals)

    params = dict(slope=slope, intercept=intercept, x_min=x_min, x_max=x_max)
    return u, xr, pdf_vals, cdf_vals, params


def fit_linear_gauss(data, n_bins=30):
    """
    Fit a linear + Gaussian mixture PDF to `data`.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with w, slope, intercept, mu, sigma
    """
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    bc = (edges[:-1] + edges[1:]) / 2

    p0     = [0.5, -0.1, 1.0, data.mean(), data.std()]
    bounds = ([0, -10, -10, data.min(), 0.01],
              [1,  10,  10, data.max(), 10  ])

    popt, _ = curve_fit(linear_gauss, bc, counts, p0=p0,
                        bounds=bounds, maxfev=20_000)
    w, slope, intercept, mu, sigma = popt

    xr       = np.linspace(data.min(), data.max(), 10000)
    pdf_vals = linear_gauss(xr, *popt)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(w=w, slope=slope, intercept=intercept, mu=mu, sigma=sigma)
    return u, xr, pdf_vals, cdf_vals, params


def fit_double_gauss(data, n_bins=50, fixed_locs=None):
    """
    Fit a two-component Gaussian mixture PDF to `data`.

    Parameters
    ----------
    fixed_locs : tuple (mu1, mu2) or None
        If provided, mu1 and mu2 are held fixed and not optimised.
    """
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    bc = (edges[:-1] + edges[1:]) / 2

    if fixed_locs is not None:
        mu1_fixed, mu2_fixed = fixed_locs

        def double_gauss_fixed(x, w, s1, s2):
            return double_gauss(x, w, mu1_fixed, s1, mu2_fixed, s2)

        p0     = [0.5, data.std() * 0.5, data.std() * 0.5]
        bounds = ([0,   0.01, 0.01],
                  [1,   5,    5   ])

        popt, _ = curve_fit(double_gauss_fixed, bc, counts, p0=p0,
                            bounds=bounds, maxfev=20_000)
        w, s1, s2 = popt
        mu1, mu2  = mu1_fixed, mu2_fixed

    else:
        p0     = [0.5, data.mean() - 0.5, data.std() * 0.5,
                       data.mean() + 0.5, data.std() * 0.5]
        bounds = ([0,  -10, 0.01, -10, 0.01],
                  [1,   10,    5,  10,    5 ])

        popt, _ = curve_fit(double_gauss, bc, counts, p0=p0,
                            bounds=bounds, maxfev=20_000)
        w, mu1, s1, mu2, s2 = popt

    full_popt = (w, mu1, s1, mu2, s2)

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10_000)
    pdf_vals = double_gauss(xr, *full_popt)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)
    params   = dict(w=w, mu1=mu1, s1=s1, mu2=mu2, s2=s2)

    return u, xr, pdf_vals, cdf_vals, params


def fit_single_gauss(data):
    """
    Fit a single Gaussian PDF to `data` via MLE.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with mu, sigma
    """
    mu, sigma = stats.norm.fit(data)

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10000)
    pdf_vals = stats.norm.pdf(xr, mu, sigma)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(mu=mu, sigma=sigma)
    return u, xr, pdf_vals, cdf_vals, params


def fit_single_gennorm(data):
    """
    Fit a single generalized normal PDF to `data` via MLE.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with beta, loc, scale
    """
    beta, loc, scale = stats.gennorm.fit(data)
    # stats.gennorm.pdf(x, beta, loc=gnorm_loc, scale=gnorm_scale)

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10000)
    pdf_vals = stats.gennorm.pdf(xr, beta, loc=loc, scale=scale)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(beta=beta, loc=loc, scale=scale)
    return u, xr, pdf_vals, cdf_vals, params


def fit_double_gennorm(data, n_bins=30, fixed_locs=None):
    """
    Fit a two-component generalised-normal mixture PDF to `data`.

    Parameters
    ----------
    fixed_locs : tuple (loc1, loc2) or None
        If provided, loc1 and loc2 are held fixed at these values
        and not included in the optimisation.
    """
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    bc = (edges[:-1] + edges[1:]) / 2

    if fixed_locs is not None:
        loc1_fixed, loc2_fixed = fixed_locs

        # Wrapper that injects the fixed locs so curve_fit only sees 5 params
        def double_gennorm_fixed(x, w, beta1, scale1, beta2, scale2):
            return double_gennorm(x, w, beta1, loc1_fixed, scale1,
                                     beta2, loc2_fixed, scale2)

        p0     = [0.5, 1.0, data.std() * 0.5,
                       1.0, data.std() * 0.5]
        bounds = ([0,   0.1, 0.01, 0.1, 0.01],
                  [1,  10,   5,   10,   5  ])

        popt, _ = curve_fit(double_gennorm_fixed, bc, counts, p0=p0,
                            bounds=bounds, maxfev=20_000)
        w, beta1, scale1, beta2, scale2 = popt
        loc1, loc2 = loc1_fixed, loc2_fixed

    else:
        p0     = [0.5, 1.0, data.mean() - 0.5, data.std() * 0.5,
                       1.0, data.mean() + 0.5, data.std() * 0.5]
        bounds = ([0,   0.1, -10, 0.01, 0.1, -10, 0.01],
                  [1,  10,   10,    5,  10,  10,    5  ])

        popt, _ = curve_fit(double_gennorm, bc, counts, p0=p0,
                            bounds=bounds, maxfev=20_000)
        w, beta1, loc1, scale1, beta2, loc2, scale2 = popt

    # Reconstruct full popt tuple for pdf/cdf evaluation
    full_popt = (w, beta1, loc1, scale1, beta2, loc2, scale2)

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10_000)
    pdf_vals = double_gennorm(xr, *full_popt)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)
    params   = dict(w=w, beta1=beta1, loc1=loc1, scale1=scale1,
                        beta2=beta2, loc2=loc2, scale2=scale2)

    return u, xr, pdf_vals, cdf_vals, params


def fit_exponweib(data):
    """
    Fit an Exponentiated Weibull PDF to `data` via MLE.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with a, c, loc, scale
    """
    a, c, loc, scale = stats.exponweib.fit(data)

    xr       = np.linspace(0, data.max() + 0.5, 10000)
    pdf_vals = stats.exponweib.pdf(xr, a, c, loc=loc, scale=scale)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(a=a, c=c, loc=loc, scale=scale)
    return u, xr, pdf_vals, cdf_vals, params


def fit_moyal(data):
    """
    Fit a Moyal PDF to `data` via MLE.

    Returns
    -------
    u        : uniform [0,1] marginals
    xr       : evaluation grid
    pdf_vals : PDF evaluated on xr
    cdf_vals : CDF evaluated on xr
    params   : dict with loc, scale
    """
    loc, scale = stats.moyal.fit(data)

    xr       = np.linspace(data.min() - 0.5, data.max() + 0.5, 10000)
    pdf_vals = stats.moyal.pdf(xr, loc=loc, scale=scale)
    cdf_vals = pdf_to_cdf(xr, pdf_vals)
    u        = cdf_transform(data, xr, cdf_vals)

    params = dict(loc=loc, scale=scale)
    return u, xr, pdf_vals, cdf_vals, params


def report_fit(label, family, params):
    """Print a one-line summary of a fitted distribution."""

    if family == "linear":
        print(f"{label:25s}  linear({params['slope']:.3f}x + {params['intercept']:.3f})")

    elif family == "linear_gauss":
        p = params
        print(f"{label:25s}  w={p['w']:.3f}  "
              f"linear({p['slope']:.3f}x + {p['intercept']:.3f})  "
              f"N({p['mu']:.3f}, {p['sigma']:.3f})")
        
    elif family == "pareto":
        p = params
        print(f"{label:25s}  Pareto(b={p['b']:.3f}, loc={p['loc']:.3f}, scale={p['scale']:.3f})")

    elif family == "gauss_gennorm":
        p = params
        print(f"{label:25s}  w={p['w']:.3f}  "
              f"N({p['mu']:.3f}, {p['sigma']:.3f})  "
              f"GNorm(beta={p['beta']:.3f}, loc={p['gnorm_loc']:.3f}, scale={p['gnorm_scale']:.3f})")
        
    elif family == "double_gauss":
        p = params
        print(f"{label:25s}  w={p['w']:.3f}  "
              f"N({p['mu1']:.3f}, {p['s1']:.3f})  "
              f"N({p['mu2']:.3f}, {p['s2']:.3f})")
        
    elif family == "single_gauss":
        print(f"{label:25s}  N({params['mu']:.3f}, {params['sigma']:.3f})")

    elif family == "single_gennorm":
        print(f"{label:25s}  GNorm(beta={params['beta']:.3f}, loc={params['loc']:.3f}, scale={params['scale']:.3f})")

    elif family == "exponweib":
        print(f"{label:25s}  ExponWeib(a={params['a']:.3f}, c={params['c']:.3f}, loc={params['loc']:.3f}, scale={params['scale']:.3f})")

    elif family == "moyal":
        print(f"{label:25s}  Moyal(loc={params['loc']:.3f}, scale={params['scale']:.3f})")

    elif family == "double_gennorm":
        p = params
        print(f"{label:25s}  w={p['w']:.3f}  "
              f"GNorm(beta={p['beta1']:.3f}, loc={p['loc1']:.3f}, scale={p['scale1']:.3f})  "
              f"GNorm(beta={p['beta2']:.3f}, loc={p['loc2']:.3f}, scale={p['scale2']:.3f})")
        
    elif family == "laplace":
        print(f"{label:25s}  Laplace(loc={params['loc']:.3f}, scale={params['scale']:.3f})")

    

def fit_all_marginals(x_all, x_fn, y_all, y_fn):
    """
    Fit parametric marginal distributions to each variable/subset pair
    and return uniform [0,1] transforms alongside fit diagnostics.

    Parameters
    ----------
    x_all : array  –  magnitude for all objects
    x_fn  : array  –  magnitude for false-negative subset
    y_all : array  –  g−i colour for all objects
    y_fn  : array  –  g−i colour for false-negative subset

    Returns
    -------
    dict keyed by variable name, each containing
        'u'        – uniform marginals
        'xr'       – evaluation grid   (None for linear fit)
        'pdf_vals' – PDF on grid        (callable for linear fit)
        'cdf_vals' – CDF on grid        (callable for linear fit)
        'params'   – fitted parameters
    """
    results = {}
    upper_clip = 1 -1e-10

    lower_clip = 1e-10
    # 1. x_all  →  linear PDF
    #u, xr, pdf_vals, cdf_vals, params = fit_linear(x_all)
    #report_fit("x_all", "linear", params)
    u, xr, pdf_vals, cdf_vals, params = fit_exponweib(x_all)
    report_fit("x_all", "exponweib", params)

    # if u is greater than 1, raise error
    if np.any(u > 1):
        raise ValueError("CDF values exceed 1, check fit and interpolation")
    if np.any(u < 0):
        raise ValueError("CDF values below 0, check fit and interpolation")
    
    # if u or v = 1, set to 0.999 to avoid issues with copula fitting
    u = np.clip(u, lower_clip, upper_clip)
    
    results["x_all"] = dict(x=x_all, u=u, xr=xr, pdf_vals=pdf_vals,
                            cdf_vals=cdf_vals, params=params)

    ## 2. x_fn  →  linear + Gaussian mixture
    #u, xr, pdf_vals, cdf_vals, params = fit_linear_gauss(x_fn)
    #report_fit("x_fn", "linear_gauss", params)
    #results["x_fn"] = dict(x=x_fn, u=u, xr=xr, pdf_vals=pdf_vals,
    #                       cdf_vals=cdf_vals, params=params)
    
    # 2. x_fn  →  Pareto
    u, xr, pdf_vals, cdf_vals, params = fit_pareto(x_fn)
        # if u is greater than 1, raise error
    # if u is greater than 1, raise error
    if np.any(u > 1):
        raise ValueError("CDF values exceed 1, check fit and interpolation")
    if np.any(u < 0):
        raise ValueError("CDF values below 0, check fit and interpolation")
    
    # if u or v = 1, set to 0.999 to avoid issues with copula fitting
    u = np.clip(u, lower_clip, upper_clip)

    report_fit("x_fn", "pareto", params)
    results["x_fn"] = dict(x=x_fn, u=u, xr=xr, pdf_vals=pdf_vals,
                           cdf_vals=cdf_vals, params=params)


    # 3. y_all  →  Gaussian + generalised-normal mixture
    u, xr, pdf_vals, cdf_vals, params = fit_double_gauss(y_all)
    #u, xr, pdf_vals, cdf_vals, params = fit_moyal(y_all)
    #u, xr, pdf_vals, cdf_vals, params = fit_double_gennorm(y_all)
    if np.any(u > 1):
        raise ValueError("CDF values exceed 1, check fit and interpolation")
    if np.any(u < 0):
        raise ValueError("CDF values below 0, check fit and interpolation")
    
    # if u or v = 1, set to 0.999 to avoid issues with copula fitting
    u = np.clip(u, lower_clip, upper_clip)

    # 3. y_all  →  Gaussian + generalised-normal mixture
    #report_fit("y_all", "gauss_gennorm", params)
    #report_fit("y_all", "moyal", params)
    report_fit("y_all", "double_gauss", params)

    results["y_all"] = dict(x=y_all, u=u, xr=xr, pdf_vals=pdf_vals,
                            cdf_vals=cdf_vals, params=params)
    
    # 4. y_fn  →  single Gaussian
    #u, xr, pdf_vals, cdf_vals, params = fit_single_gauss(y_fn)
    #u, xr, pdf_vals, cdf_vals, params = fit_single_gennorm(y_fn)
    # 4. y_fn  →  double gennorm, inheriting loc1/loc2 from y_all fit
    inherited_locs = (params["mu1"], params["mu2"])
    u, xr, pdf_vals, cdf_vals, params = fit_double_gauss(
        y_fn, fixed_locs=inherited_locs
    )
    # ... validation/clipping ...
    report_fit("y_fn", "double_gauss (fixed locs)", params)
    results["y_fn"] = dict(x=y_fn, u=u, xr=xr, pdf_vals=pdf_vals,
                           cdf_vals=cdf_vals, params=params)

    return results


#### EMPIRCAL TRANSFORMS


def compute_histogram_pdf(data, n_bins):
    """
    Histogram PDF in original space with denser bins near zero.

    Parameters
    ----------
    zero_frac  : fraction of bins allocated to the [x_min, percentile_split] region
    zero_bins  : if set, overrides zero_frac for the number of fine bins near zero
    """
    data = np.asarray(data)
    if np.max(data) > 3.5:
        bins = np.linspace(0, 6., n_bins)
    else:
        bins = np.linspace(0, 2., n_bins)

    counts, edges = np.histogram(data, bins=bins, density=True)
    xr = (edges[:-1] + edges[1:]) / 2

    # density=True already normalises by bin width, so counts IS the pdf
    return xr, counts


def compute_kde_pdf(data, x_min=0, n_points=200, bw_method='scott'):
    """
    KDE-based PDF estimate in original space, evaluated on a grid.

    Parameters
    ----------
    bw_method : bandwidth selector — 'scott', 'silverman', or a float scalar
    """
    data = np.asarray(data)
    data = data[data >= x_min]
    # quick hack to do the relcection trick for x vals and not y vals:

    if data.max() > 2.5:
        data_reflected = np.concatenate([data, 2 * x_min - data])
        kde = gaussian_kde(data_reflected, bw_method=bw_method)

    else:
        kde = gaussian_kde(data, bw_method=bw_method)
    # Evaluate only on x >= x_min and double the density to compensate
    #pdf_vals = 2 * kde(xr)

    #kde = gaussian_kde(data, bw_method=bw_method)

    # Grid: denser near zero, sparser in the tail
    xr = np.concatenate([
        np.linspace(x_min, np.percentile(data, 20), n_points // 2),
        np.linspace(np.percentile(data, 20), data.max(), n_points // 2)
    ])
    xr = np.unique(xr)

    #pdf_vals = kde(xr)
    if data.max() > 2.5:
        pdf_vals = 2 * kde(xr)
    else:
        pdf_vals = kde(xr)
    return xr, pdf_vals


def fit_empirical(data, n_bins=25, x_min=0,
                  use_kde=True, kde_bw='scott', kde_points=200):
    data = np.asarray(data)
    data = data[data >= x_min]

    # ECDF
    xr_cdf = np.sort(data)
    n = len(xr_cdf)
    cdf_vals_full = np.arange(1, n + 1) / n
    u = np.interp(data, xr_cdf, cdf_vals_full)

    # PDF
    if use_kde:
        xr, pdf_vals = compute_kde_pdf(
            data, x_min=x_min, n_points=kde_points, bw_method=kde_bw
        )
    else:
        xr, pdf_vals = compute_histogram_pdf(
            data, n_bins=n_bins)

    # Interpolate CDF onto the PDF grid
    cdf_vals = np.interp(xr, xr_cdf, cdf_vals_full, left=0, right=1)

    params = dict(x_min=xr[0], x_max=xr[-1])
    return u, xr, pdf_vals, cdf_vals, params


def fit_all_marginals_empirical(x_all, x_fn, y_all, y_fn):
    """
    Compute empirical marginal CDFs for each variable/subset pair
    and return uniform [0,1] transforms.

    Parameters
    ----------
    x_all : array  –  magnitude for all objects
    x_fn  : array  –  magnitude for false-negative subset
    y_all : array  –  g−i colour for all objects
    y_fn  : array  –  g−i colour for false-negative subset

    Returns
    -------
    dict keyed by variable name, each containing
        'u'        – uniform [0,1] marginals
        'xr'       – sorted data values (ECDF x-axis)
        'pdf_vals' – None
        'cdf_vals' – empirical CDF values on xr
        'params'   – dict with x_min, x_max
    """
    results = {}

    for label, data in [("x_all", x_all), ("x_fn", x_fn),
                        ("y_all", y_all), ("y_fn",  y_fn)]:
        u, xr, pdf_vals, cdf_vals, params = fit_empirical(data)
        print(f"{label:25s}  empirical ECDF  "
              f"[{params['x_min']:.3f}, {params['x_max']:.3f}]  "
              f"n={len(data)}")
        results[label] = dict(x=data, u=u, xr=xr, pdf_vals=pdf_vals,
                              cdf_vals=cdf_vals, params=params)

    return results


# ════════════════════════════════════════════════════════════════════════════════
# INVERSE CDF  (uniform → real space)
# ════════════════════════════════════════════════════════════════════════════════

def inverse_cdf_numerical(u, xr, cdf_vals):
    """Map u in [0,1] back to x via interpolation of the numerical CDF."""
    return np.interp(np.asarray(u), cdf_vals, xr)


def inverse_cdf_linear(u, params):
    """
    Analytic inverse of F(x) = 0.5*slope*x^2 + intercept*x + C.
    Solves the quadratic for x given u in [0,1].
    """
    slope     = params['slope']
    intercept = params['intercept']
    x_min     = params['x_min']
    x_max     = params['x_max']
    C         = -(0.5 * slope * x_min**2 + intercept * x_min)

    u = np.asarray(u)
    a_coef = 0.5 * slope
    b_coef = intercept
    c_coef = C - u

    if abs(a_coef) < 1e-12:           # degenerate: nearly flat, linear solve
        return np.clip(-c_coef / b_coef, x_min, x_max)

    discriminant = np.clip(b_coef**2 - 4 * a_coef * c_coef, 0, None)
    x1 = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)
    x2 = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef)
    candidates = np.where(
        (x1 >= x_min - 1e-6) & (x1 <= x_max + 1e-6), x1, x2
    )
    return np.clip(candidates, x_min, x_max)


def inverse_cdf_single_gauss(u, params):
    """Analytic inverse (PPF) of a single Gaussian."""
    return stats.norm.ppf(np.asarray(u), loc=params['mu'], scale=params['sigma'])


def inverse_cdf_single_gennorm(u, params):
    """Analytic inverse (PPF) of a single generalised normal."""
    return stats.gennorm.ppf(np.asarray(u), beta=params['beta'], loc=params['loc'], scale=params['scale'])


def inverse_cdf_exponweib(u, params):
    """Analytic inverse (PPF) of an Exponentiated Weibull."""
    return stats.exponweib.ppf(np.asarray(u), a=params['a'], c=params['c'], loc=params['loc'], scale=params['scale'])


def inverse_cdf_moyal(u, params):
    """Analytic inverse (PPF) of a Moyal distribution."""
    return stats.moyal.ppf(np.asarray(u), loc=params['loc'], scale=params['scale'])


def inverse_cdf_laplace(u, params):
    """Analytic inverse (PPF) of a Laplace distribution."""
    return stats.laplace.ppf(np.asarray(u), loc=params['loc'], scale=params['scale'])


def invert_cdf(u, key, pdf_transformations):
    """
    Dispatch inverse CDF for any key in pdf_transformations.
    Uses analytic inverse where available, numerical otherwise.
    """
    res = pdf_transformations[key]
    if key == 'x_all':
        #return inverse_cdf_linear(u, res['params'])
        return inverse_cdf_exponweib(u, res['params'])
    elif key == 'y_fn':
        #return inverse_cdf_single_gauss(u, res['params'])
        #return inverse_cdf_single_gennorm(u, res['params'])
        return inverse_cdf_numerical(u, res['xr'], res['cdf_vals'])
    #elif key == 'y_all':
    #    return inverse_cdf_moyal(u, res['params'])
    else:
        return inverse_cdf_numerical(u, res['xr'], res['cdf_vals'])


# ════════════════════════════════════════════════════════════════════════════════
# FORWARD CDF  (real space → uniform [0,1])
# ════════════════════════════════════════════════════════════════════════════════

def forward_cdf_linear(x, params):
    """Analytic CDF for the linear fit."""
    slope     = params['slope']
    intercept = params['intercept']
    x_min     = params['x_min']
    C         = -(0.5 * slope * x_min**2 + intercept * x_min)
    return np.clip(0.5 * slope * x**2 + intercept * x + C, 0, 1)


def forward_cdf_numerical(x, xr, cdf_vals):
    """Map x in real space to u in [0,1] via interpolation of the numerical CDF."""
    return np.interp(np.asarray(x), xr, cdf_vals, left=0, right=1)


def forward_cdf_single_gauss(x, params):
    """Analytic CDF of a single Gaussian."""
    return stats.norm.cdf(np.asarray(x), loc=params['mu'], scale=params['sigma'])


def forward_cdf_single_gennorm(x, params):
    """Analytic CDF of a single generalised normal."""
    return stats.gennorm.cdf(np.asarray(x), beta=params['beta'], loc=params['loc'], scale=params['scale'])


def forward_cdf_exponweib(x, params):
    """Analytic CDF of an Exponentiated Weibull."""
    return stats.exponweib.cdf(np.asarray(x), a=params['a'], c=params['c'], loc=params['loc'], scale=params['scale'])


def forward_cdf_moyal(x, params):
    """Analytic CDF of a Moyal distribution."""
    return stats.moyal.cdf(np.asarray(x), loc=params['loc'], scale=params['scale'])


def forward_cdf_laplace(x, params):
    """Analytic CDF of a Laplace distribution."""
    return stats.laplace.cdf(np.asarray(x), loc=params['loc'], scale=params['scale'])


def forward_cdf(x, key, pdf_transformations, model_type = 'parametric'):
    """
    Dispatch forward CDF for any key in pdf_transformations.
    Uses analytic CDF where available, numerical otherwise.
    """
    if model_type not in ['parametric', 'empirical']:
        raise ValueError("model_type must be either 'parametric' or 'empirical'")
    
    res = pdf_transformations[key]
    if model_type == 'parametric':
        if key == 'x_all':
            #return forward_cdf_linear(x, res['params'])
            return forward_cdf_exponweib(x, res['params'])
        elif key == 'y_fn':
            #return forward_cdf_single_gauss(x, res['params'])
            #return forward_cdf_single_gennorm(x, res['params'])
            return forward_cdf_numerical(x, res['xr'], res['cdf_vals'])
     #   elif key == 'y_all':
     #       return forward_cdf_moyal(x, res['params'])
        else:
            return forward_cdf_numerical(x, res['xr'], res['cdf_vals'])
    elif model_type == 'empirical':
        return forward_cdf_numerical(x, res['xr'], res['cdf_vals'])


# ════════════════════════════════════════════════════════════════════════════════
# PDF  (density evaluation)
# ════════════════════════════════════════════════════════════════════════════════

def density_numerical(x, xr, pdf_vals):
    """Interpolate density at x from a precomputed PDF array."""
    return np.interp(np.asarray(x), xr, pdf_vals, left=0, right=0)


def density_numerical_xs(x, xr, pdf_vals):
    return np.interp(np.asarray(x), xr, pdf_vals, right=0)


def density_numerical_ys(x, xr, pdf_vals):
    """Interpolate density at x from a precomputed PDF array."""
    return np.interp(np.asarray(x), xr, pdf_vals, left=0, right=0)


def density_linear(x, params):
    """Analytic linear PDF."""
    return np.clip(params['slope'] * np.asarray(x) + params['intercept'], 0, None)


def density_single_gauss(x, params):
    """Analytic single Gaussian PDF."""
    return stats.norm.pdf(np.asarray(x), loc=params['mu'], scale=params['sigma'])


def density_single_gennorm(x, params):
    """Analytic single generalised normal PDF."""
    return stats.gennorm.pdf(np.asarray(x), beta=params['beta'], loc=params['loc'], scale=params['scale'])


def density_exponweib(x, params):
    """Analytic Exponentiated Weibull PDF."""
    return stats.exponweib.pdf(np.asarray(x), a=params['a'], c=params['c'], loc=params['loc'], scale=params['scale'])


def density_moyal(x, params):
    """Analytic Moyal PDF."""
    return stats.moyal.pdf(np.asarray(x), loc=params['loc'], scale=params['scale'])
                                                                          

def density_laplace(x, params):
    """Analytic Laplace PDF."""
    return stats.laplace.pdf(np.asarray(x), loc=params['loc'], scale=params['scale'])
                                                                         

def empirical_cdf(x):
    """Map data to uniform [0,1] via empirical CDF (rank-based)."""
    return rankdata(np.asarray(x)) / (len(x) + 1)


def density(x, key, pdf_transformations, model_type='parametric'):
    """
    Dispatch PDF evaluation for any key in pdf_transformations.
    Uses analytic PDF where available, numerical otherwise.
    """
    if model_type not in ['parametric', 'empirical']:
        raise ValueError("model_type must be either 'parametric' or 'empirical'")
    res = pdf_transformations[key]
    if model_type == 'parametric':
        if key == 'x_all':
            #return density_linear(x, res['params'])
            return density_exponweib(x, res['params'])
        elif key == 'y_fn':
            #return density_single_gauss(x, res['params'])
            #return density_single_gennorm(x, res['params'])
            return density_numerical(x, res['xr'], res['pdf_vals'])
        #elif key == 'y_all':
        #    return density_moyal(x, res['params'])
        else:
            return density_numerical(x, res['xr'], res['pdf_vals'])
        

    elif model_type == 'empirical':
        if key in ['x_all', 'x_fn']:
            return density_numerical_xs(x, res['xr'], res['pdf_vals'])
        elif key in ['y_all', 'y_fn']:
            return density_numerical_ys(x, res['xr'], res['pdf_vals'])


def xy2xy_parameteric_cdf_transform(xy, pdf_transformations):
    """
    Transform a new (x, y) dataset into the g09 marginal space.

    Applies an empirical CDF to each marginal of `xy`, then maps
    through the inverse CDFs of x_all (linear) and y_all (double Gaussian).

    Parameters
    ----------
    xy      : array of shape (n, 2)
    pdf_transformations : dict returned by fit_all_marginals()

    Returns
    -------
    array of shape (n, 2) in the g09 marginal space
    """
    x, y = xy[:, 0], xy[:, 1]

    u = empirical_cdf(x)
    v = empirical_cdf(y)

    x_transformed = invert_cdf(u, 'x_all', pdf_transformations)
    y_transformed = invert_cdf(v, 'y_all', pdf_transformations)

    return np.column_stack([x_transformed, y_transformed])


def empirical_cdf_transform(values, reference):
        ref_sorted = np.sort(reference)
        probs = np.arange(1, len(reference) + 1) / (len(reference) + 1)
        # Forward: map input values to CDF probabilities using reference distribution
        u = np.interp(values, ref_sorted, probs)
        # Inverse: map those probabilities back to the reference domain (quantile transform)
        x_transformed = np.interp(u, probs, ref_sorted)
        return x_transformed

def xy2xy_empirical_cdf_transform(xy_input, xy_original):
    x = empirical_cdf_transform(xy_input[:, 0], xy_original[:, 0])
    y = empirical_cdf_transform(xy_input[:, 1], xy_original[:, 1])
    return np.column_stack([x, y])