import numpy as np
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import itertools
from scipy.special import logsumexp
from scipy import stats
from matplotlib.colors import LogNorm
import pandas as pd
from probability_funcs import forward_cdf, density, xy2xy_parameteric_cdf_transform, xy2xy_empirical_cdf_transform



def train_copulas_empirical(uv_all, uv_fn, make_plots=False):

    controls_np = pv.FitControlsBicop(family_set=[pv.BicopFamily.tll])
    cop_np_all = pv.Bicop.from_data(data=uv_all, controls=controls_np)
    cop_np_fn  = pv.Bicop.from_data(data=uv_fn,  controls=controls_np)
    print('Empirical TLL copula fits:')
    print(f'All: {cop_np_all}')
    print(f'FN: {cop_np_fn}')

    if make_plots:
        print('All Fits...')
        cop_np_all.plot()
        cop_np_all.plot(type="contour", margin_type="unif")
        print(f'All AIC: {cop_np_all.aic()}')
        print(f'All loglik: {cop_np_all.loglik()}')
        print('FN Fits...')
        cop_np_fn.plot()
        cop_np_fn.plot(type="contour", margin_type="unif")
        print(f'FN AIC: {cop_np_fn.aic()}')
        print(f'FN loglik: {cop_np_fn.loglik()}')

    return cop_np_all, cop_np_fn


def train_copulas_parametric(uv_all, uv_fn, make_plots=False, 
                             twocomponent_mixture_all= False, fam1_copula_all = pv.BicopFamily.bb8, fam2_copula_all = pv.BicopFamily.tawn,
                             twocomponent_mixture_fn=False, fam1_copula_fn=pv.BicopFamily.joe, fam2_copula_fn=pv.BicopFamily.tawn):
    """Fit parametric copula models to the "all" and "fn" datasets.
    For "all", we fit a single parametric copula, selecting the best family and rotation by AIC.
    For "fn", we fit either a single parametric copula (selecting by AIC) or a two-component mixture of copulas (selecting by AIC).
    Returns the fitted copula models for "all" and "fn".
    Parameters
    ----------
        uv_all: array-like of shape (n_samples, 2)
            The input data for the "all" dataset, transformed to uniform margins
        uv_fn: array-like of shape (n_samples, 2)
            The input data for the "fn" dataset, transformed to uniform margins
        make_plots: bool, default False
            Whether to generate diagnostic plots of the fitted copula densities
        twocomponent_mixture_fn: bool, default True
            Whether to fit a two-component mixture model for the "fn" dataset instead of a single copula
        fam1_copula_fn: pyvinecopulib BicopFamily, default pv.BicopFamily.joe
            The family to use for component 1 of the mixture model (if twocomponent_mixture_fn is True)
        fam2_copula_fn: pyvinecopulib BicopFamily, default pv.BicopFamily.tawn
            The family to use for component 2 of the mixture model (if twocomponent_mixture_fn is True)
    Returns
    -------
        cop_all: fitted pyvinecopulib Bicop model for the "all" dataset
        cop_fn: fitted pyvinecopulib Bicop model (or TwoComponentVinecopulibMixture) for the "fn" dataset
    """ 
    controls_fn = pv.FitControlsBicop(family_set=[pv.BicopFamily.bb8], allow_rotations=True)# pv.parametric
    controls_all = pv.FitControlsBicop(family_set=[pv.BicopFamily.bb8], allow_rotations=True) #pv.parametric

    if twocomponent_mixture_all:
        cop_all = TwoComponentVinecopulibMixture(family1=fam1_copula_all, family2=fam2_copula_all)
        cop_all.fit(uv_all)

    else:
    # Tawn looks OK.
        cop_all = pv.Bicop.from_data(data=uv_all, controls=controls_all)

    if twocomponent_mixture_fn:
        #cop_fn = TwoComponentVinecopulibMixture(family1=fam1_copula_fn, family2=fam2_copula_fn)
        cop_fn = TwoComponentVinecopulibMixture(family1=fam1_copula_fn, family2=fam2_copula_fn)
        cop_fn.fit(uv_fn)
    else:
        cop_fn  = pv.Bicop.from_data(data=uv_fn,  controls=controls_fn)

    print('Parametric copula fits:')
    print(f'All: {cop_all}')
    print(f'FN: {cop_fn}')
    if make_plots:
        print('All Fits...')
        if twocomponent_mixture_all:
            cop_all.plot(plot_type="heatmap", margin_type="unif")
        else:
            cop_all.plot(type="contour", margin_type="unif")
            cop_all.plot()
        print(f'All AIC: {cop_all.aic()}')
        print(f'All loglik: {cop_all.loglik()}')



        print('FN Fits...')

        if twocomponent_mixture_fn:
            cop_fn.plot(plot_type="heatmap", margin_type="unif")
        else:
            cop_fn.plot(type="contour", margin_type="unif")
            cop_fn.plot()

        print(f'FN AIC: {cop_fn.aic()}')
        print(f'FN loglik: {cop_fn.loglik()}')

    return cop_all, cop_fn


def get_completeness(xy_input, copula_all, copula_fn, pi_fn, pdf_transformations, apply_xy2xy_transform=False, cdf_type='parametric', mapping_type='parametric'):
    """
    Compute completeness for the input data using the trained copula models and marginal transformations.
    Parameters
    ----------
        xy_input: array-like of shape (n_samples, 2)
            The input data points for which to compute completeness, in the original (x,y) space
        copula_all: fitted copula pyvinecopulib model for the "all" dataset"
        copula_fn: fitted copula pyvinecopulib model for the "fn" dataset
        pi_fn: scaling value for the "fn" dataset (scalar)
        pdf_transformations: dict containing the fitted marginal transformations for x and y, e.g. as returned by fit_all_marginals()
        apply_xy2xy_transform: whether to apply the xy2xy transform to the input data before computing completeness
        cdf_type: 'parametric' or 'empirical', indicating which type of marginal CDF to use for the forward transformation
        mapping_type: 'parametric' or 'empirical', indicating which type of xy2xy mapping to apply (if apply_xy2xy_transform is True)
    
    Returns
    -------
        completeness: array of shape (n_samples,) with values in [0,1] representing the estimated completeness for each input point
    """
    if cdf_type not in ['parametric', 'empirical']:
        raise ValueError("cdf_type must be 'parametric' or 'empirical'")
    if mapping_type not in ['parametric', 'empirical']:
        raise ValueError("mapping_type must be 'parametric' or 'empirical'")

    if mapping_type == 'empirical':
        xy_original = np.column_stack((pdf_transformations['x_all']['x'], pdf_transformations['y_all']['x']))

        if apply_xy2xy_transform:
            xy_input = xy2xy_empirical_cdf_transform(xy_input, xy_original)

    elif mapping_type == 'parametric':
        if apply_xy2xy_transform:
            xy_input = xy2xy_parameteric_cdf_transform(xy_input, pdf_transformations)
    
    
    u_input_all = forward_cdf(xy_input[:,0], 'x_all', pdf_transformations, model_type=cdf_type)
    v_input_all = forward_cdf(xy_input[:,1], 'y_all', pdf_transformations, model_type=cdf_type)
    uv_input_all = np.column_stack((u_input_all, v_input_all))


    u_input_fn = forward_cdf(xy_input[:,0], 'x_fn', pdf_transformations, model_type=cdf_type)
    v_input_fn = forward_cdf(xy_input[:,1], 'y_fn', pdf_transformations, model_type=cdf_type)
    uv_input_fn = np.column_stack((u_input_fn, v_input_fn))

    u_all = copula_all.pdf(uv_input_all)
    u_fn = copula_fn.pdf(uv_input_fn)

    den_x_all = density(xy_input[:, 0], 'x_all', pdf_transformations, model_type=cdf_type)
    den_x_fn = density(xy_input[:, 0], 'x_fn', pdf_transformations, model_type=cdf_type)

    den_y_all = density(xy_input[:, 1], 'y_all', pdf_transformations, model_type=cdf_type)
    den_y_fn = density(xy_input[:, 1], 'y_fn', pdf_transformations, model_type=cdf_type)

    den_fn = u_fn * den_x_fn * den_y_fn
    den_all = u_all * den_x_all * den_y_all
    # Compute densities at the recovered points

    completeness = 1 - den_fn * pi_fn  / (den_all + 1e-12)
    # Clip completeness to [0,1]
    completeness = np.clip(completeness, 0, 1)
    return completeness


def _valid_rotations_for_family(family):
    """
    Return the rotations we allow for a given family.
    Symmetric families (Gaussian, Student, Frank, Independence) only support 0.
    All others get 0 and 180.
    """

    _ROTATION_ZERO_ONLY = {
        pv.BicopFamily.indep,
        pv.BicopFamily.gaussian,
        pv.BicopFamily.student,
        pv.BicopFamily.frank,
    }
    if family in _ROTATION_ZERO_ONLY:
        return [0]
    return [0, 90, 180, 270]


def _fit_bicop_fixed_family_rotation(u, family, rotation, weights=None):
    """
    Fit a Bicop with a fixed family AND fixed rotation.

    Uses Bicop.from_family() to lock in the rotation, then calls .fit()
    with allow_rotations=False so pyvinecopulib cannot change it.
    """
    controls = pv.FitControlsBicop(
        family_set=[family],
        parametric_method="mle",
        allow_rotations=False,
        weights=np.asarray(weights, dtype=float) if weights is not None else np.array([])
    )

    # Instantiate with the locked rotation, then fit in-place
    cop = pv.Bicop.from_family(family=family, rotation=rotation)
    cop.fit(data=np.asfortranarray(u), controls=controls)
    return cop


def _safe_log_pdf(copula, u, eps=1e-300):
    d = np.asarray(copula.pdf(u), dtype=float)
    return np.log(np.maximum(d, eps))


class TwoComponentVinecopulibMixture:
    """
    Two-component finite mixture of pyvinecopulib Bicop models.

        c(u, v) = w * c1(u, v; family1, rot1) + (1 - w) * c2(u, v; family2, rot2)

    Rotations are fixed at initialisation and never change during EM.
    Component 1 always uses rot1, component 2 always uses rot2.

    Initialisations are structured:
      - n_init must be a multiple of 2
      - Half of them use rotation pair (0, 0), the other half use (0, 180)
        [or (0, 0) twice if family2 only supports rotation 0]
      - Within each rotation pair, initial weights are drawn from init_weights
        (default: [0.25, 0.5, 0.75]), cycling if n_init > len(init_weights) * 2

    Default n_init=6 gives:
      rotation (0,0)   x weights [0.25, 0.5, 0.75]  → 3 inits
      rotation (0,180) x weights [0.25, 0.5, 0.75]  → 3 inits

        n_init=12,
        max_iter=200,
        tol=1e-2,
        criterion="loglik",
        init_weights=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        random_state=1
    """

    def __init__(
        self,
        family1= None,
        family2 = None,
        n_init=10,
        max_iter=200,
        tol=1e-3,
        min_weight=0.01,
        init_weights=[0.1, 0.25, 0.5, 0.75, 0.9],
        random_state=None, 
        criterion="aic",
    ):
        if n_init % 2 != 0:
            raise ValueError(f"n_init must be a multiple of 2, got {n_init}")

        self.family1 = family1
        self.family2 = family2
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.min_weight = min_weight
        self.init_weights = init_weights if init_weights is not None else [0.25, 0.5, 0.75]
        self.random_state = np.random.default_rng(random_state)
        self.criterion = criterion

        self.weight_ = None
        self.copula1_ = None
        self.copula2_ = None
        self.rotation1_ = None
        self.rotation2_ = None
        self.loglik_ = -np.inf
        self.aic_ = np.inf
        self.bic_ = np.inf
        self.converged_ = False
        self.n_iter_ = 0

    def _build_init_schedule(self):
        """
        Build the list of (w_init, rot1, rot2) for all n_init initialisations.

        n_init/2 use rotation pair (0, 0).
        n_init/2 use rotation pair (0, 180) — falling back to (0, 0) if
        family2 doesn't support 180.

        Initial weights cycle through self.init_weights within each half.
        """
        #valid2 = _valid_rotations_for_family(self.family2)
        #rot_pairs = [
        #    (0, 0),
        #    (0, 180 if 180 in valid2 else 0),
        #]

        valid1 = _valid_rotations_for_family(self.family1)
        valid2 = _valid_rotations_for_family(self.family2)

        rot_pairs = [(r1, r2) for r1 in valid1 for r2 in valid2]

        half = self.n_init // 2
        schedule = []
        for rot1, rot2 in rot_pairs:
            for i in range(half):
                w = self.init_weights[i % len(self.init_weights)]
                schedule.append((w, rot1, rot2))
        return schedule

    def fit(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be n x 2")
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("u must lie strictly inside (0, 1). Use pseudo_obs().")
        if np.any(u == 0) or np.any(u == 1):
            # drop points exactly on the boundary, or nans will occur in logpdf and break the EM algorithm.
            u = u[~np.any(u == 0, axis=1) & ~np.any(u == 1, axis=1)]

            #u = np.clip(u, 1e-12, 1 - 1e-12)

        n = u.shape[0]
        u_f = np.asfortranarray(u)  # pyvinecopulib prefers Fortran order
        best = None

        schedule = self._build_init_schedule()

        for (w_init, rot1, rot2) in schedule:
            # Initialise responsibilities from fixed w_init
            r1 = self.random_state.beta(2, 2, size=n)
            r1 = w_init * r1 / np.mean(r1)
            r1 = np.clip(r1, 1e-3, 1 - 1e-3)
            r2 = 1.0 - r1
            old_ll = -np.inf
            converged = False
            for it in range(1, self.max_iter + 1):
                # --- M-step: refit with current responsibilities ---
                w = np.clip(np.mean(r1), self.min_weight, 1.0 - self.min_weight)
                try:
                    c1 = _fit_bicop_fixed_family_rotation(u_f, self.family1, rot1, weights=r1)
                    c2 = _fit_bicop_fixed_family_rotation(u_f, self.family2, rot2, weights=r2)
                except Exception:
                    break

                # --- Evaluate log-likelihood of new parameters ---
                log_d1 = np.log(w) + _safe_log_pdf(c1, u_f)
                log_d2 = np.log(1.0 - w) + _safe_log_pdf(c2, u_f)
                log_den = logsumexp(np.column_stack([log_d1, log_d2]), axis=1)
                ll = float(np.sum(log_den))

                # --- Convergence check ---
                if abs(ll - old_ll) < self.tol:
                    converged = True
                    break
                old_ll = ll

                # --- E-step: true posteriors, no clipping ---
                r1 = np.exp(log_d1 - log_den)
                r2 = 1.0 - r1
                # w is the only thing clipped, responsibilities are left as true posteriors

            # Evaluate final ll with current w/c1/c2
            w = np.clip(np.mean(r1), self.min_weight, 1.0 - self.min_weight)
            log_d1 = np.log(w) + _safe_log_pdf(c1, u_f)
            log_d2 = np.log(1.0 - w) + _safe_log_pdf(c2, u_f)
            ll = float(np.sum(logsumexp(np.column_stack([log_d1, log_d2]), axis=1)))

            k = int(c1.npars + c2.npars + 1)
            aic = 2 * k - 2 * ll
            bic = np.log(n) * k - 2 * ll

            candidate = {
                "weight": w,
                "copula1": c1,
                "copula2": c2,
                "rotation1": rot1,
                "rotation2": rot2,
                "loglik": ll,
                "aic": aic,
                "bic": bic,
                "converged": converged,
                "n_iter": it,
                "npars": k,
            }

            if best is None or candidate["loglik"] > best["loglik"]:
                best = candidate

        if best is None:
            raise RuntimeError(
                f"All initialisations failed for {self.family1}+{self.family2}"
            )

        self.weight_ = best["weight"]
        self.copula1_ = best["copula1"]
        self.copula2_ = best["copula2"]
        self.rotation1_ = best["rotation1"]
        self.rotation2_ = best["rotation2"]
        self.loglik_ = best["loglik"]
        self.aic_ = best["aic"]
        self.bic_ = best["bic"]
        self.converged_ = best["converged"]
        self.n_iter_ = best["n_iter"]
        self.npars_ = best["npars"]
        return self

    def pdf(self, u):
        u = np.asfortranarray(np.asarray(u, dtype=float))
        return (
            self.weight_ * self.copula1_.pdf(u)
            + (1.0 - self.weight_) * self.copula2_.pdf(u)
        )

    def logpdf(self, u):
        return np.log(np.maximum(self.pdf(u), 1e-300))

    def summary(self):
        return {
            "family1": str(self.family1),
            "family2": str(self.family2),
            "rotation1": self.rotation1_,
            "rotation2": self.rotation2_,
            "weight1": self.weight_,
            "weight2": 1.0 - self.weight_,
            "copula1": str(self.copula1_),
            "copula2": str(self.copula2_),
            "loglik": self.loglik_,
            "aic": self.aic_,
            "bic": self.bic_,
            "npars": self.npars_,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
        }
    
    def select(self, u, families_list='all'):
        if families_list == 'all':
            families_list = [
                pv.BicopFamily.bb1, pv.BicopFamily.bb6, pv.BicopFamily.bb7,
                pv.BicopFamily.bb8, pv.BicopFamily.clayton, pv.BicopFamily.frank,
                pv.BicopFamily.gaussian, pv.BicopFamily.gumbel, pv.BicopFamily.joe,
                pv.BicopFamily.tawn, pv.BicopFamily.student
            ]

        rows = []
        for fam1, fam2 in itertools.combinations_with_replacement(families_list, 2):
            self.family1 = fam1
            self.family2 = fam2
            self.fit(u)
            s = self.summary()
            rows.append({
                "family1": s["family1"],
                "family2": s["family2"],
                "rotation1": s["rotation1"],
                "rotation2": s["rotation2"],
                "loglik": s["loglik"],
                "aic": s["aic"],
                "bic": s["bic"],
                "npars": s["npars"],
                "weight1": s["weight1"],
                "weight2": s["weight2"],
                "converged": s["converged"],
                "n_iter": s["n_iter"],
            })

        fit_results = pd.DataFrame(rows)
        fit_results.sort_values(by=self.criterion, inplace=True, ignore_index=True)


        best = fit_results.iloc[0]
        self.family1 = best["family1"]
        self.family2 = best["family2"]
        self.rotation1_ = best["rotation1"]
        self.rotation2_ = best["rotation2"]
        self.weight_ = best["weight1"]
        self.copula1_ = _fit_bicop_fixed_family_rotation(u, self.family1, self.rotation1_, weights=None)
        self.copula2_ = _fit_bicop_fixed_family_rotation(u, self.family2, self.rotation2_, weights=None)
        self.loglik_ = best["loglik"]
        self.aic_ = best["aic"]
        self.bic_ = best["bic"]
        self.converged_ = best["converged"]
        self.n_iter_ = best["n_iter"]
        self.npars_ = best["npars"]
        return fit_results

    
    def aic(self):
        return self.aic_
    

    def bic(self):
        return self.bic_
    

    def loglik(self):
        return self.loglik_
    

    def n_parameters(self):
        return self.npars_


    def converged(self):        
        return self.converged_


    def plot(self, plot_type="contour", margin_type="unif", n=100):
        # Build grid
        u = np.linspace(0.001, 0.999, n)
        v = np.linspace(0.001, 0.999, n)

        # Points passed to copula pdf should stay on [0, 1]
        uu, vv = np.meshgrid(u, v)
        uv = np.column_stack([uu.ravel(), vv.ravel()])

        pdf_vals = self.pdf(uv).reshape(n, n)

        # Plot axes
        if margin_type == "unif":
            x, y = u, v
            xx, yy = uu, vv
            xlabel, ylabel = "u", "v"

        elif margin_type == "norm":
            x = stats.norm.ppf(u)
            y = stats.norm.ppf(v)
            xx, yy = np.meshgrid(x, y)
            xlabel, ylabel = "Normal margin x", "Normal margin y"

            # Optional: transform copula density to joint normal density
            pdf_vals = pdf_vals * stats.norm.pdf(xx) * stats.norm.pdf(yy)

        else:
            raise ValueError("margin_type must be 'unif' or 'norm'")

        plt.figure(figsize=(6, 5))

        if plot_type == "contour":
            plt.contourf(xx, yy, pdf_vals, levels=20, norm=LogNorm())
            plt.colorbar(label="Density")
        elif plot_type == "heatmap":
            plt.imshow(
                pdf_vals,
                origin="lower",
                extent=(x.min(), x.max(), y.min(), y.max()),
                aspect="auto",
                norm=LogNorm()
            )
            plt.colorbar(label="Density")
        else:
            raise ValueError("plot_type must be 'contour' or 'heatmap'")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(
            f"Mixture of {self.copula1_} and {self.copula2_} "
            f"with w1={self.weight_:.2f}"
        )
        plt.show()


class ThreeComponentVinecopulibMixture:
    """
    Three-component finite mixture of pyvinecopulib Bicop models.

        c(u, v) = w1*c1(u,v; family1, rot1) + w2*c2(u,v; family2, rot2) + w3*c3(u,v; Tawn, 90)

    Component 3 is always Tawn at rotation 90, with weight w3 constrained to [min_weight, max_weight3].
    Rotations for components 1 and 2 are explored via the init schedule.

    Default families:
        family1 = pv.BicopFamily.joe   (rotation 0)
        family2 = pv.BicopFamily.tawn  (initial rotation 180)
        family3 = pv.BicopFamily.tawn  (rotation 90, fixed, weight < 10%)
    """

    FAMILY3 = None          # set in __init__ (pv.BicopFamily.tawn)
    ROTATION3 = 0
    MAX_WEIGHT3 = 0.2      # hard upper bound on w3

    def __init__(
        self,
        family1=None,   # default: pv.BicopFamily.joe
        family2=None,   # default: pv.BicopFamily.tawn
        n_init=10,
        max_iter=200,
        tol=1e-3,
        min_weight=0.01,
        max_weight3=0.20,
        init_weights=None,
        random_state=None,
        criterion="aic",
    ):
        if n_init % 2 != 0:
            raise ValueError(f"n_init must be a multiple of 2, got {n_init}")

        self.family1 = family1 if family1 is not None else pv.BicopFamily.joe
        self.family2 = family2 if family2 is not None else pv.BicopFamily.tawn
        self.family3 = pv.BicopFamily.tawn          # always fixed
        self.rotation3 = 0                      # always fixed

        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.min_weight = min_weight
        self.max_weight3 = max_weight3              # w3 <= this value
        self.init_weights = init_weights if init_weights is not None else [0.1, 0.25, 0.5, 0.75, 0.9]
        self.random_state = np.random.default_rng(random_state)
        self.criterion = criterion

        self.weight1_ = None
        self.weight2_ = None
        self.weight3_ = None
        self.copula1_ = None
        self.copula2_ = None
        self.copula3_ = None
        self.rotation1_ = None
        self.rotation2_ = None
        self.loglik_ = -np.inf
        self.aic_ = np.inf
        self.bic_ = np.inf
        self.converged_ = False
        self.n_iter_ = 0
        self.npars_ = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_init_schedule(self):
        """
        Build list of (w1_init, rot1, rot2) for all n_init initialisations.

        w1_init cycles through self.init_weights.
        w3 is always initialised at max_weight3 / 2 (i.e. 5% by default),
        and the remainder 1 - w1 - w3 goes to component 2.

        rot1/rot2 pairs are drawn from the valid rotations for each family.
        """
        valid1 = _valid_rotations_for_family(self.family1)
        valid2 = _valid_rotations_for_family(self.family2)
        rot_pairs = [(r1, r2) for r1 in valid1 for r2 in valid2]

        half = self.n_init // 2
        schedule = []
        for rot1, rot2 in rot_pairs:
            for i in range(half):
                w1 = self.init_weights[i % len(self.init_weights)]
                schedule.append((w1, rot1, rot2))
        return schedule

    # ------------------------------------------------------------------
    # Core EM
    # ------------------------------------------------------------------

    def fit(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be n x 2")
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("u must lie strictly inside (0, 1). Use pseudo_obs().")
        # Drop boundary points (log-pdf is -inf there)
        mask = ~(np.any(u == 0, axis=1) | np.any(u == 1, axis=1))
        u = u[mask]

        n = u.shape[0]
        u_f = np.asfortranarray(u)
        best = None

        schedule = self._build_init_schedule()

        for (w1_init, rot1, rot2) in schedule:
            # --- Initialise responsibilities ---
            # w3 starts at half its maximum; remainder split w1 : w2
            w3_init = self.max_weight3 / 2.0
            w_rem   = 1.0 - w3_init
            # r1 ~ Beta centred on w1_init * w_rem
            r1 = self.random_state.beta(2, 2, size=n)
            r1 = (w1_init * w_rem) * r1 / np.mean(r1)
            r1 = np.clip(r1, 1e-3, 1.0 - 1e-3)

            r3 = np.full(n, w3_init)
            r2 = np.clip(1.0 - r1 - r3, 1e-3, 1.0 - 1e-3)
            # Renormalise so rows sum to 1
            row_sum = r1 + r2 + r3
            r1, r2, r3 = r1 / row_sum, r2 / row_sum, r3 / row_sum

            old_ll = -np.inf
            converged = False
            c1 = c2 = c3 = None

            for it in range(1, self.max_iter + 1):
                # --- M-step ---
                w1 = np.clip(np.mean(r1), self.min_weight, 1.0 - 2 * self.min_weight)
                w3 = np.clip(np.mean(r3), self.min_weight, self.max_weight3)
                w2 = np.clip(1.0 - w1 - w3, self.min_weight, 1.0 - 2 * self.min_weight)
                # Re-normalise in case clipping broke the sum
                total = w1 + w2 + w3
                w1, w2, w3 = w1 / total, w2 / total, w3 / total

                try:
                    c1 = _fit_bicop_fixed_family_rotation(u_f, self.family1, rot1,  weights=r1)
                    c2 = _fit_bicop_fixed_family_rotation(u_f, self.family2, rot2,  weights=r2)
                    c3 = _fit_bicop_fixed_family_rotation(u_f, self.family3, self.rotation3, weights=r3)
                except Exception:
                    break

                # --- Log-likelihood ---
                log_d1 = np.log(w1) + _safe_log_pdf(c1, u_f)
                log_d2 = np.log(w2) + _safe_log_pdf(c2, u_f)
                log_d3 = np.log(w3) + _safe_log_pdf(c3, u_f)
                log_mix = np.column_stack([log_d1, log_d2, log_d3])
                log_den = logsumexp(log_mix, axis=1)
                ll = float(np.sum(log_den))

                # --- Convergence ---
                if abs(ll - old_ll) < self.tol:
                    converged = True
                    break
                old_ll = ll

                # --- E-step (true posteriors) ---
                log_post = log_mix - log_den[:, None]   # (n, 3)
                r1 = np.exp(log_post[:, 0])
                r2 = np.exp(log_post[:, 1])
                r3 = np.exp(log_post[:, 2])

            if c1 is None or c2 is None or c3 is None:
                continue

            # --- Final evaluation ---
            w1 = np.clip(np.mean(r1), self.min_weight, 1.0 - 2 * self.min_weight)
            w3 = np.clip(np.mean(r3), self.min_weight, self.max_weight3)
            w2 = np.clip(1.0 - w1 - w3, self.min_weight, 1.0 - 2 * self.min_weight)
            total = w1 + w2 + w3
            w1, w2, w3 = w1 / total, w2 / total, w3 / total

            log_d1 = np.log(w1) + _safe_log_pdf(c1, u_f)
            log_d2 = np.log(w2) + _safe_log_pdf(c2, u_f)
            log_d3 = np.log(w3) + _safe_log_pdf(c3, u_f)
            ll = float(np.sum(logsumexp(np.column_stack([log_d1, log_d2, log_d3]), axis=1)))

            k = int(c1.npars + c2.npars + c3.npars + 2)   # 2 free weights
            aic = 2 * k - 2 * ll
            bic = np.log(n) * k - 2 * ll

            candidate = dict(
                weight1=w1, weight2=w2, weight3=w3,
                copula1=c1, copula2=c2, copula3=c3,
                rotation1=rot1, rotation2=rot2,
                loglik=ll, aic=aic, bic=bic,
                converged=converged, n_iter=it, npars=k,
            )

            if best is None or candidate["loglik"] > best["loglik"]:
                best = candidate

        if best is None:
            raise RuntimeError(
                f"All initialisations failed for {self.family1}+{self.family2}+Tawn(90)"
            )

        self.weight1_  = best["weight1"]
        self.weight2_  = best["weight2"]
        self.weight3_  = best["weight3"]
        self.copula1_  = best["copula1"]
        self.copula2_  = best["copula2"]
        self.copula3_  = best["copula3"]
        self.rotation1_ = best["rotation1"]
        self.rotation2_ = best["rotation2"]
        self.loglik_   = best["loglik"]
        self.aic_      = best["aic"]
        self.bic_      = best["bic"]
        self.converged_ = best["converged"]
        self.n_iter_   = best["n_iter"]
        self.npars_    = best["npars"]
        return self

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    def pdf(self, u):
        u = np.asfortranarray(np.asarray(u, dtype=float))
        return (
            self.weight1_ * self.copula1_.pdf(u)
            + self.weight2_ * self.copula2_.pdf(u)
            + self.weight3_ * self.copula3_.pdf(u)
        )

    def logpdf(self, u):
        return np.log(np.maximum(self.pdf(u), 1e-300))

    # ------------------------------------------------------------------
    # Summary / accessors
    # ------------------------------------------------------------------

    def summary(self):
        return {
            "family1":   str(self.family1),
            "family2":   str(self.family2),
            "family3":   str(self.family3),
            "rotation1": self.rotation1_,
            "rotation2": self.rotation2_,
            "rotation3": self.rotation3,
            "weight1":   self.weight1_,
            "weight2":   self.weight2_,
            "weight3":   self.weight3_,
            "copula1":   str(self.copula1_),
            "copula2":   str(self.copula2_),
            "copula3":   str(self.copula3_),
            "loglik":    self.loglik_,
            "aic":       self.aic_,
            "bic":       self.bic_,
            "npars":     self.npars_,
            "converged": self.converged_,
            "n_iter":    self.n_iter_,
        }

    def aic(self):        return self.aic_
    def bic(self):        return self.bic_
    def loglik(self):     return self.loglik_
    def n_parameters(self): return self.npars_
    def converged(self):  return self.converged_

    # ------------------------------------------------------------------
    # Model selection over family pairs
    # ------------------------------------------------------------------

    def select(self, u, families_list="all"):
        if families_list == "all":
            families_list = [
                pv.BicopFamily.bb1, pv.BicopFamily.bb6, pv.BicopFamily.bb7,
                pv.BicopFamily.bb8, pv.BicopFamily.clayton, pv.BicopFamily.frank,
                pv.BicopFamily.gaussian, pv.BicopFamily.gumbel, pv.BicopFamily.joe,
                pv.BicopFamily.tawn, pv.BicopFamily.student,
            ]

        rows = []
        for fam1, fam2 in itertools.combinations_with_replacement(families_list, 2):
            self.family1 = fam1
            self.family2 = fam2
            self.fit(u)
            s = self.summary()
            rows.append({
                "family1":   s["family1"],
                "family2":   s["family2"],
                "rotation1": s["rotation1"],
                "rotation2": s["rotation2"],
                "loglik":    s["loglik"],
                "aic":       s["aic"],
                "bic":       s["bic"],
                "npars":     s["npars"],
                "weight1":   s["weight1"],
                "weight2":   s["weight2"],
                "weight3":   s["weight3"],
                "converged": s["converged"],
                "n_iter":    s["n_iter"],
            })

        fit_results = pd.DataFrame(rows)
        fit_results.sort_values(by=self.criterion, inplace=True, ignore_index=True)

        best = fit_results.iloc[0]
        self.family1   = best["family1"]
        self.family2   = best["family2"]
        self.rotation1_ = best["rotation1"]
        self.rotation2_ = best["rotation2"]
        self.weight1_  = best["weight1"]
        self.weight2_  = best["weight2"]
        self.weight3_  = best["weight3"]

        u_f = np.asfortranarray(np.asarray(u, dtype=float))
        self.copula1_ = _fit_bicop_fixed_family_rotation(u_f, self.family1, self.rotation1_)
        self.copula2_ = _fit_bicop_fixed_family_rotation(u_f, self.family2, self.rotation2_)
        self.copula3_ = _fit_bicop_fixed_family_rotation(u_f, self.family3, self.rotation3)
        self.loglik_  = best["loglik"]
        self.aic_     = best["aic"]
        self.bic_     = best["bic"]
        self.converged_ = best["converged"]
        self.n_iter_  = best["n_iter"]
        self.npars_   = best["npars"]
        return fit_results

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, plot_type="contour", margin_type="unif", n=100):
        u = np.linspace(0.001, 0.999, n)
        v = np.linspace(0.001, 0.999, n)
        uu, vv = np.meshgrid(u, v)
        uv = np.column_stack([uu.ravel(), vv.ravel()])

        pdf_vals = self.pdf(uv).reshape(n, n)

        if margin_type == "unif":
            x, y = u, v
            xx, yy = uu, vv
            xlabel, ylabel = "u", "v"
        elif margin_type == "norm":
            x = stats.norm.ppf(u)
            y = stats.norm.ppf(v)
            xx, yy = np.meshgrid(x, y)
            xlabel, ylabel = "Normal margin x", "Normal margin y"
            pdf_vals = pdf_vals * stats.norm.pdf(xx) * stats.norm.pdf(yy)
        else:
            raise ValueError("margin_type must be 'unif' or 'norm'")

        plt.figure(figsize=(6, 5))

        if plot_type == "contour":
            plt.contourf(xx, yy, pdf_vals, levels=20, norm=LogNorm())
            plt.colorbar(label="Density")
        elif plot_type == "heatmap":
            plt.imshow(
                pdf_vals, origin="lower",
                extent=(x.min(), x.max(), y.min(), y.max()),
                aspect="auto", norm=LogNorm(),
            )
            plt.colorbar(label="Density")
        else:
            raise ValueError("plot_type must be 'contour' or 'heatmap'")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(
            f"Mixture: {self.copula1_} (w={self.weight1_:.2f}) + "
            f"{self.copula2_} (w={self.weight2_:.2f}) + "
            f"Tawn/° (w={self.weight3_:.2f})"
        )
        plt.show()
        

