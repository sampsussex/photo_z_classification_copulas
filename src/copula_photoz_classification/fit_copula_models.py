import itertools
import numpy as np
import pandas as pd
import pyvinecopulib as pv
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import stats
from utils import load_g09_waveswide_xys
from matplotlib.colors import LogNorm

# Families where rotation must be 0 (symmetric or no rotation support)
_ROTATION_ZERO_ONLY = {
    pv.BicopFamily.indep,
    pv.BicopFamily.gaussian,
    pv.BicopFamily.student,
    pv.BicopFamily.frank,
}


def pseudo_obs(x):
    """Convert raw 2D data to pseudo-observations in (0, 1)^2."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be n x 2")
    n = x.shape[0]
    ranks = np.empty_like(x, dtype=float)
    for j in range(2):
        order = np.argsort(x[:, j], kind="mergesort")
        ranks[order, j] = np.arange(1, n + 1)
    return ranks / (n + 1.0)


def _valid_rotations_for_family(family):
    """
    Return the rotations we allow for a given family.
    Symmetric families (Gaussian, Student, Frank, Independence) only support 0.
    All others get 0 and 180.
    """
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
        allow_rotations=True,#False,
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
            u = np.clip(u, 1e-12, 1 - 1e-12)


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




def fit_all_two_component_parametric_mixtures(
    u,
    n_init=6,
    max_iter=100,
    tol=1e-6,
    criterion="bic",
    init_weights=None,
    random_state=123
):
    """
    Fit all 2-component mixtures over pyvinecopulib parametric families.
    n_init must be a multiple of 2. Default of 6 gives:
      3 inits at rotation (0, 0)   x weights [0.25, 0.5, 0.75]
      3 inits at rotation (0, 180) x weights [0.25, 0.5, 0.75]

    Returns: best_model, results_dataframe
    """
    if n_init % 2 != 0:
        raise ValueError(f"n_init must be a multiple of 2, got {n_init}")

    families = (
        pv.BicopFamily.bb1, pv.BicopFamily.bb6, pv.BicopFamily.bb7,
        pv.BicopFamily.bb8, pv.BicopFamily.clayton, pv.BicopFamily.frank,
        pv.BicopFamily.gaussian, pv.BicopFamily.gumbel, pv.BicopFamily.joe,
        pv.BicopFamily.tawn, pv.BicopFamily.student
        # indep excluded: degenerate in mixtures
    )

    rows = []
    models = {}

    for fam1, fam2 in itertools.combinations_with_replacement(families, 2):
        #if fam1 != pv.BicopFamily.b:
        #    continue
        print(f"Fitting mixture: {fam1} + {fam2}")
        model = TwoComponentVinecopulibMixture(
            fam1,
            fam2,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            init_weights=init_weights,
            random_state=random_state
        )

        try:
            model.fit(u)
            key = (str(fam1), str(fam2))
            models[key] = model

            rows.append({
                "family1": str(fam1),
                "family2": str(fam2),
                "rotation1": model.rotation1_,
                "rotation2": model.rotation2_,
                "loglik": model.loglik_,
                "aic": model.aic_,
                "bic": model.bic_,
                "npars": model.npars_,
                "weight1": model.weight_,
                "weight2": 1 - model.weight_,
                "converged": model.converged_,
                "n_iter": model.n_iter_,
            })
            print(
                f"  rot=({model.rotation1_},{model.rotation2_}) "
                f"loglik={model.loglik_:.2f} aic={model.aic_:.2f} "
                f"bic={model.bic_:.2f} w1={model.weight_:.3f}"
            )

        except Exception as e:
            rows.append({
                "family1": str(fam1),
                "family2": str(fam2),
                "rotation1": np.nan,
                "rotation2": np.nan,
                "loglik": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "npars": np.nan,
                "weight1": np.nan,
                "weight2": np.nan,
                "converged": False,
                "n_iter": np.nan,
                "error": str(e),
            })
            print(f"  Failed: {e}")

    results = pd.DataFrame(rows)
    valid = results.dropna(subset=[criterion])
    if valid.empty:
        raise RuntimeError("No mixture fits succeeded.")

    # Fix sort direction: higher loglik is better, lower AIC/BIC is better
    ascending = (criterion != "loglik")
    best_row = valid.sort_values(criterion, ascending=ascending).iloc[0]
    best_key = (best_row["family1"], best_row["family2"])
    best_model = models[best_key]

    return best_model, results.sort_values(criterion, ascending=ascending)


def main():
    xy_all, xy_fn, tp_mask = load_g09_waveswide_xys()
    u = pseudo_obs(xy_all)

    best, table = fit_all_two_component_parametric_mixtures(
        u,
        n_init=12,
        max_iter=200,
        tol=1e-2,
        criterion="loglik",
        init_weights=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        random_state=1
    )

    # Full results table
    print("\n--- All results (sorted by log-likelihood) ---")
    print(table.to_string(index=False))

    # Top 10
    print("\n--- Top 10 by BIC ---")
    print(table.head(10).to_string(index=False))

    # Best model
    print("\n--- Best model summary ---")
    s = best.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    table.to_csv("bicop_mixture_fits.csv", index=False)

if __name__ == "__main__":
    main()