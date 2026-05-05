import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, logsumexp
from scipy.stats import beta
from utils import load_g09_waveswide_xys
import pyvinecopulib as pv

def pseudo_obs(x):
    """
    Convert raw 2D data to pseudo-observations in (0, 1)^2.
    If your data are already uniforms, skip this.
    """
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x must be n x 2")

    n = x.shape[0]
    ranks = np.empty_like(x, dtype=float)

    for j in range(2):
        order = np.argsort(x[:, j], kind="mergesort")
        ranks[order, j] = np.arange(1, n + 1)

    return ranks / (n + 1.0)


def _log_beta_basis(u, order):
    """
    Bernstein copula density basis:
        b_i(u) = BetaPDF(u; i+1, order-i+1), i=0..order

    Returns n x K log-basis matrix, K = order + 1.
    """
    u = np.asarray(u, dtype=float)
    u = np.clip(u, 1e-12, 1 - 1e-12)

    K = order + 1
    out = np.empty((u.size, K))

    for i in range(K):
        a = i + 1
        b = order - i + 1
        out[:, i] = (
            (a - 1) * np.log(u)
            + (b - 1) * np.log1p(-u)
            - betaln(a, b)
        )

    return out


def _sinkhorn_to_copula_weights(logits, max_iter=200, tol=1e-12):
    """
    Convert unconstrained logits into a nonnegative K x K matrix A satisfying:

        row_sums(A) = 1 / K
        col_sums(A) = 1 / K

    This enforces uniform copula margins.
    """
    L = logits - np.max(logits)
    A = np.exp(L)

    K = A.shape[0]
    target = 1.0 / K

    for _ in range(max_iter):
        A *= target / A.sum(axis=1, keepdims=True)
        A *= target / A.sum(axis=0, keepdims=True)

        row_err = np.max(np.abs(A.sum(axis=1) - target))
        col_err = np.max(np.abs(A.sum(axis=0) - target))
        if max(row_err, col_err) < tol:
            break

    return A


class BernsteinBicop:
    """
    Low-order Bernstein copula density estimator.

    Density:
        c(u, v) = sum_ij A_ij b_i(u) b_j(v)

    where b_i are beta density Bernstein basis functions and A is constrained
    to have nonnegative entries with uniform row/column sums.

    Effective DoF:
        order = 2  -> K=3 -> 4 DoF
        order = 3  -> K=4 -> 9 DoF
    """

    def __init__(
        self,
        order=3,
        l2_penalty=0.0,
        max_iter=1000,
        random_state=123,
    ):
        if order < 1:
            raise ValueError("order must be >= 1")

        self.order = order
        self.K = order + 1
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.random_state = np.random.default_rng(random_state)

        self.coef_ = None
        self.loglik_ = -np.inf
        self.aic_ = np.inf
        self.bic_ = np.inf
        self.npars_ = (self.K - 1) ** 2
        self.success_ = False
        self.result_ = None

    def fit(self, u, n_init=10):
        u = np.asarray(u, dtype=float)

        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("u must be n x 2")
        if np.any(u <= 0) or np.any(u >= 1):
            raise ValueError("u must lie strictly inside (0, 1)")

        n = u.shape[0]
        log_bu = _log_beta_basis(u[:, 0], self.order)
        log_bv = _log_beta_basis(u[:, 1], self.order)

        K = self.K

        def unpack(theta):
            logits = theta.reshape(K, K)
            return _sinkhorn_to_copula_weights(logits)

        def neg_pen_loglik(theta):
            A = unpack(theta)
            logA = np.log(np.maximum(A, 1e-300))

            # log c(u_n, v_n)
            terms = (
                log_bu[:, :, None]
                + log_bv[:, None, :]
                + logA[None, :, :]
            )
            log_pdf = logsumexp(terms.reshape(n, K * K), axis=1)

            ll = np.sum(log_pdf)

            if self.l2_penalty > 0:
                # Smoothness penalty: discourages spiky checkerboards.
                d_row = np.diff(A, axis=0)
                d_col = np.diff(A, axis=1)
                penalty = self.l2_penalty * (
                    np.sum(d_row ** 2) + np.sum(d_col ** 2)
                )
                ll -= penalty

            return -ll

        best_res = None

        # include near-independence start
        starts = [np.zeros(K * K)]

        # random starts
        for _ in range(n_init - 1):
            starts.append(self.random_state.normal(scale=0.5, size=K * K))

        for theta0 in starts:
            res = minimize(
                neg_pen_loglik,
                theta0,
                method="L-BFGS-B",
                options={
                    "maxiter": self.max_iter,
                    "ftol": 1e-10,
                    "gtol": 1e-6,
                },
            )

            if best_res is None or res.fun < best_res.fun:
                best_res = res

        self.result_ = best_res
        self.coef_ = unpack(best_res.x)

        self.loglik_ = -float(best_res.fun)
        self.aic_ = 2 * self.npars_ - 2 * self.loglik_
        self.bic_ = np.log(n) * self.npars_ - 2 * self.loglik_
        self.success_ = bool(best_res.success)

        return self

    def pdf(self, u):
        u = np.asarray(u, dtype=float)
        log_bu = _log_beta_basis(u[:, 0], self.order)
        log_bv = _log_beta_basis(u[:, 1], self.order)

        logA = np.log(np.maximum(self.coef_, 1e-300))
        n = u.shape[0]
        K = self.K

        terms = (
            log_bu[:, :, None]
            + log_bv[:, None, :]
            + logA[None, :, :]
        )
        return np.exp(logsumexp(terms.reshape(n, K * K), axis=1))

    def logpdf(self, u):
        return np.log(np.maximum(self.pdf(u), 1e-300))

    def cdf(self, u):
        u = np.asarray(u, dtype=float)
        K = self.K

        Bu = np.column_stack([
            beta.cdf(u[:, 0], i + 1, self.order - i + 1)
            for i in range(K)
        ])
        Bv = np.column_stack([
            beta.cdf(u[:, 1], j + 1, self.order - j + 1)
            for j in range(K)
        ])

        return np.einsum("ni,ij,nj->n", Bu, self.coef_, Bv)

    def summary(self):
        return {
            "type": "BernsteinBicop",
            "order": self.order,
            "K": self.K,
            "npars": self.npars_,
            "loglik": self.loglik_,
            "aic": self.aic_,
            "bic": self.bic_,
            "success": self.success_,
            "coef": self.coef_,
        }
    


xy_all, xy_fn, tp_mask = load_g09_waveswide_xys()
u = pseudo_obs(xy_all)

bern = BernsteinBicop(
    order=3,          # 9 DoF
    l2_penalty=0.0,   # try 1e-3 or 1e-2 if surface is too spiky
    max_iter=2000,
    random_state=1,
)

bern.fit(u, n_init=20)

print(bern.summary())

bb8 = pv.Bicop.from_family(family=pv.BicopFamily.bb8, rotation=0)
bb8.fit(np.asfortranarray(u))

ll_bb8 = np.sum(np.log(np.maximum(bb8.pdf(np.asfortranarray(u)), 1e-300)))
ll_bern = bern.loglik_

print("BB8 loglik:", ll_bb8)
print("Bernstein loglik:", ll_bern)
print("Bernstein - BB8:", ll_bern - ll_bb8)


for order in [2, 3, 4]:
    m = BernsteinBicop(order=order, l2_penalty=1e-3, random_state=1)
    m.fit(u, n_init=20)
    print(order, m.npars_, m.loglik_, m.aic_, m.bic_)