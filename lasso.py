"""
LASSO: estimation, oracle inequalities, and parameter vector bounds.

Theoretical framework
---------------------
Bickel, Ritov & Tsybakov (2009), "Simultaneous analysis of Lasso and
Dantzig selector", Ann. Statist. 37(4):1705-1732.

Model:  y = X β* + ε,  ε_i i.i.d. subgaussian(σ²),  β* ∈ ℝᵖ

Estimator:
  β̂_λ = argmin_β  (1/2n)‖y − Xβ‖² + λ‖β‖₁

Restricted Eigenvalue (RE) condition  [Definition 4, BRT 2009]:
  κ²(s, c₀) = min { ‖XΔ‖²/n  /  ‖Δ_S‖²₂  :  |S| ≤ s,
                    ‖Δ_{Sᶜ}‖₁ ≤ c₀ ‖Δ_S‖₁,  Δ ≠ 0 }

Oracle inequalities  [Theorem 7.2, BRT 2009]
  Hold when λ ≥ λ_min := 2‖Xᵀε‖_∞/n, which with confidence 1−δ is
  guaranteed by  λ = 2σ√(2 log(2p/δ)/n).

  Let κ₀² = κ²(s*, 3),  s* = ‖β*‖₀.  Then:

    Prediction : (1/n)‖X(β̂−β*)‖²  ≤  16 s* λ² / κ₀²
    ℓ₂ error  : ‖β̂−β*‖₂           ≤   4 λ (√s* + 3s*) / κ₀²
    ℓ₁ error  : ‖β̂−β*‖₁           ≤  16 s* λ / κ₀²

  Derivation sketch:
    From the KKT/basic inequality and λ ≥ λ_min:
      (cone)        ‖Δ_{Sᶜ}‖₁ ≤ 3 ‖Δ_S‖₁
      (prediction)  ‖XΔ‖²/n   ≤ 4λ‖Δ_S‖₁ ≤ 4λ√s‖Δ_S‖₂
    The RE condition lower-bounds ‖XΔ‖²/n by κ₀²‖Δ_S‖₂², so:
      κ₀² ‖Δ_S‖₂ ≤ 4λ√s  ⟹  ‖Δ_S‖₂ ≤ 4λ√s/κ₀²
    Substituting back gives the three bounds above.

Support recovery [Zhao & Yu 2006]:
  Sign consistency requires:
    (beta-min)         min_{j∈S}|β*_j| > 2λ√s/κ₀²
    (irrepresentability)  ‖X_{Sᶜ}ᵀ X_S (X_Sᵀ X_S)⁻¹ sign(β*_S)‖_∞ ≤ 1
"""

import numpy as np
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class REResult:
    """Result of a Restricted Eigenvalue condition computation."""
    kappa_sq: float       # κ²(s, c₀) — the RE constant
    kappa: float          # √kappa_sq
    sparsity: int
    cone_constant: float  # c₀ used
    method: str           # "monte_carlo" or "exact_eigenvalue"


@dataclass
class OracleBounds:
    """Upper bounds from the BRT (2009) oracle inequalities."""
    prediction_bound: float   # (1/n)‖X(β̂−β*)‖² ≤ this
    l2_bound: float           # ‖β̂−β*‖₂         ≤ this
    l1_bound: float           # ‖β̂−β*‖₁         ≤ this
    lambda_used: float
    lambda_min: float         # λ_min for the bounds to hold w.h.p.
    re_constant: float        # κ₀²
    sparsity: int
    noise_level: float


@dataclass
class SupportRecoveryConditions:
    """Sign/support consistency conditions."""
    beta_min: float
    beta_min_threshold: float     # must exceed this
    beta_min_satisfied: bool
    irrepresentability_value: float   # ‖X_{Sᶜ}ᵀ X_S (X_Sᵀ X_S)⁻¹ sign(β*_S)‖_∞
    irrepresentability_satisfied: bool


# ---------------------------------------------------------------------------
# Prox operator
# ---------------------------------------------------------------------------

def soft_threshold(z: np.ndarray, lam: float) -> np.ndarray:
    """Soft-thresholding: prox operator for the ℓ₁ norm."""
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


# ---------------------------------------------------------------------------
# LASSO solvers
# ---------------------------------------------------------------------------

class LassoSolver:
    """
    LASSO via cyclic coordinate descent (Friedman et al. 2007).

    Minimizes  (1/2n)‖y − Xβ‖² + λ‖β‖₁.

    The coordinate update for feature j is:
      β_j ← S( Xⱼᵀ r^(j)/n,  λ )  /  (‖Xⱼ‖²/n)
    where r^(j) = y − X_{-j}β_{-j} is the partial residual and
    S(z, t) = sign(z)·max(|z|−t, 0) is soft-thresholding.
    """

    def __init__(
        self,
        lam: float,
        max_iter: int = 5000,
        tol: float = 1e-8,
        warm_start: Optional[np.ndarray] = None,
    ):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.coef_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.objective_history_: List[float] = []

    def _objective(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        r = y - X @ beta
        return float(r @ r) / (2 * len(y)) + self.lam * float(np.sum(np.abs(beta)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoSolver":
        n, p = X.shape
        beta = self.warm_start.copy() if self.warm_start is not None else np.zeros(p)
        col_sq = (X ** 2).sum(axis=0) / n  # ‖Xⱼ‖²/n

        self.objective_history_ = [self._objective(X, y, beta)]

        for it in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                if col_sq[j] < 1e-12:
                    continue
                # Partial residual: add back j-th contribution
                rj = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j] @ rj / n
                beta[j] = soft_threshold(rho_j, self.lam) / col_sq[j]

            self.objective_history_.append(self._objective(X, y, beta))
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


class LassoFISTA:
    """
    LASSO via FISTA (Beck & Teboulle 2009) — O(1/k²) convergence.

    Minimizes  (1/2n)‖y − Xβ‖² + λ‖β‖₁  via proximal gradient with
    Nesterov momentum.  Step size is 1/L where L = σ_max(X)²/n is the
    Lipschitz constant of the smooth part.
    """

    def __init__(
        self,
        lam: float,
        max_iter: int = 2000,
        tol: float = 1e-8,
        warm_start: Optional[np.ndarray] = None,
    ):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.coef_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.objective_history_: List[float] = []

    def _grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return X.T @ (X @ beta - y) / len(y)

    def _objective(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        r = y - X @ beta
        return float(r @ r) / (2 * len(y)) + self.lam * float(np.sum(np.abs(beta)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoFISTA":
        n, p = X.shape
        L = float(np.linalg.norm(X, ord=2) ** 2) / n  # Lipschitz constant

        beta = self.warm_start.copy() if self.warm_start is not None else np.zeros(p)
        z = beta.copy()
        t = 1.0

        self.objective_history_ = [self._objective(X, y, beta)]

        for it in range(self.max_iter):
            beta_prev = beta.copy()
            beta = soft_threshold(z - self._grad(X, y, z) / L, self.lam / L)
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            z = beta + ((t - 1.0) / t_new) * (beta - beta_prev)
            t = t_new

            self.objective_history_.append(self._objective(X, y, beta))
            if np.max(np.abs(beta - beta_prev)) < self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


# ---------------------------------------------------------------------------
# Restricted Eigenvalue condition
# ---------------------------------------------------------------------------

def compute_re_constant(
    X: np.ndarray,
    s: int,
    c0: float = 3.0,
    n_trials: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> REResult:
    """
    Monte-Carlo lower bound on the RE constant κ²(s, c₀).

    For each trial:
      1. Sample a random support S of size s.
      2. Sample Δ_S uniformly on the unit ℓ₂ sphere.
      3. Fill Δ_{Sᶜ} subject to ‖Δ_{Sᶜ}‖₁ ≤ c₀‖Δ_S‖₁ (adversarially scaled).
      4. Evaluate ‖XΔ‖²/(n ‖Δ_S‖²).

    The minimum over all trials is a stochastic lower bound on κ²(s, c₀).
    Increase n_trials for tighter estimates.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, p = X.shape
    s = min(s, p)
    min_ratio = np.inf

    for _ in range(n_trials):
        S = rng.choice(p, size=s, replace=False)
        Sc = np.setdiff1d(np.arange(p), S)

        delta = np.zeros(p)
        # Δ_S on unit ℓ₂ sphere
        delta_S = rng.standard_normal(s)
        delta_S /= np.linalg.norm(delta_S) + 1e-15
        delta[S] = delta_S

        # Δ_{Sᶜ}: adversarial direction, scaled to be at boundary c₀‖Δ_S‖₁
        if len(Sc) > 0:
            z = rng.standard_normal(len(Sc))
            l1_S = float(np.sum(np.abs(delta_S)))
            scale = c0 * l1_S * rng.uniform()
            z = z / (np.sum(np.abs(z)) + 1e-15) * scale
            delta[Sc] = z

        norm_S_sq = float(np.dot(delta[S], delta[S]))
        if norm_S_sq < 1e-15:
            continue

        Xd = X @ delta
        ratio = float(np.dot(Xd, Xd)) / (n * norm_S_sq)
        if ratio < min_ratio:
            min_ratio = ratio

    kappa_sq = max(min_ratio if np.isfinite(min_ratio) else 0.0, 0.0)
    return REResult(
        kappa_sq=kappa_sq,
        kappa=np.sqrt(kappa_sq),
        sparsity=s,
        cone_constant=c0,
        method="monte_carlo",
    )


def compute_re_constant_exact(
    X: np.ndarray,
    s: int,
    c0: float = 3.0,
) -> REResult:
    """
    Exact RE lower bound via minimum eigenvalue over all supports of size s.

    Uses the fact that for unit vectors supported on S:
      min_{v: ‖v‖=1, supp(v)⊆S} ‖Xv‖²/n = λ_min(X_Sᵀ X_S / n)

    This is a lower bound on κ²(s, c₀) for any c₀ ≥ 0.
    Feasible only for small p (≲ 20) and small s.
    """
    n, p = X.shape
    s = min(s, p)
    min_eigval = np.inf

    for S in combinations(range(p), s):
        XS = X[:, list(S)]
        gram = XS.T @ XS / n
        eigval = float(np.linalg.eigvalsh(gram).min())
        if eigval < min_eigval:
            min_eigval = eigval

    kappa_sq = max(min_eigval, 0.0)
    return REResult(
        kappa_sq=kappa_sq,
        kappa=np.sqrt(kappa_sq),
        sparsity=s,
        cone_constant=c0,
        method="exact_eigenvalue",
    )


# ---------------------------------------------------------------------------
# Lambda selection
# ---------------------------------------------------------------------------

def universal_lambda(p: int, n: int, sigma: float, confidence: float = 0.95) -> float:
    """
    Universal threshold  λ = 2σ√(2 log(2p/δ)/n),  where δ = 1 − confidence.

    For Gaussian ε ~ N(0, σ²Iₙ):
      P(‖Xᵀε‖_∞/n ≤ λ/2) ≥ 1 − δ  (by a union bound over p coordinates).

    Choosing λ ≥ 2‖Xᵀε‖_∞/n is the condition required for the BRT oracle
    inequalities to hold.
    """
    delta = 1.0 - confidence
    return 2.0 * sigma * np.sqrt(2.0 * np.log(2.0 * p / delta) / n)


def estimate_noise_level(X: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate σ from data.

    n > p : OLS residual standard deviation (unbiased).
    n ≤ p : median absolute deviation of y (scale-consistent for Gaussian).
    """
    n, p = X.shape
    if n > p:
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_ols
        return float(np.std(residuals, ddof=p))
    return float(np.median(np.abs(y - np.median(y))) / 0.6745)


def cross_validate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    cv: int = 5,
    solver: str = "cd",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray]:
    """
    K-fold cross-validation for λ selection.

    Returns (best_lambda, mean_cv_mse_per_lambda).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y)
    cv = min(cv, n)
    fold_ids = rng.permutation(n) % cv
    cv_errors = np.zeros(len(lambdas))
    SolverClass = LassoSolver if solver == "cd" else LassoFISTA

    for fold in range(cv):
        train, val = fold_ids != fold, fold_ids == fold
        X_tr, y_tr = X[train], y[train]
        X_val, y_val = X[val], y[val]
        for i, lam in enumerate(lambdas):
            sol = SolverClass(lam=float(lam)).fit(X_tr, y_tr)
            r = y_val - sol.predict(X_val)
            cv_errors[i] += float(r @ r) / int(val.sum())

    cv_errors /= cv
    best_lam = float(lambdas[np.argmin(cv_errors)])
    return best_lam, cv_errors


# ---------------------------------------------------------------------------
# Oracle inequalities
# ---------------------------------------------------------------------------

def compute_oracle_bounds(
    X: np.ndarray,
    sparsity: int,
    sigma: float,
    lam: float,
    kappa_sq: float,
) -> OracleBounds:
    """
    BRT (2009) Theorem 7.2 upper bounds on estimation and prediction error.

    Parameters
    ----------
    X        : (n, p) design matrix (columns assumed mean-zero)
    sparsity : s = ‖β*‖₀
    sigma    : noise standard deviation
    lam      : regularization parameter used for β̂
    kappa_sq : RE constant κ²(s, 3)

    Returns
    -------
    OracleBounds dataclass with the three BRT bounds and the minimum λ
    required for the bounds to hold with 95% confidence.

    Notes
    -----
    Prediction bound derivation:
      From RE and the cone constraint:
        ‖XΔ‖²/n ≤ 4λ√s · ‖Δ_S‖₂  and  κ₀² ‖Δ_S‖₂ ≤ 4λ√s
      so  ‖XΔ‖/√n ≤ 4λ√s/κ₀  and  ‖XΔ‖²/n ≤ 16sλ²/κ₀².

    ℓ₂ bound derivation:
      ‖Δ‖₂ ≤ ‖Δ_S‖₂ + ‖Δ_{Sᶜ}‖₂
            ≤ ‖Δ_S‖₂ + ‖Δ_{Sᶜ}‖₁   (since ‖·‖₂ ≤ ‖·‖₁)
            ≤ ‖Δ_S‖₂ + 3√s‖Δ_S‖₂   (cone constraint)
            ≤ (1 + 3√s) · 4λ√s/κ₀²
             = 4λ(√s + 3s)/κ₀².

    ℓ₁ bound derivation:
      ‖Δ‖₁ ≤ 4‖Δ_S‖₁ ≤ 4√s‖Δ_S‖₂ ≤ 16sλ/κ₀².
    """
    n, p = X.shape
    s = sparsity
    k = kappa_sq

    prediction_bound = 16.0 * s * lam ** 2 / k
    l2_bound = 4.0 * lam * (np.sqrt(s) + 3.0 * s) / k
    l1_bound = 16.0 * s * lam / k
    lam_min = universal_lambda(p, n, sigma, confidence=0.95)

    return OracleBounds(
        prediction_bound=prediction_bound,
        l2_bound=l2_bound,
        l1_bound=l1_bound,
        lambda_used=lam,
        lambda_min=lam_min,
        re_constant=kappa_sq,
        sparsity=s,
        noise_level=sigma,
    )


def verify_oracle_inequality(
    X: np.ndarray,
    beta_hat: np.ndarray,
    beta_star: np.ndarray,
    bounds: OracleBounds,
) -> dict:
    """
    Empirically evaluate β̂ against the oracle bounds.

    Returns a dict with actual errors, bound values, pass/fail flags, and
    tightness ratios (actual / bound).  A tightness ratio near 1 indicates
    a nearly tight bound; a ratio much less than 1 indicates slack.
    """
    n = X.shape[0]
    delta = beta_hat - beta_star

    pred_err = float((X @ delta) @ (X @ delta)) / n
    l2_err = float(np.linalg.norm(delta))
    l1_err = float(np.sum(np.abs(delta)))

    def _tight(actual, bound):
        return actual / bound if bound > 1e-15 else np.inf

    return {
        "actual_prediction_error": pred_err,
        "actual_l2_error": l2_err,
        "actual_l1_error": l1_err,
        "prediction_bound": bounds.prediction_bound,
        "l2_bound": bounds.l2_bound,
        "l1_bound": bounds.l1_bound,
        "prediction_bound_holds": pred_err <= bounds.prediction_bound,
        "l2_bound_holds": l2_err <= bounds.l2_bound,
        "l1_bound_holds": l1_err <= bounds.l1_bound,
        "lambda_adequate": bounds.lambda_used >= bounds.lambda_min,
        "tightness_prediction": _tight(pred_err, bounds.prediction_bound),
        "tightness_l2": _tight(l2_err, bounds.l2_bound),
        "tightness_l1": _tight(l1_err, bounds.l1_bound),
    }


# ---------------------------------------------------------------------------
# Support recovery conditions
# ---------------------------------------------------------------------------

def check_support_recovery(
    X: np.ndarray,
    beta_star: np.ndarray,
    lam: float,
    kappa_sq: float,
) -> SupportRecoveryConditions:
    """
    Evaluate the two classical sufficient conditions for sign consistency.

    Beta-min condition [sufficient, Wainwright 2009]:
      min_{j∈S}|β*_j| > 2λ√s/κ₀²
      Ensures the true nonzeros are large enough to survive thresholding.

    Irrepresentability condition [necessary & sufficient, Zhao & Yu 2006]:
      ‖X_{Sᶜ}ᵀ X_S (X_Sᵀ X_S)⁻¹ sign(β*_S)‖_∞ ≤ 1
      The left side measures how much the irrelevant features are
      correlated with the relevant ones; it must be < 1 for consistent
      model selection.
    """
    n = X.shape[0]
    S = np.where(np.abs(beta_star) > 1e-10)[0]
    s = len(S)

    beta_min = float(np.min(np.abs(beta_star[S]))) if s > 0 else 0.0
    threshold = 2.0 * lam * np.sqrt(s) / kappa_sq if kappa_sq > 1e-15 else np.inf

    Sc = np.setdiff1d(np.arange(X.shape[1]), S)
    if s == 0 or len(Sc) == 0:
        irr_value = 0.0
    else:
        XS = X[:, S]
        XSc = X[:, Sc]
        sign_S = np.sign(beta_star[S])
        gram_S = XS.T @ XS / n
        try:
            irr_vec = (XSc.T @ XS / n) @ np.linalg.solve(gram_S, sign_S)
            irr_value = float(np.max(np.abs(irr_vec)))
        except np.linalg.LinAlgError:
            irr_value = np.inf

    return SupportRecoveryConditions(
        beta_min=beta_min,
        beta_min_threshold=threshold,
        beta_min_satisfied=beta_min > threshold,
        irrepresentability_value=irr_value,
        irrepresentability_satisfied=irr_value <= 1.0,
    )


# ---------------------------------------------------------------------------
# Regularization path
# ---------------------------------------------------------------------------

def lasso_path(
    X: np.ndarray,
    y: np.ndarray,
    n_lambdas: int = 50,
    lambda_min_ratio: float = 1e-3,
    solver: str = "cd",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full LASSO regularization path computed with warm starts.

    λ_max is the smallest value that drives all coefficients to zero:
      λ_max = ‖Xᵀy‖_∞ / n

    Lambda grid is log-spaced from λ_max down to λ_max * lambda_min_ratio.

    Returns
    -------
    lambdas   : (n_lambdas,) decreasing sequence
    coef_path : (p, n_lambdas) — column j is β̂ at lambdas[j]
    """
    n, p = X.shape
    lam_max = float(np.max(np.abs(X.T @ y))) / n
    lambdas = np.geomspace(lam_max, lam_max * lambda_min_ratio, num=n_lambdas)
    coef_path = np.zeros((p, n_lambdas))
    SolverClass = LassoSolver if solver == "cd" else LassoFISTA
    warm = None

    for i, lam in enumerate(lambdas):
        sol = SolverClass(lam=float(lam), warm_start=warm).fit(X, y)
        coef_path[:, i] = sol.coef_
        warm = sol.coef_.copy()

    return lambdas, coef_path


# ---------------------------------------------------------------------------
# Top-level analysis class
# ---------------------------------------------------------------------------

class LassoAnalysis:
    """
    End-to-end LASSO analysis: estimation + oracle bounds + support recovery.

    When beta_star is provided, computes the RE constant, all three oracle
    inequality bounds, empirically verifies them, and checks the two support
    recovery conditions.

    Parameters
    ----------
    lam       : regularization parameter; if None, selected by cross-validation
    solver    : "cd" (coordinate descent) or "fista"
    re_trials : number of Monte-Carlo trials for the RE constant estimate

    Usage
    -----
    la = LassoAnalysis(lam=0.1)
    la.fit(X, y, beta_star=beta_true, sigma=0.5)
    print(la.summary())
    """

    def __init__(
        self,
        lam: Optional[float] = None,
        solver: str = "cd",
        max_iter: int = 5000,
        tol: float = 1e-8,
        cv_folds: int = 5,
        re_trials: int = 5000,
        rng_seed: int = 0,
    ):
        self.lam = lam
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.cv_folds = cv_folds
        self.re_trials = re_trials
        self._rng = np.random.default_rng(rng_seed)

        self.coef_: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None
        self.re_result_: Optional[REResult] = None
        self.oracle_bounds_: Optional[OracleBounds] = None
        self.verification_: Optional[dict] = None
        self.support_recovery_: Optional[SupportRecoveryConditions] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_star: Optional[np.ndarray] = None,
        sigma: Optional[float] = None,
    ) -> "LassoAnalysis":
        n, p = X.shape

        self.sigma_ = sigma if sigma is not None else estimate_noise_level(X, y)

        if self.lam is None:
            lam_max = float(np.max(np.abs(X.T @ y))) / n
            lambdas = np.geomspace(lam_max, lam_max * 1e-4, num=40)
            self.lam, _ = cross_validate_lambda(
                X, y, lambdas=lambdas,
                cv=self.cv_folds, solver=self.solver, rng=self._rng,
            )

        SolverClass = LassoSolver if self.solver == "cd" else LassoFISTA
        sol = SolverClass(
            lam=self.lam, max_iter=self.max_iter, tol=self.tol,
        ).fit(X, y)
        self.coef_ = sol.coef_
        self._n_iter = sol.n_iter_
        self._objective_history = sol.objective_history_

        if beta_star is not None:
            s = max(int(np.sum(np.abs(beta_star) > 1e-10)), 1)

            self.re_result_ = compute_re_constant(
                X, s=s, c0=3.0, n_trials=self.re_trials, rng=self._rng,
            )

            self.oracle_bounds_ = compute_oracle_bounds(
                X=X,
                sparsity=s,
                sigma=self.sigma_,
                lam=self.lam,
                kappa_sq=self.re_result_.kappa_sq,
            )

            self.verification_ = verify_oracle_inequality(
                X=X,
                beta_hat=self.coef_,
                beta_star=beta_star,
                bounds=self.oracle_bounds_,
            )

            self.support_recovery_ = check_support_recovery(
                X=X,
                beta_star=beta_star,
                lam=self.lam,
                kappa_sq=self.re_result_.kappa_sq,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def summary(self) -> str:
        W = 62
        lines = ["=" * W, f"{'LASSO Analysis Summary':^{W}}", "=" * W]
        lines += [
            f"  Solver:           {self.solver.upper()}  ({self._n_iter} iterations)",
            f"  Lambda (lambda):  {self.lam:.6g}",
            f"  Noise level (sigma): {self.sigma_:.6g}",
            f"  Nonzero coefficients: "
            f"{int(np.sum(np.abs(self.coef_) > 1e-10))} / {len(self.coef_)}",
        ]
        if self.re_result_ is not None:
            r = self.re_result_
            lines += [
                "",
                f"  {'Restricted Eigenvalue Condition':^{W-2}}",
                f"  k^2(s={r.sparsity}, c0={r.cone_constant}) = {r.kappa_sq:.6g}",
                f"  kappa = {r.kappa:.6g}   [{r.method}]",
            ]
        if self.oracle_bounds_ is not None:
            b = self.oracle_bounds_
            chk = "OK" if b.lambda_used >= b.lambda_min else "WARNING: lambda too small"
            lines += [
                "",
                f"  {'Oracle Bounds  [BRT 2009, Thm 7.2]':^{W-2}}",
                f"  lambda_min (95% conf): {b.lambda_min:.6g}   [{chk}]",
                f"  Prediction bound:  {b.prediction_bound:.6g}",
                f"  L2 error bound:    {b.l2_bound:.6g}",
                f"  L1 error bound:    {b.l1_bound:.6g}",
            ]
        if self.verification_ is not None:
            v = self.verification_
            sym = lambda holds: "PASS" if holds else "FAIL"
            lines += [
                "",
                f"  {'Empirical Verification':^{W-2}}",
                f"  Prediction error: {v['actual_prediction_error']:.6g}"
                f"  [{sym(v['prediction_bound_holds'])},"
                f" tightness={v['tightness_prediction']:.3f}]",
                f"  L2 error:         {v['actual_l2_error']:.6g}"
                f"  [{sym(v['l2_bound_holds'])},"
                f" tightness={v['tightness_l2']:.3f}]",
                f"  L1 error:         {v['actual_l1_error']:.6g}"
                f"  [{sym(v['l1_bound_holds'])},"
                f" tightness={v['tightness_l1']:.3f}]",
            ]
        if self.support_recovery_ is not None:
            sr = self.support_recovery_
            lines += [
                "",
                f"  {'Support Recovery Conditions':^{W-2}}",
                f"  Beta-min: {sr.beta_min:.6g}"
                f"  (threshold={sr.beta_min_threshold:.6g},"
                f" {'PASS' if sr.beta_min_satisfied else 'FAIL'})",
                f"  Irrepresentability: {sr.irrepresentability_value:.6g}"
                f"  ({'PASS' if sr.irrepresentability_satisfied else 'FAIL'} <= 1)",
            ]
        lines.append("=" * W)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, p, s = 300, 60, 6
    sigma = 0.5

    # True sparse parameter vector
    beta_star = np.zeros(p)
    support = rng.choice(p, s, replace=False)
    beta_star[support] = rng.choice([-1.0, 1.0], size=s) * rng.uniform(1.5, 3.0, s)

    # Design matrix (standardized columns)
    X_raw = rng.standard_normal((n, p))
    X = X_raw / X_raw.std(axis=0)
    y = X @ beta_star + sigma * rng.standard_normal(n)

    # ---- 1. Full analysis with oracle bounds --------------------------------
    print("Running full LASSO analysis...")
    la = LassoAnalysis(
        lam=universal_lambda(p, n, sigma, confidence=0.95),
        solver="fista",
        re_trials=3000,
    )
    la.fit(X, y, beta_star=beta_star, sigma=sigma)
    print(la.summary())

    # ---- 2. Regularization path --------------------------------------------
    print("\nComputing regularization path...")
    lambdas, path = lasso_path(X, y, n_lambdas=100, solver="cd")
    sparsity_path = (np.abs(path) > 1e-6).sum(axis=0)
    print(f"  Lambda range: [{lambdas[-1]:.4f}, {lambdas[0]:.4f}]")
    print(f"  Sparsity at lambda_min : {sparsity_path[-1]}")
    print(f"  Sparsity at lambda_max : {sparsity_path[0]}")

    # ---- 3. Cross-validated lambda -----------------------------------------
    print("\nCross-validating lambda...")
    lam_cv, cv_errors = cross_validate_lambda(X, y, lambdas=lambdas, cv=5)
    print(f"  CV-selected lambda: {lam_cv:.6f}")
    la_cv = LassoAnalysis(lam=lam_cv, solver="cd").fit(X, y, beta_star=beta_star, sigma=sigma)
    print(la_cv.summary())
