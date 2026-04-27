"""
numerical_pm_pw.py — P(m) and P(w) marginals under logit-normal dispersion

Outer integrals over h(s) ∝ s^σ paralleling Pm_single in
numerical/distributions.py, with the BetaBin/NB conditional pmf swapped for
the Poisson-LogitNormal conditional from numerical/logitnormal.py.

P(m) follows the SM's reduced form (eq. 28): P(m) = ∫ p(m|s) h(s) ds. The
conditional p(m|s) is the Poisson-LogitN compound implemented in
log_pm_cond_pln.

P(w) follows the SM's compound form (eq. 624):
    P(w) = ∑_m ∫_s p(w|m,s) p(m|s) h(s) ds.
For BetaBin this admits the closed form Pw_single_local in
numerical/distributions.py (SM eq. 43, Poisson-Gamma-Beta integral). For
LogitN there is no clean closed form, so Pw_logitn_mc evaluates the
compounding by Monte Carlo over the joint (s, q, m, q', w).

References
----------
- Companion doc .claude/logit_normal_companion.md §3 (P(m)) and §4 (P(w))
- numerical/distributions.py — the BetaBin/NB sibling pipeline
"""

import numpy as np
from typing import Dict

from numerical.distributions import (
    log_h,
    _integrate_logspace,
    log_poisson_pmf,
)
from numerical.logitnormal import log_pm_cond_pln


# =============================================================================
# P(m) under logit-normal
# =============================================================================

def Pm_single_logitn(m: int, params: Dict, tau: float,
                     n_s: int = 5000, n_z: int = 400) -> float:
    """
    P(m) = ∫ p(m | s, tau) h(s) ds   under the logit-normal conditional.

    Conditional rate (companion §3.1, small-s/large-M approximation):
        lambda(q, s) = lambda_s · q / s,   lambda_s = K_0 / s,
    so the conditional mean is preserved at lambda_s when q = s.
    """
    s_min = params['s_min']
    s_max = params['s_max']
    K0 = params['K0']

    s_grid = np.logspace(np.log10(s_min), np.log10(s_max), n_s)

    def log_integrand(s):
        lambda_s = K0 / s
        return log_pm_cond_pln(m, lambda_s, s, tau, n_z=n_z) + log_h(s, params)

    return _integrate_logspace(log_integrand, s_grid)


def compute_Pm_logitn(m_values: np.ndarray, params: Dict, tau: float,
                      include_poisson: bool = True,
                      verbose: bool = True,
                      n_s: int = 5000, n_z: int = 400) -> Dict:
    """
    Array wrapper around Pm_single_logitn, mirroring compute_Pm signature.

    Returns
    -------
    dict with keys
        m : (K,) int array of input m values
        P_logitn : (K,) ndarray
        P_poisson : (K,) ndarray (if include_poisson)
        tau : float
        params : dict
    """
    m_values = np.asarray(m_values, dtype=int)
    n_m = len(m_values)

    P_logitn = np.zeros(n_m)
    P_poisson = np.zeros(n_m) if include_poisson else None

    if verbose:
        print(f"Computing P(m) under logit-normal (tau={tau}) for {n_m} values...")

    for i, m in enumerate(m_values):
        P_logitn[i] = Pm_single_logitn(int(m), params, tau,
                                       n_s=n_s, n_z=n_z)
        if include_poisson:
            P_poisson[i] = _Pm_single_poisson(int(m), params, n_s=n_s)
        if verbose and (i + 1) % max(1, n_m // 10) == 0:
            print(f"  {100*(i+1)/n_m:5.1f}% complete (m={m})")

    if verbose:
        print("  Done!")

    results = {
        'm': m_values,
        'P_logitn': P_logitn,
        'tau': tau,
        'params': params,
    }
    if include_poisson:
        results['P_poisson'] = P_poisson
    return results


def _Pm_single_poisson(m: int, params: Dict, n_s: int = 5000) -> float:
    """Poisson reference (tau → 0 limit). Mirrors Pm_single(model='poisson')."""
    s_min = params['s_min']
    s_max = params['s_max']
    K0 = params['K0']

    s_grid = np.logspace(np.log10(s_min), np.log10(s_max), n_s)

    def log_integrand(s):
        return log_poisson_pmf(m, K0 / s) + log_h(s, params)

    return _integrate_logspace(log_integrand, s_grid)


# =============================================================================
# P(w) under logit-normal — SM-style compounding via Monte Carlo
# =============================================================================

def Pw_logitn_mc(w_grid: np.ndarray, params: Dict, tau: float,
                 n_mc: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """
    Marginal P(w) under logit-normal, computed by Monte Carlo over the full
    SM compounding chain:

        s        ~ h(s) ∝ s^σ on [s_min, s_max]
        q  | s   ~ LogitN(logit s, tau²)                [equilibrium jump prob]
        m  | s,q ~ Poisson(K_0 · q / s²)                [equilibrium occupancy]
        q' | s   ~ LogitN(logit s, tau²)                [fresh draw at activation]
        w  | m,q'~ Poisson(m · q')                      [Bin → Poisson at large m]

    Each row implements one factor of SM eq. 624,
        P(w) = ∑_m ∫_s p(w|m, s) p(m|s) h(s) ds,
    with the LogitN-specific p(m|s) and p(w|m, s) replacing the BetaBin/NB
    counterparts. The closed-form NB chain in SM §C does not generalize to
    LogitN, hence MC.

    Parameters
    ----------
    w_grid : (W,) array of integer w values to evaluate
    params : output of numerical.distributions.compute_params
    tau    : logit-scale dispersion
    n_mc   : MC sample count (default 1e6 — gives sub-1% noise on log-binned
             histograms across ~8 decades)
    seed   : RNG seed for reproducibility

    Returns
    -------
    P : (len(w_grid),) ndarray, the integer pmf evaluated at each w in w_grid.
    """
    rng = np.random.default_rng(seed)
    sigma = params['sigma']
    s_min, s_max, K0 = params['s_min'], params['s_max'], params['K0']

    # Inverse-CDF sampling for h(s) ∝ s^σ on [s_min, s_max]
    u = rng.uniform(0.0, 1.0, n_mc)
    if np.isclose(sigma, -1.0):
        s = s_min * (s_max / s_min) ** u
    else:
        a = sigma + 1.0
        s = (u * (s_max ** a - s_min ** a) + s_min ** a) ** (1.0 / a)

    logit_s = np.log(s) - np.log1p(-s)

    # q | s
    z1 = rng.standard_normal(n_mc)
    q = 1.0 / (1.0 + np.exp(-(logit_s + tau * z1)))

    # m | s, q  ~  Poisson(K_0 q / s²)
    m = rng.poisson(K0 * q / (s * s))

    # q' | s, independent of q
    z2 = rng.standard_normal(n_mc)
    q_prime = 1.0 / (1.0 + np.exp(-(logit_s + tau * z2)))

    # w | m, q'  ~  Poisson(m q')
    w_samples = rng.poisson(m * q_prime)

    w_max_grid = int(w_grid.max())
    counts = np.bincount(w_samples, minlength=w_max_grid + 1)
    pmf = counts / n_mc
    return pmf[np.asarray(w_grid, dtype=int)]
