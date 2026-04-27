"""
logitnormal.py — Poisson-LogitNormal conditional pmf for P(m), P(w)

Companion to numerical/distributions.py: provides the conditional pmf
under logit-normal dispersion,

    q ~ LogitN(logit s, tau^2),   w | M, q ~ Poisson(M q),

as the parallel-track replacement for the BetaBin/NB conditional.

The closed-form NB chain in SM §B.2/§C breaks under logit-normal, but the
variance-decomposition algebra survives (and is in fact cleaner: the
overdispersion contribution sigma^2_tau ≈ lambda_s^2 · tau^2 is independent
of s in the small-s, large-M limit).

Numerics
--------
The inner integral over the standard-normal residual z uses a fine trapezoidal
rule rather than Gauss-Hermite. GH puts most of its weight near z=0; at the
calibrated tau ~ 1, the integrand Poisson(w; M sigmoid(logit s + tau z)) varies
rapidly over the broad q-range that LogitN samples, and GH below n ≈ 200
leaves visible aliasing bumps. Trapezoidal on a fine z-grid (default n_z=400
on z ∈ [-8, 8]) is robust at large tau and roughly as cheap.

References
----------
- Companion doc .claude/logit_normal_companion.md §3 (P(m)), §4 (P(w))
- "Modeling financial transactions via random walks on temporal networks"
  Mattsson, Cellerini, Ojer & Starnini  (BetaBin route — SM §B.2/§C)
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import gammaln, logsumexp

# |logit q| clipped to this magnitude for float64 stability near s ≈ 0 or 1.
# Companion doc §7 (boundary effects).
LOGIT_CLIP = 30.0

# Default trapezoidal z-grid: ±8 sigma covers the standard normal to ~1e-15.
DEFAULT_N_Z = 400
DEFAULT_Z_MAX = 8.0


# =============================================================================
# Quadrature helpers
# =============================================================================

def gh_nodes(n_quad: int = 25):
    """
    Probabilist's Hermite nodes/weights, scaled so that
        sum_k weights[k] · f(z_k)  ≈  E_{Z ~ N(0,1)} f(Z).

    Kept for reference / sanity tests; the production path uses trap_nodes
    because GH aliases for tau on the order of 1 unless n_quad is large.
    """
    z, w = hermegauss(n_quad)
    return z, w / np.sqrt(2.0 * np.pi)


def trap_nodes(n_z: int = DEFAULT_N_Z, z_max: float = DEFAULT_Z_MAX):
    """
    Trapezoidal nodes/weights for E_{Z ~ N(0,1)} f(Z) ≈ sum_k w_k f(z_k),
    with z_k uniformly spaced on [-z_max, z_max] and w_k = phi(z_k) · dz.

    For z_max=8 and n_z=400 this matches scipy.integrate.quad to ~1e-12 on
    smooth integrands — entirely sufficient for our pmf evaluations.
    """
    z = np.linspace(-z_max, z_max, n_z)
    dz = z[1] - z[0]
    w = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * dz
    return z, w


def logitn_q_quadrature(s: float, tau: float,
                        n_z: int = DEFAULT_N_Z, z_max: float = DEFAULT_Z_MAX):
    """
    Return (q_nodes, weights) realizing q ~ LogitN(logit s, tau^2) via
    trapezoidal in z with logit q = logit s + tau · z.

    The logit values are clipped to |logit q| <= LOGIT_CLIP so that the
    sigmoid is well-behaved at float64.
    """
    if s <= 0.0 or s >= 1.0:
        raise ValueError(f"s must be in (0, 1); got {s}")
    z, w = trap_nodes(n_z, z_max)
    logit_s = np.log(s) - np.log1p(-s)
    logit_q = np.clip(logit_s + tau * z, -LOGIT_CLIP, LOGIT_CLIP)
    q = 1.0 / (1.0 + np.exp(-logit_q))
    return q, w


# =============================================================================
# Poisson-LogitNormal compound pmf
# =============================================================================

def log_pln_compound(k: int, lambdas: np.ndarray, weights: np.ndarray) -> float:
    """
    log of the discrete compound  sum_j weights[j] · Poisson(k; lambdas[j]).

    Stable via log-sum-exp; ignores entries with lambda <= 0.
    """
    if k < 0:
        return -np.inf
    valid = lambdas > 0.0
    if not np.any(valid):
        return -np.inf if k > 0 else 0.0
    lam = lambdas[valid]
    w = weights[valid]
    log_pois = k * np.log(lam) - lam - gammaln(k + 1)
    return float(logsumexp(log_pois + np.log(w)))


def log_pm_cond_pln(m: int, lambda_s: float, s: float, tau: float,
                    n_z: int = DEFAULT_N_Z, z_max: float = DEFAULT_Z_MAX) -> float:
    """
    log p(m | s, tau) for the equilibrium occupancy under logit-normal jumps.

    Uses the small-s, large-M approximation in which the per-activation arrival
    rate scales linearly with the realized jump fraction:
        lambda(q, s) = lambda_s · q / s,
    so that the conditional mean is preserved at lambda_s when q = s.

    Companion doc §3.1.
    """
    if lambda_s <= 0.0:
        return -np.inf if m > 0 else 0.0
    q, weights = logitn_q_quadrature(s, tau, n_z=n_z, z_max=z_max)
    lam = lambda_s * q / s
    return log_pln_compound(m, lam, weights)


def log_pw_cond_pln(w: int, M: float, s: float, tau: float,
                    n_z: int = DEFAULT_N_Z, z_max: float = DEFAULT_Z_MAX) -> float:
    """
    log p(w | M, s, tau) for w ~ Poisson(M q), q ~ LogitN(logit s, tau^2).

    Companion doc §4.1.
    """
    if M <= 0.0:
        return -np.inf if w > 0 else 0.0
    q, weights = logitn_q_quadrature(s, tau, n_z=n_z, z_max=z_max)
    lam = M * q
    return log_pln_compound(w, lam, weights)


# =============================================================================
# Closed-form moments (delta-method, first order in tau)
# =============================================================================

def pln_var_q(s: float, tau: float) -> float:
    """
    First-order Var(q | s, tau) ≈ tau^2 · [s(1-s)]^2.

    Companion doc §1. Used as the variance target for the per-account scaling
    plot in dispersion_alternatives.ipynb F2g (slope-2 in s(1-s) on log-log).
    """
    return (tau ** 2) * (s * (1.0 - s)) ** 2


def pln_var_w(M: float, s: float, tau: float) -> float:
    """
    First-order Var(w | M, s, tau) ≈ M s + (M s)^2 · tau^2 · (1-s)^2.

    Companion doc §4.1. The first term is the Poisson contribution; the
    second is the overdispersion contribution.
    """
    Ms = M * s
    return Ms + (Ms ** 2) * (tau ** 2) * ((1.0 - s) ** 2)
