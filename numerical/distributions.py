"""
occupancy_distributions.py - Numerical Integration for P(m) and P(w)

This module computes the occupancy distribution P(m) and transaction size 
distribution P(w) for random walkers on activity-driven temporal networks,
as described in:

    "Modeling financial transactions via random walks on temporal networks"
    Mattsson, Cellerini, Ojer & Starnini

The key distributions are:

    P(m) = ∫ NB(m; r_eff(s), p(s)) h(s) ds       [Eq. 29 / 30 of SM]
    
    P(w) = ∫∫∫ Pois(w; ρκ) Gamma(κ) Beta(ρ) dρ dκ h(s) ds   [Eq. 43 of SM]

where NB uses the (n, p) parametrization with n = r and p = r/(r+λ).

Two sources of overdispersion are modeled:
    - Global overdispersion (r parameter): correlation from co-jumping,
      where walkers arriving together create correlated occupancy.
      r(s) = (1 - ⟨s⟩) / s  [Eq. 16 of SM]
    
    - Local overdispersion (ξ parameter): variance in individual spending
      behavior modeled by Beta-Binomial transactions.
      Combined via: 1/r_eff = 1/r + 1/ξ  [Eq. 20 of SM]

Numerical methods:
    - Log-space arithmetic with log-sum-exp for stability
    - Trapezoidal integration on log-spaced grids
    - Grid sizes: n_s=5000 for 1D integrals; n_s=200, n_κ=100, n_ρ=50 for P(w)

Author: Carolina Mattsson (with code assistance from Claude, Anthropic)
"""

import numpy as np
from scipy.special import gammaln, betaln
from typing import Dict, List, Optional, Tuple
import warnings


# =============================================================================
# Parameter Computation
# =============================================================================

def compute_s_mean(sigma: float, s_min: float, s_max: float) -> float:
    """
    Compute ⟨s⟩_h, the mean of s under h(s) ∝ s^σ.
    
    ⟨s⟩_h = ∫ s · h(s) ds / ∫ h(s) ds
    
    Parameters
    ----------
    sigma : float
        Exponent in h(s) ∝ s^σ
    s_min, s_max : float
        Bounds of spending rate distribution
        
    Returns
    -------
    float
        Mean spending rate ⟨s⟩_h
    """
    sigma = float(sigma)
    
    # Numerator: ∫ s^(σ+1) ds
    if np.isclose(sigma, -2):
        numer = np.log(s_max / s_min)
    else:
        numer = (s_max**(sigma + 2) - s_min**(sigma + 2)) / (sigma + 2)
    
    # Denominator: ∫ s^σ ds
    if np.isclose(sigma, -1):
        denom = np.log(s_max / s_min)
    else:
        denom = (s_max**(sigma + 1) - s_min**(sigma + 1)) / (sigma + 1)
    
    return numer / denom


def compute_s_inv_mean(sigma: float, s_min: float, s_max: float) -> float:
    """
    Compute ⟨s⁻¹⟩_h, the mean of 1/s under h(s) ∝ s^σ.
    
    Used for K₀ = ⟨m⟩ / ⟨s⁻¹⟩_h.
    
    Parameters
    ----------
    sigma : float
        Exponent in h(s) ∝ s^σ
    s_min, s_max : float
        Bounds of spending rate distribution
        
    Returns
    -------
    float
        Mean of inverse spending rate ⟨s⁻¹⟩_h
    """
    sigma = float(sigma)
    
    # Numerator: ∫ s^(σ-1) ds
    if np.isclose(sigma, 0):
        numer = np.log(s_max / s_min)
    else:
        numer = (s_max**sigma - s_min**sigma) / sigma
    
    # Denominator: ∫ s^σ ds
    if np.isclose(sigma, -1):
        denom = np.log(s_max / s_min)
    else:
        denom = (s_max**(sigma + 1) - s_min**(sigma + 1)) / (sigma + 1)
    
    return numer / denom


def compute_params(sigma: float, s_min: float, s_max: float, 
                   N: int, M: int) -> Dict:
    """
    Compute all derived parameters for the model.
    
    Parameters
    ----------
    sigma : float
        Exponent in h(s) ∝ s^σ (σ=0 is uniform)
    s_min, s_max : float
        Bounds of spending rate distribution
    N : int
        Number of nodes
    M : int
        Number of walkers (units of currency)
        
    Returns
    -------
    dict
        Dictionary containing all model parameters:
        - sigma, s_min, s_max, N, M: inputs
        - mean_m: average occupancy M/N
        - s_mean: ⟨s⟩_h
        - s_inv_mean: ⟨s⁻¹⟩_h  
        - K0: scale parameter = mean_m / s_inv_mean
        - h_norm: normalization for h(s)
        - crossover_m: m value where power-law tail begins
    """
    sigma = float(sigma)
    s_min, s_max = float(s_min), float(s_max)
    mean_m = M / N
    
    s_mean = compute_s_mean(sigma, s_min, s_max)
    s_inv_mean = compute_s_inv_mean(sigma, s_min, s_max)
    K0 = mean_m / s_inv_mean
    
    # h(s) normalization: ∫ s^σ ds
    if np.isclose(sigma, -1):
        h_norm = np.log(s_max / s_min)
    else:
        h_norm = (s_max**(sigma + 1) - s_min**(sigma + 1)) / (sigma + 1)
    
    # Crossover: where tail scaling P(m) ~ m^{-(σ+2)} begins
    crossover_m = K0 / s_max
    
    return {
        'sigma': sigma,
        's_min': s_min,
        's_max': s_max,
        'N': N,
        'M': M,
        'mean_m': mean_m,
        's_mean': s_mean,
        's_inv_mean': s_inv_mean,
        'K0': K0,
        'h_norm': h_norm,
        'crossover_m': crossover_m
    }


def print_params(params: Dict, xi: Optional[float] = None) -> None:
    """Print parameters in a readable format."""
    print(f"Model parameters:")
    print(f"  σ = {params['sigma']} (spending rate distribution h(s) ∝ s^σ)")
    print(f"  s_min = {params['s_min']:.2e}, s_max = {params['s_max']}")
    print(f"  N = {params['N']:.2e} nodes, M = {params['M']:.2e} walkers")
    print(f"  ⟨m⟩ = M/N = {params['mean_m']:.2f}")
    print(f"  ⟨s⟩_h = {params['s_mean']:.6f}")
    print(f"  ⟨s⁻¹⟩_h = {params['s_inv_mean']:.6f}")
    print(f"  K₀ = ⟨m⟩/⟨s⁻¹⟩_h = {params['K0']:.4f}")
    print(f"  Crossover m* ≈ K₀/s_max = {params['crossover_m']:.1f}")
    print(f"  Expected tail: P(m) ~ m^{-(params['sigma'] + 2):.1f} for m >> m*")
    if xi is not None:
        xi_str = '∞' if np.isinf(xi) else f'{xi}'
        print(f"  ξ = {xi_str} (local overdispersion from Beta-Binomial)")


# =============================================================================
# Log-space PMF Functions
# =============================================================================

def log_poisson_pmf(m: int, lam: float) -> float:
    """
    Log of Poisson PMF: log[λ^m e^{-λ} / m!]
    
    Stable for large m and λ.
    """
    if lam <= 0:
        return -np.inf if m > 0 else 0.0
    return m * np.log(lam) - lam - gammaln(m + 1)


def log_nb_pmf(m: int, r: float, p: float) -> float:
    """
    Log of Negative Binomial PMF in (n, p) parametrization.
    
    NB(m; r, p) = Γ(m+r) / (m! Γ(r)) · p^r · (1-p)^m
    
    This matches scipy.stats.nbinom convention:
        - r (often called n): shape/dispersion parameter
        - p: "success probability" 
        - mean = r(1-p)/p
        - variance = r(1-p)/p²
    
    To convert from mean λ: p = r/(r+λ), giving mean = λ.
    
    Parameters
    ----------
    m : int
        Count value (number of walkers)
    r : float
        Shape parameter (r → ∞ recovers Poisson)
    p : float
        Probability parameter, p = r/(r+λ) where λ is the mean
        
    Returns
    -------
    float
        Log of the PMF value
    """
    if r <= 0 or p <= 0 or p >= 1:
        return -np.inf
    
    return (gammaln(m + r) - gammaln(m + 1) - gammaln(r)
            + r * np.log(p) + m * np.log(1 - p))


def log_h(s: float, params: Dict) -> float:
    """
    Log of normalized h(s) = s^σ / h_norm.
    """
    if s <= 0:
        return -np.inf
    return params['sigma'] * np.log(s) - np.log(params['h_norm'])


def log_gamma_pdf(kappa: float, r: float, lambda_s: float) -> float:
    """
    Log of Gamma PDF with shape r and mean λ_s.
    
    Gamma(κ; r, r/λ_s) — rate parametrization with rate β = r/λ_s
    
    This is the mixing distribution for NB as Poisson-Gamma mixture.
    Mean = λ_s, Variance = λ_s²/r.
    
    Parameters
    ----------
    kappa : float
        The random variable (κ in SM Eq. 43)
    r : float
        Shape parameter
    lambda_s : float
        Mean parameter λ_s = K₀/s
    """
    if kappa <= 0 or r <= 0 or lambda_s <= 0:
        return -np.inf
    
    rate = r / lambda_s
    return (r * np.log(rate) - gammaln(r) 
            + (r - 1) * np.log(kappa) - rate * kappa)


def log_beta_pdf(rho: float, alpha: float, beta: float) -> float:
    """
    Log of Beta PDF.
    
    Beta(ρ; α, β) = ρ^{α-1} (1-ρ)^{β-1} / B(α, β)
    
    For Beta-Binomial thinning: α = ξs, β = ξ(1-s).
    
    Parameters
    ----------
    rho : float
        The random variable (ρ in SM Eq. 43)
    alpha, beta : float
        Shape parameters
    """
    if rho <= 0 or rho >= 1 or alpha <= 0 or beta <= 0:
        return -np.inf
    
    return ((alpha - 1) * np.log(rho) 
            + (beta - 1) * np.log(1 - rho) 
            - betaln(alpha, beta))


# =============================================================================
# Overdispersion Parameters
# =============================================================================

def compute_r_global(s: float, s_mean: float) -> float:
    """
    Compute global overdispersion parameter r(s).
    
    r(s) = (1 - ⟨s⟩_h) / s   [SM Eq. 16]
    
    This captures correlation from co-jumping: walkers that jump together
    to the same destination create correlated occupancy. The form r ∝ 1/s
    ensures constant intra-class correlation across node types.
    
    Physical interpretation (homogeneous case s = ⟨s⟩):
        r = (1-s)/s is the ratio of stay-probability to jump-probability.
    
    Parameters
    ----------
    s : float
        Spending rate of the node
    s_mean : float
        Population mean ⟨s⟩_h
        
    Returns
    -------
    float
        Global overdispersion parameter r(s)
    """
    return (1 - s_mean) / s


def compute_r_effective(s: float, s_mean: float, xi: float) -> float:
    """
    Compute effective overdispersion parameter combining global and local.
    
    1/r_eff = 1/r + 1/ξ   [SM Eq. 20]
    
    where:
        - r = (1-⟨s⟩)/s is global overdispersion (from co-jumping)
        - ξ is local overdispersion (from Beta-Binomial transactions)
    
    The two sources contribute independently to variance, so their
    inverse shape parameters add (harmonic combination).
    
    Parameters
    ----------
    s : float
        Spending rate of the node
    s_mean : float
        Population mean ⟨s⟩_h
    xi : float
        Beta-Binomial precision (ξ → ∞ is Binomial limit)
        
    Returns
    -------
    float
        Effective overdispersion parameter r_eff(s)
    """
    r_global = compute_r_global(s, s_mean)
    
    if np.isinf(xi):
        # Binomial limit: only global contribution
        return r_global
    else:
        # Harmonic combination: 1/r_eff = 1/r + 1/ξ
        return 1.0 / (1.0/r_global + 1.0/xi)


# =============================================================================
# Stable Numerical Integration
# =============================================================================

def _integrate_logspace(log_integrand_func, grid: np.ndarray) -> float:
    """
    Integrate exp(log_integrand) using stable log-space arithmetic.
    
    Algorithm:
    1. Evaluate log(integrand) on grid
    2. Find maximum for numerical stability
    3. Compute exp(log_val - max) (rescaled integrand)
    4. Trapezoidal integration of rescaled function
    5. Multiply result by exp(max)
    
    This handles integrands spanning many orders of magnitude.
    
    Parameters
    ----------
    log_integrand_func : callable
        Function returning log of integrand at each grid point
    grid : ndarray
        Integration grid (typically log-spaced)
        
    Returns
    -------
    float
        The integral value
    """
    log_vals = np.array([log_integrand_func(x) for x in grid])
    
    valid = np.isfinite(log_vals)
    if not np.any(valid):
        return 0.0
    
    log_max = np.max(log_vals[valid])
    
    rescaled = np.zeros_like(log_vals)
    rescaled[valid] = np.exp(log_vals[valid] - log_max)
    
    integral_rescaled = np.trapezoid(rescaled, grid)
    
    return integral_rescaled * np.exp(log_max)


# =============================================================================
# P(m) Computation
# =============================================================================

def Pm_single(m: int, params: Dict, model: str = 'nb',
              xi: Optional[float] = None, n_s: int = 5000) -> float:
    """
    Compute P(m) for a single occupancy value.
    
    P(m) = ∫ p(m|s) h(s) ds
    
    where p(m|s) is either:
        - Poisson(λ_s) for independent walkers
        - NB(r_eff(s), p(s)) accounting for overdispersion
    
    Parameters
    ----------
    m : int
        Occupancy value (number of walkers on node)
    params : dict
        Model parameters from compute_params()
    model : str
        'nb' for Negative Binomial, 'poisson' for Poisson
    xi : float or None
        Local overdispersion parameter. None or np.inf means
        global overdispersion only (Binomial transactions).
    n_s : int
        Number of grid points for integration (default 5000)
        
    Returns
    -------
    float
        Probability P(m)
    """
    s_min = params['s_min']
    s_max = params['s_max']
    K0 = params['K0']
    s_mean = params['s_mean']
    
    if xi is None:
        xi = np.inf
    
    s_grid = np.logspace(np.log10(s_min), np.log10(s_max), n_s)
    
    if model == 'poisson':
        def log_integrand(s):
            lambda_s = K0 / s
            return log_poisson_pmf(m, lambda_s) + log_h(s, params)
    else:  # nb
        def log_integrand(s):
            lambda_s = K0 / s
            r_eff = compute_r_effective(s, s_mean, xi)
            p = r_eff / (r_eff + lambda_s)  # NB probability parameter
            return log_nb_pmf(m, r_eff, p) + log_h(s, params)
    
    return _integrate_logspace(log_integrand, s_grid)


def compute_Pm(m_values: np.ndarray, params: Dict,
               xi: Optional[float] = None,
               include_poisson: bool = True,
               verbose: bool = True, n_s: int = 5000) -> Dict:
    """
    Compute P(m) for an array of occupancy values.
    
    Parameters
    ----------
    m_values : array-like
        Array of m values to compute
    params : dict
        Model parameters from compute_params()
    xi : float or None
        Local overdispersion parameter (None = global only)
    include_poisson : bool
        Whether to compute Poisson reference
    verbose : bool
        Print progress
    n_s : int
        Grid points for integration
        
    Returns
    -------
    dict
        Results containing:
        - m: input values
        - P_nb: NB probabilities (with specified xi)
        - P_poisson: Poisson probabilities (if requested)
        - params: copy of parameters
        - xi: the xi value used
    """
    m_values = np.asarray(m_values, dtype=int)
    n_m = len(m_values)
    
    if xi is None:
        xi = np.inf
    
    P_nb = np.zeros(n_m)
    P_poisson = np.zeros(n_m) if include_poisson else None
    
    if verbose:
        print_params(params, xi)
        print(f"\nComputing P(m) for {n_m} values...")
    
    for i, m in enumerate(m_values):
        P_nb[i] = Pm_single(m, params, model='nb', xi=xi, n_s=n_s)
        if include_poisson:
            P_poisson[i] = Pm_single(m, params, model='poisson', n_s=n_s)
        
        if verbose and (i + 1) % max(1, n_m // 10) == 0:
            print(f"  {100*(i+1)/n_m:5.1f}% complete (m={m})")
    
    if verbose:
        print("  Done!")
    
    results = {
        'm': m_values,
        'P_nb': P_nb,
        'xi': xi,
        'params': params
    }
    if include_poisson:
        results['P_poisson'] = P_poisson
    
    return results


def compute_Pm_xi_sweep(m_values: np.ndarray, params: Dict,
                        xi_values: List[float],
                        include_poisson: bool = True,
                        verbose: bool = True, n_s: int = 5000) -> Dict:
    """
    Compute P(m) for multiple ξ values.
    
    Parameters
    ----------
    m_values : array-like
        Array of m values
    params : dict
        Model parameters
    xi_values : list
        List of ξ values (use np.inf for global-only)
    include_poisson : bool
        Include Poisson reference
    verbose : bool
        Print progress
    n_s : int
        Grid points
        
    Returns
    -------
    dict
        Results with P_m[xi] arrays for each xi
    """
    m_values = np.asarray(m_values, dtype=int)
    n_m = len(m_values)
    
    P_m = {xi: np.zeros(n_m) for xi in xi_values}
    P_poisson = np.zeros(n_m) if include_poisson else None
    
    if verbose:
        print_params(params)
        xi_strs = ['∞' if np.isinf(xi) else str(xi) for xi in xi_values]
        print(f"\nComputing P(m) for {n_m} values, ξ ∈ {{{', '.join(xi_strs)}}}")
    
    for i, m in enumerate(m_values):
        if include_poisson:
            P_poisson[i] = Pm_single(m, params, model='poisson', n_s=n_s)
        
        for xi in xi_values:
            P_m[xi][i] = Pm_single(m, params, model='nb', xi=xi, n_s=n_s)
        
        if verbose and (i + 1) % max(1, n_m // 10) == 0:
            print(f"  {100*(i+1)/n_m:5.1f}% complete (m={m})")
    
    if verbose:
        print("  Done!")
    
    results = {
        'm': m_values,
        'xi_values': xi_values,
        'P_m': P_m,
        'params': params
    }
    if include_poisson:
        results['P_poisson'] = P_poisson
    
    return results


# =============================================================================
# P(w) Computation — Global Overdispersion Only (ξ = ∞)
# =============================================================================

def Pw_single_global(w: int, params: Dict, n_s: int = 5000) -> float:
    """
    Compute P(w) with global overdispersion only (Binomial thinning).
    
    For NB occupancy + Binomial thinning:
        P(w) = ∫ NB(w; r(s), p_w) h(s) ds
    
    where the thinned mean is s·λ_s = s·(K₀/s) = K₀ (independent of s),
    but r(s) = (1-⟨s⟩)/s still varies.
    
    Parameters
    ----------
    w : int
        Transaction size (number of jumping walkers)
    params : dict
        Model parameters
    n_s : int
        Grid points for integration
        
    Returns
    -------
    float
        Probability P(w)
    """
    s_min = params['s_min']
    s_max = params['s_max']
    K0 = params['K0']
    s_mean = params['s_mean']
    
    s_grid = np.logspace(np.log10(s_min), np.log10(s_max), n_s)
    
    def log_integrand(s):
        r_s = compute_r_global(s, s_mean)
        # After Binomial thinning: mean is K0, shape is still r_s
        p = r_s / (r_s + K0)
        return log_nb_pmf(w, r_s, p) + log_h(s, params)
    
    return _integrate_logspace(log_integrand, s_grid)


def Pw_poisson(w: int, params: Dict) -> float:
    """
    Compute P(w) for Poisson (analytical).
    
    For Poisson occupancy + Binomial thinning, the s-dependence cancels:
        P(w) = Poisson(w; K₀)
    
    Parameters
    ----------
    w : int
        Transaction size
    params : dict
        Model parameters
        
    Returns
    -------
    float
        Probability P(w) = Poisson(K₀)
    """
    return np.exp(log_poisson_pmf(w, params['K0']))


# =============================================================================
# P(w) Computation — With Local Overdispersion (finite ξ)
# =============================================================================

def Pw_single_local(w: int, params: Dict, xi: float,
                    n_kappa: int = 100, n_rho: int = 50, n_s: int = 200,
                    kappa_range_sigmas: float = 6.0) -> float:
    """
    Compute P(w) with local overdispersion via Poisson-Gamma-Beta integral.
    
    From SM Eq. 43:
        P(w) = ∫_s h(s) [∫∫ Pois(w; ρκ) Gamma(κ; r_eff, λ_s) 
                              Beta(ρ; ξs, ξ(1-s)) dρ dκ] ds
    
    The inner 2D integral uses the mixture representations:
        - NB = Poisson-Gamma mixture
        - BetaBinomial = Binomial-Beta mixture
    
    Parameters
    ----------
    w : int
        Transaction size
    params : dict
        Model parameters
    xi : float
        Local overdispersion (Beta-Binomial precision)
    n_kappa : int
        Grid points for κ (Gamma) integration
    n_rho : int
        Grid points for ρ (Beta) integration
    n_s : int
        Grid points for s integration
    kappa_range_sigmas : float
        How many std deviations to span for κ grid
        
    Returns
    -------
    float
        Probability P(w)
    """
    if np.isinf(xi):
        return Pw_single_global(w, params, n_s=n_s*25)  # Use finer grid
    
    s_min = params['s_min']
    s_max = params['s_max']
    K0 = params['K0']
    s_mean = params['s_mean']
    
    s_grid = np.logspace(np.log10(s_min), np.log10(s_max), n_s)
    log_w_fact = gammaln(w + 1)
    
    def inner_integral(s):
        """Compute the 2D integral over (κ, ρ) for fixed s."""
        lambda_s = K0 / s
        r_eff = compute_r_effective(s, s_mean, xi)
        alpha_s = xi * s
        beta_s = xi * (1 - s)
        
        # Set up κ grid centered on mean λ_s with spread based on variance
        kappa_std = lambda_s / np.sqrt(max(r_eff, 0.1))
        kappa_min = max(lambda_s - kappa_range_sigmas * kappa_std, 1e-10)
        kappa_max = lambda_s + kappa_range_sigmas * kappa_std
        kappa_grid = np.logspace(np.log10(max(kappa_min, 1e-10)),
                                  np.log10(kappa_max), n_kappa)
        
        # ρ grid on (0, 1), avoiding edges
        rho_grid = np.linspace(0.001, 0.999, n_rho)
        
        # Meshgrid for vectorized computation
        KAPPA, RHO = np.meshgrid(kappa_grid, rho_grid, indexing='ij')
        
        # Log of integrand components (vectorized)
        mu = RHO * KAPPA
        log_pois = np.where(mu > 0, w * np.log(mu) - mu - log_w_fact, -np.inf)
        
        rate = r_eff / lambda_s
        log_gamma = (r_eff * np.log(rate) - gammaln(r_eff)
                     + (r_eff - 1) * np.log(KAPPA) - rate * KAPPA)
        
        log_beta = ((alpha_s - 1) * np.log(RHO) 
                    + (beta_s - 1) * np.log(1 - RHO)
                    - betaln(alpha_s, beta_s))
        
        log_vals = log_pois + log_gamma + log_beta
        
        # Stable integration
        finite_mask = np.isfinite(log_vals)
        if not np.any(finite_mask):
            return 0.0
        
        log_max = np.max(log_vals[finite_mask])
        integrand = np.exp(log_vals - log_max)
        integrand[~finite_mask] = 0.0
        
        # Trapezoidal: first over ρ, then over κ
        integral_rho = np.trapezoid(integrand, rho_grid, axis=1)
        integral = np.trapezoid(integral_rho, kappa_grid)
        
        return np.exp(log_max) * integral
    
    # Outer integral over s
    def log_integrand_s(s):
        inner = inner_integral(s)
        if inner <= 0:
            return -np.inf
        return np.log(inner) + log_h(s, params)
    
    return _integrate_logspace(log_integrand_s, s_grid)


def compute_Pw(w_values: np.ndarray, params: Dict,
               xi: Optional[float] = None,
               include_poisson: bool = True,
               verbose: bool = True,
               n_kappa: int = 100, n_rho: int = 50, n_s: int = 200) -> Dict:
    """
    Compute P(w) for an array of transaction sizes.
    
    Parameters
    ----------
    w_values : array-like
        Array of w values
    params : dict
        Model parameters
    xi : float or None
        Local overdispersion (None or np.inf = global only)
    include_poisson : bool
        Include Poisson reference
    verbose : bool
        Print progress
    n_kappa, n_rho, n_s : int
        Grid points for integration
        
    Returns
    -------
    dict
        Results containing P_nb and optionally P_poisson
    """
    w_values = np.asarray(w_values, dtype=int)
    n_w = len(w_values)
    
    if xi is None:
        xi = np.inf
    
    P_nb = np.zeros(n_w)
    P_poisson = np.zeros(n_w) if include_poisson else None
    
    if verbose:
        print_params(params, xi)
        print(f"\nComputing P(w) for {n_w} values...")
        if not np.isinf(xi):
            print(f"  Using Poisson-Gamma-Beta integration "
                  f"(n_κ={n_kappa}, n_ρ={n_rho}, n_s={n_s})")
    
    for i, w in enumerate(w_values):
        if include_poisson:
            P_poisson[i] = Pw_poisson(w, params)
        
        if np.isinf(xi):
            P_nb[i] = Pw_single_global(w, params)
        else:
            P_nb[i] = Pw_single_local(w, params, xi,
                                       n_kappa=n_kappa, n_rho=n_rho, n_s=n_s)
        
        if verbose:
            print(f"  w={w}: P_nb={P_nb[i]:.4e}")
    
    if verbose:
        print("  Done!")
    
    results = {
        'w': w_values,
        'P_nb': P_nb,
        'xi': xi,
        'params': params
    }
    if include_poisson:
        results['P_poisson'] = P_poisson
    
    return results


def compute_Pw_xi_sweep(w_values: np.ndarray, params: Dict,
                        xi_values: List[float],
                        include_poisson: bool = True,
                        verbose: bool = True,
                        n_kappa: int = 100, n_rho: int = 50, 
                        n_s: int = 200) -> Dict:
    """
    Compute P(w) for multiple ξ values.
    
    Parameters
    ----------
    w_values : array-like
        Array of w values
    params : dict
        Model parameters
    xi_values : list
        List of ξ values
    include_poisson : bool
        Include Poisson reference
    verbose : bool
        Print progress
    n_kappa, n_rho, n_s : int
        Grid points
        
    Returns
    -------
    dict
        Results with P_w[xi] for each xi
    """
    w_values = np.asarray(w_values, dtype=int)
    n_w = len(w_values)
    
    xi_finite = [xi for xi in xi_values if not np.isinf(xi)]
    xi_inf = [xi for xi in xi_values if np.isinf(xi)]
    
    P_w = {xi: np.zeros(n_w) for xi in xi_values}
    P_poisson = np.zeros(n_w) if include_poisson else None
    
    if verbose:
        print_params(params)
        xi_strs = ['∞' if np.isinf(xi) else str(xi) for xi in xi_values]
        print(f"\nComputing P(w) for {n_w} values, ξ ∈ {{{', '.join(xi_strs)}}}")
        if xi_finite:
            print(f"  Finite ξ: Poisson-Gamma-Beta integration "
                  f"(n_κ={n_kappa}, n_ρ={n_rho}, n_s={n_s})")
    
    for i, w in enumerate(w_values):
        if verbose:
            print(f"  w = {w} ({i+1}/{n_w})...")
        
        if include_poisson:
            P_poisson[i] = Pw_poisson(w, params)
        
        # Global-only (ξ = ∞) — fast 1D integral
        for xi in xi_inf:
            P_w[xi][i] = Pw_single_global(w, params)
        
        # Finite ξ — slow 2D integral
        for xi in xi_finite:
            P_w[xi][i] = Pw_single_local(w, params, xi,
                                          n_kappa=n_kappa, n_rho=n_rho, n_s=n_s)
            if verbose:
                print(f"    ξ={xi}: P(w)={P_w[xi][i]:.4e}")
    
    if verbose:
        print("  Done!")
    
    results = {
        'w': w_values,
        'xi_values': xi_values,
        'P_w': P_w,
        'params': params
    }
    if include_poisson:
        results['P_poisson'] = P_poisson
    
    return results


# =============================================================================
# Analytical Formula (Poisson, large m)
# =============================================================================

def Pm_poisson_analytical(m: int, params: Dict) -> float:
    """
    Compute P(m) for Poisson using analytical incomplete gamma formula.
    
    From SM Eq. 18:
        P(m) ∝ [Γ(a, x_min) - Γ(a, x_max)] / m!
    
    where a = m - σ - 1, x_min = K₀/s_max, x_max = K₀/s_min.
    
    Requires mpmath for numerical stability at large m.
    
    Parameters
    ----------
    m : int
        Occupancy value
    params : dict
        Model parameters
        
    Returns
    -------
    float
        Probability P(m)
    """
    try:
        from mpmath import mp, mpf, gamma, gammainc, fac
        mp.dps = 50
    except ImportError:
        warnings.warn("mpmath not available, using numerical integration")
        return Pm_single(m, params, model='poisson')
    
    sigma = params['sigma']
    K0 = params['K0']
    s_min = params['s_min']
    s_max = params['s_max']
    h_norm = params['h_norm']
    
    a = m - sigma - 1
    if a <= 0:
        return 0.0
    
    x_min = mpf(K0) / mpf(s_max)
    x_max = mpf(K0) / mpf(s_min)
    
    try:
        a_mp = mpf(a)
        gamma_a = gamma(a_mp)
        
        upper_x_min = gamma_a - gammainc(a_mp, 0, x_min, regularized=False)
        upper_x_max = gamma_a - gammainc(a_mp, 0, x_max, regularized=False)
        
        factorial_m = fac(m)
        K0_mp = mpf(K0)
        h_norm_mp = mpf(h_norm)
        
        P_m = ((upper_x_min - upper_x_max) * K0_mp**(-sigma - 1) 
               / (factorial_m * h_norm_mp))
        
        return max(0.0, float(P_m))
    except Exception:
        return 0.0


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing occupancy_distributions module")
    print("=" * 70)
    
    # Test parameters matching SM
    N = 10**4
    M = 10**7
    s_max = 0.1
    s_min = s_max / N
    sigma = 0  # uniform h(s)
    
    params = compute_params(sigma, s_min, s_max, N, M)
    print_params(params)
    
    # Test P(m)
    print("\n" + "=" * 70)
    print("Test: P(m) for a few values")
    print("=" * 70)
    
    m_test = np.array([50, 100, 500, 1000])
    
    results = compute_Pm(m_test, params, xi=np.inf, verbose=False)
    
    print(f"\n{'m':>6} | {'P_nb (ξ=∞)':>12} | {'P_poisson':>12} | {'ratio':>8}")
    print("-" * 50)
    for i, m in enumerate(m_test):
        ratio = results['P_nb'][i] / results['P_poisson'][i]
        print(f"{m:>6} | {results['P_nb'][i]:>12.4e} | "
              f"{results['P_poisson'][i]:>12.4e} | {ratio:>8.4f}")
    
    # Test P(m) with local overdispersion
    print("\n" + "=" * 70)
    print("Test: P(m) with local overdispersion (ξ = 10)")
    print("=" * 70)
    
    results_xi = compute_Pm(m_test, params, xi=10, verbose=False)
    
    print(f"\n{'m':>6} | {'P_nb (ξ=∞)':>12} | {'P_nb (ξ=10)':>12} | {'ratio':>8}")
    print("-" * 50)
    for i, m in enumerate(m_test):
        ratio = results_xi['P_nb'][i] / results['P_nb'][i]
        print(f"{m:>6} | {results['P_nb'][i]:>12.4e} | "
              f"{results_xi['P_nb'][i]:>12.4e} | {ratio:>8.4f}")
    
    # Test P(w)
    print("\n" + "=" * 70)
    print("Test: P(w) for a few values")
    print("=" * 70)
    
    w_test = np.array([1, 5, 10, 20])
    
    results_w = compute_Pw(w_test, params, xi=np.inf, verbose=False)
    
    print(f"\n{'w':>6} | {'P_nb (ξ=∞)':>12} | {'P_poisson':>12}")
    print("-" * 40)
    for i, w in enumerate(w_test):
        print(f"{w:>6} | {results_w['P_nb'][i]:>12.4e} | "
              f"{results_w['P_poisson'][i]:>12.4e}")
    
    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
