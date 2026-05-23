#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy import stats

import pycop.simulation as cop

def paired_samples(N,
                   same=False,
                   params={'copula':None},
                   rng=None):
    '''
    Initialize activity and fitness values for N nodes, according to the specified distributions.
    By default, the values are independently sampled.
        Specify a copula and its parameters to sample correlated values from the respective distributions (see dists.py).
        Or, specify same_sample=True to use the same sample for both distributions.
    The parameters for the distributions are given as dictionaries.
        The options are 'pareto' or 'pwl' or 'uniform' or 'constant', with their relevant parameters.
    '''
    if rng is None:
        rng = np.random.default_rng()
    # create activity and attractivity distributions, together or separately
    unifs = {}
    if same or ('theta' in params and np.isinf(params['theta'])):
        unifs['act'] = rng.random(N)
        unifs['att'] = unifs['act']
    else:
        # unless a copula and its parameters are specified, the sampled distributions are independent
        unifs['act'], unifs['att'] = random_unifs(N, rng=rng, **params)
    # return the vectors
    return unifs['act'], unifs['att']

def powlaw_ppf(a, xmin=1, xmax=np.inf):
    if a <= 1:
        raise ValueError("Parameter 'a' must be greater than 1 for a valid power-law distribution.")

    # Precompute terms to use in the PPF calculation
    b = 1 - a
    c = (xmax**b - xmin**b) if xmax < np.inf else -xmin**b

    # Define the PPF as a function of the quantile r
    def ppf(r):
        if np.any((r < 0) | (r > 1)):
            raise ValueError("Input 'r' must be between 0 and 1.")
        return (r * c + xmin**b)**(1 / b)

    return ppf

def scale_pareto(unif, beta=2.0):
    '''
    Generate a vector size N with pareto distributed values
    Pareto: f(x,β) = β / x^(β+1) scaled & shifted such that the mean is 1
    The x_min corresponds to that on wikipedia, where alpha is used instead of beta
    '''
    assert beta > 1, "The 'beta' parameter must be greater than 1."
    if beta == np.inf:
        return np.ones(len(unif))
    x_min = (beta-1)/beta # getting the average activity to be 1
    pareto = stats.pareto(beta,scale=x_min)
    pwl = pareto.ppf(unif)
    # now return
    return pwl

def random_unifs(N, copula=None, reversed=False, theta=0, resample=100, rng=None):
    '''
    Generate two vectors size N with uniform distributed values coupled by the given copula
    Resample from the copula up to 'resample' numbers of times so there are no 1s in the reversed vector
    # nice one is reversed 'clayton' with theta=5

    Note: pycop's simu_archimedean uses numpy's global RNG. To keep this call
    deterministic from a passed-in rng, we seed the numpy global from the rng
    immediately before invocation. This affects the numpy global state.
    '''
    if rng is None:
        rng = np.random.default_rng()
    if copula is not None and theta != 0:
        sample = 0
        while sample < resample:
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            unif_1, unif_2 = cop.simu_archimedean(copula, 2, N, theta=theta)
            if not np.any(unif_1==0) and not np.any(unif_2==0):
                break
        if sample==resample:
            raise ValueError("Theta is too high, there are sampling issues. Use perturbation instead.")
        # now return them both
        if reversed:
            return np.subtract(1,unif_1), np.subtract(1,unif_2)  # <-- flipped
        else:
            return unif_1, unif_2
    else:
        return rng.random(N), rng.random(N)
