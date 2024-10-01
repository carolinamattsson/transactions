#!/usr/bin/env python
# coding: utf-8


import sys
import math
import random
import numpy as np
from scipy import stats

import pycop.simulation as cop

def paired_samples(N, 
                   same=False, 
                   params={'copula':None},
                   rng=np.random.default_rng()):
    '''
    Initialize activity and fitness values for N nodes, according to the specified distributions.
    By default, the values are independently sampled.
        Specify a copula and its parameters to sample correlated values from the respective distributions (see dists.py).
        Or, specify same_sample=True to use the same sample for both distributions.
    The parameters for the distributions are given as dictionaries.
        The options are 'pareto' or 'pwl' or 'uniform' or 'constant', with their relevant parameters.
    '''
    # create activity and attractivity distributions, together or separately
    unifs = {}
    if same or ('theta' in params and np.isinf(params['theta'])):
        unifs['act'] = rng.random(N)
        unifs['att'] = unifs['act']
    else:
        # unless a copula and its parameters are specified, the sampled distributions are independent
        unifs['act'], unifs['att'] = random_unifs(N, **params)
    # return the vectors
    return unifs['act'], unifs['att']

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

def scale_pwl(unif, beta=1.0, loc=0, scale=1):
    '''
    Generate a vector size N with pareto distributed values 
    pwl distrs: f(x,β) = β / x^(β+1) with possible scale & shift (for details see scipy.stats.pareto)
    '''
    pareto = stats.pareto(beta,loc=loc,scale=scale)
    pwl = pareto.ppf(unif)
    # now return
    return pwl

def random_unifs(N, copula=None, reversed=False, theta=0, resample=100, rng=np.random.default_rng()):
    '''
    Generate two vectors size N with uniform distributed values coupled by the given copula
    Resample from the copula up to 'resample' numbers of times so there are no 1s in the reversed vector
    # nice one is reversed 'clayton' with theta=5
    '''
    if copula is not None and theta != 0:
        sample = 0
        while sample < resample: 
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

def random_pwls_perturb(N, pwl_beta=1.0, pwl_loc=0, pwl_scale=1, per_beta=1, per_std=0):
    '''
    Generate a vector size N with pareto distributed values and one with proportional, gaussian perturbation
    primary distr: f(x,β) = β / x^(β+1) with possible scale & shift (for details see scipy.stats.pareto)
    permuted distr: b = a^β2 + a^β2 * Normal(0,var) 
    '''
    pwl = stats.pareto.rvs(pwl_beta, size=N, loc=pwl_loc, scale=pwl_scale)
    pwl_p = np.multiply(np.power(pwl,per_beta),(1+stats.norm.rvs(loc=0,scale=per_std,size=N)))
    if np.any(pwl_p<0):
        raise ValueError("Too much perturbation, there are negative values. Use a copula.")
    # now return them both
    return pwl, pwl_p

def random_paretos(N, betas=(2.0,2.0), means_iet=(1,1), **kwargs):
    '''
    Generate a vector size N with pareto distributed values 
    Pareto: f(x,β) = β / x^(β+1) scaled & shifted such that the mean is 'mean'
    The x_min corresponds to that on wikipedia, where alpha is used instead of beta
    '''
    assert betas[0] > 1, "The shape parameter must be greater than 1."
    assert betas[1] > 1, "The shape parameter must be greater than 1."
    assert means_iet[0] > 0, "The mean inter-event time must be greater than 1."
    assert means_iet[1] > 0, "The mean inter-event time must be greater than 1."
    means_act = (1/means_iet[0], 1/means_iet[1])
    x_min = ((betas[0]-1)/betas[0], (betas[1]-1)/betas[1]) # getting the average activity to be 'mean'
    if betas[0] == np.inf and betas[1] == np.inf:
        return np.full(N, means_act[0]), np.full(N, means_act[1])
    elif betas[0] == np.inf:
        return np.full(N, means_act[0]), random_pareto(N, betas[1], means_iet[1])
    elif betas[1] == np.inf:
        return random_pareto(N, betas[0], means_iet[0]), np.full(N, means_act[1])
    else:
        unif_1, unif_2 = random_unifs(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
        pareto_1 = stats.pareto(betas[0],scale=means_act[0]*x_min[0])
        pareto_2 = stats.pareto(betas[1],scale=means_act[1]*x_min[1])
        pwl_1 = pareto_1.ppf(unif_1)
        pwl_2 = pareto_2.ppf(unif_2)
        # now return them both
        return pwl_1, pwl_2

def random_pwls(N, beta=1.0, loc=0, scale=1, beta_2=1.0, loc_2=0, scale_2=1, **kwargs):
    '''
    Generate two vectors size N with pareto distributed values, starting from the given uniform distributions
    pwl distrs: f(x,β) = β / x^(β+1) with possible scale & shift (for details see scipy.stats.pareto)
    '''
    unif_1, unif_2 = random_unifs(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
    pareto_1 = stats.pareto(beta,loc=loc,scale=scale)
    pareto_2 = stats.pareto(beta_2,loc=loc_2,scale=scale_2)
    pwl_1 = pareto_1.ppf(unif_1)
    pwl_2 = pareto_2.ppf(unif_2)
    # now return them both
    return pwl_1, pwl_2

def rnd_pwl(xmin, xmax, g, size=1):
    r = np.random.random(size=size)
    return (r*( xmax**(1.-g) - xmin**(1.-g) ) + xmin**(1.-g) )**(1./(1.-g))

def rnd_pwl_beta(beta, c0=1, size=1):
    r = np.random.random(size=size)
    return c0*((1.0-r)**(-1.0/beta))

def rnd_intertime(alpha, c, size=1, tmax=sys.float_info.max/2):
    '''Invert function f(t) = \alphac (ct+1)^-(\alpha+1)'''
    r = np.random.random(size=size)
    r = r*( 1. - (c*tmax + 1)**-alpha )
    return ((1.-r)**(-1./alpha)-1)/c

def balance_weibull(N, shape=1, mean=100):
    '''Scale a weibull to the given mean. Sample N values.'''
    scale = mean/(math.gamma(1+1/shape))
    return scale*np.random.weibull(shape,size=N)