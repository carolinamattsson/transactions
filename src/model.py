#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import heapq as hq
from dataclasses import dataclass
from decimal import Decimal
from scipy.stats import betabinom

from src.dists import paired_samples, random_unifs, scale_pareto, scale_pwl

def create_nodes(N, activity=1, attractivity=1, spending=0.5, burstiness=1, mean_iet=1, seed=321):
    '''
    Initialize node attributes from the given lists
    '''
    np.random.seed(seed)

    def list_len_N(value, name):
        if isinstance(value, (int, float, Decimal)):
            return np.full(N, value)
        elif isinstance(value, (list, np.ndarray)):
            assert len(value) == N, f"Please provide a list or array of length N for '{name}'."
            return np.array(value)
        else:
            raise TypeError(f"The '{name}' attribute must be a single value or a list/array of length N.")

    # Ensure attributes are correct length
    activity = list_len_N(activity, 'activity')
    attractivity = list_len_N(attractivity, 'attractivity')
    spending = list_len_N(spending, 'spending')
    burstiness = list_len_N(burstiness, 'burstiness')

    # Confirm valid attributes
    assert sum(activity) > 0, "The sum of activity values must be > 0."
    assert sum(attractivity) > 0, "The sum of attractivity values must be > 0."
    assert all(0 < spend <= 1 for spend in spending), "Spending values must be between 0 and 1."
    assert all(0 < burst for burst in burstiness), "Burstiness values must be > 0."
    assert mean_iet > 0, "Mean inter-event time must be > 0."

    # Convert activity potential into activity rate
    activity_rate_converter = 1 / mean_iet
    activity_rate = activity * activity_rate_converter

    # Normalize attractivities to sum to 1
    total_att = sum(attractivity)
    attractiveness = attractivity / total_att

    # Create node dictionary
    nodes = {i: {} for i in range(N)}
    for node in nodes:
        nodes[node]['act_pot'] = activity[node]
        nodes[node]['att_pot'] = attractivity[node]
        nodes[node]["act"] = activity_rate[node]
        nodes[node]["att"] = attractiveness[node]
        nodes[node]["spr"] = spending[node]
        nodes[node]["iet"] = (1. / (math.gamma(1 + 1 / burstiness[node])), burstiness[node])

    return nodes

def initialize_activations(nodes, mean_iet=1):
    '''
    Initialize the activation heap for the given nodes
    '''
    activations = [(activate(0, nodes[node]["act"], nodes[node]["iet"], mean_iet), node) for node in nodes]
    hq.heapify(activations)
    return activations


def activate(now,activity,distribution,mean_iet = 1,rng=np.random.default_rng()): 
    '''
    Get the next activation time for the given node
    '''
    # draw inter-event time from the relevant distribution
    scale_act = 1/activity # invert the activity
    scale_iet, k = distribution
    l = scale_act * scale_iet * mean_iet
    next = now + l * rng.weibull(a=k) 
    return next


@dataclass
class TargetSampler:
    '''
    Holds the precomputed state needed to draw a target node for a given source.

    Three decisions are decoupled from each other:
      - sample_self (here): does the sampling distribution include self?
      - record_self (in transact/interact): when self is drawn, do we emit a
        transaction or skip the activation?
      - storage strategy (here, via matrix_n_threshold in build_target_sampler):
        precomputed (N, N) cumulative-row matrix vs. shared (N,) cumulative
        attractivity vector. The matrix is the extension point for per-source
        heterogeneous attractivities; the shared vector exists so N > ~20k fits
        in memory.
    '''
    kind: str            # "matrix" or "resample"
    data: np.ndarray     # (N, N) cumulative-row array, or (N,) shared cumulative attractivity vector
    sample_self: bool
    n: int


def build_target_sampler(nodes, *, sample_self=False, matrix_n_threshold=20_000):
    '''
    Build a TargetSampler for picking transaction targets.

    Parameters
    ----------
    sample_self : bool, default False
        Whether self is in the support of the sampling distribution. With the
        matrix path, sample_self=False zeroes the diagonal so searchsorted will
        never return the source. With the resample path, it triggers rejection
        sampling (redraw until j != i).
    matrix_n_threshold : int, default 20_000
        N at or below which the precomputed (N, N) matrix is built. Above this,
        sampling falls back to an O(N)-memory shared cumulative vector. Set
        very high to force the matrix path, or 0 to force the resample path.
    '''
    N = len(nodes)
    attractivities = np.array([nodes[node]["att_pot"] for node in nodes])

    if N <= matrix_n_threshold:
        transition_matrix = np.zeros((N, N))
        if sample_self:
            for i in range(N):
                norm_factor = np.sum(attractivities)
                transition_matrix[i, :] = attractivities / norm_factor
        else:
            for i in range(N):
                available_nodes = np.delete(attractivities, i)
                norm_factor = np.sum(available_nodes)
                transition_matrix[i, :i] = available_nodes[:i] / norm_factor
                transition_matrix[i, i+1:] = available_nodes[i:] / norm_factor
        data = np.cumsum(transition_matrix, axis=1)
        kind = "matrix"
    else:
        # Shared cumulative attractivity vector; every source draws from the same
        # distribution. When sample_self=False, exclusion is enforced by rejection
        # in sample_target.
        data = np.cumsum(attractivities)
        kind = "resample"

    return TargetSampler(kind=kind, data=data, sample_self=sample_self, n=N)


def sample_target(sampler, i, rng):
    '''
    Draw a target node index for source node i.
    '''
    if sampler.kind == "matrix":
        cumrow = sampler.data[i]
        return int(np.searchsorted(cumrow, rng.random() * cumrow[-1], side='right'))
    # resample path
    cumvec = sampler.data
    if sampler.sample_self:
        return int(np.searchsorted(cumvec, rng.random() * cumvec[-1], side='right'))
    while True:
        j = int(np.searchsorted(cumvec, rng.random() * cumvec[-1], side='right'))
        if j != i:
            return j


def initialize_balances(nodes,balances=None,decimals=4):
    '''
    Initialize the balances for the given nodes
    '''
    # If the initial balances are not given, set them to the default value
    if balances is None:
        balances = np.ones(len(nodes))           # one unit of currency per node
    assert len(balances) == len(nodes), f"Please give a list or array that is the length of N for 'balances'."
    # create a dictionary of balances, keyed by node
    if decimals is not None:
        bal_vect = np.round(balances,decimals)
        bal_vect = [Decimal(f"{bal:.{decimals}f}") for bal in bal_vect]
    else:
        bal_vect = np.float64(balances)
    balances = {node:bal_vect[node] for node in nodes}
    return balances

 
def select(attractivities, current_node, rng=np.random.default_rng()):
    '''
    Select a node to transact with, ensuring no self-selection.
    
    Parameters:
    - attractivities: dict, keys are node IDs, values are probabilities
    - current_node: the node that is selecting (to avoid self-selection)
    - rng: random number generator (default: np.random.default_rng())
    
    Returns:
    - node_j: the selected node
    '''
    # Remove self from selection
    available_nodes = {k: v for k, v in attractivities.items() if k != current_node}
    
    # Normalize probabilities to sum to 1
    total_weight = sum(available_nodes.values())
    probabilities = [v / total_weight for v in available_nodes.values()]

    # Select target node
    node_j = rng.choice(list(available_nodes.keys()), p=probabilities)
    
    return node_j


def pay_random_share(node_i, node_j, balances, p, s, rng=np.random.default_rng()):
    '''
    Pay the selected node a random share of the available balance:
        - If the balance is continuous, the transaction size is a Beta sampled fraction.
        - If the balance is discrete, the transaction size is a Beta Binomial sample.

    '''
    beta_a, beta_b = p * s, (1 - p) * s
    # todo: 'a' and 'b' parametrized with balance and overdispersion parameter

    if isinstance(balances[node_i],Decimal):
        exp = balances[node_i].as_tuple().exponent # -(number of decimal places)
        n = int(balances[node_i].scaleb(-exp))
        txn_size_dist = betabinom(n,beta_a,beta_b) # integer valued distribution
        txn_size = txn_size_dist.rvs() # sample from the distribution
        # Note: betabinom.rvs() draws from scipy's global RNG, not the `rng` passed in.
        # Known reproducibility wart on the discrete-balance path; ignore unless seed-fixing matters.
        txn_size = Decimal(txn_size).scaleb(exp) # integer to decimal (e.g.: 1234 -> 12.34)
    else:
        txn_size = balances[node_i]*rng.beta(beta_a,beta_b)
    # process the transaction
    balances[node_i] -= txn_size
    balances[node_j] += txn_size
    # return the transaction details
    return txn_size


def pay_random_share_logitn(node_i, node_j, balances, p, tau, rng=np.random.default_rng()):
    '''
    Pay the selected node a random share of the available balance under
    logit-normal dispersion: q ~ LogitN(logit p, tau^2), transaction = q * balance.

        - If the balance is continuous, the transaction size is q * balance.
        - If the balance is discrete, the transaction size is Binomial(balance, q).

    Note on naming: 'p' here is the per-node spending propensity (the SM's s);
    'tau' is the new logit-scale dispersion parameter (parallel to BetaBin's xi).
    '''
    # draw q from LogitN(logit p, tau^2)
    z = rng.standard_normal()
    logit_p = math.log(p) - math.log1p(-p)
    q = 1.0 / (1.0 + math.exp(-(logit_p + tau * z)))

    if isinstance(balances[node_i], Decimal):
        exp = balances[node_i].as_tuple().exponent
        n = int(balances[node_i].scaleb(-exp))
        txn_size = int(rng.binomial(n, q))
        txn_size = Decimal(txn_size).scaleb(exp)
    else:
        txn_size = balances[node_i] * q
    balances[node_i] -= txn_size
    balances[node_j] += txn_size
    return txn_size


def pay_share(node_i, node_j, share, balances, rng=np.random.default_rng()):
    '''
    Pay the selected node a share of the available balance:
        - If the balance is continuous, the transaction size is a fixed fraction.
        - If the balance is discrete, the transaction size is a Binomial sample.

        Example:
        For balances[0] = Decimal('1234.56') with exp = -2:
            1234.56 -> 123456 (scaled), random sample -> 1100, rescaled -> 11.00 .
        For balances[1] = Decimal('123.4') with exp = -1:
            123.4 -> 1234 (scaled), random sample -> 12, rescaled -> 1.2 .
    '''
    # sample transaction weight
    if isinstance(balances[node_i],Decimal):
        exp = balances[node_i].as_tuple().exponent
        txn_size = rng.binomial(balances[node_i].scaleb(-exp),share) 
        txn_size = Decimal(txn_size).scaleb(exp)
    else:
        txn_size = balances[node_i]*share
    # process the transaction
    balances[node_i] -= txn_size
    balances[node_j] += txn_size
    # return the transaction size
    return txn_size


def transact(nodes, activations, sampler, balances, rng=np.random.default_rng(), *, record_self=False, record_zero=True, **kwargs):
    '''
    Simulate the next transaction using a TargetSampler.

    When sampler.sample_self is True and record_self is False, a self-draw is
    skipped: the source's next activation is scheduled but no transaction is
    emitted, and the function returns None. The realized transaction rate per
    node scales down by p_ii relative to record_self=True.

    When record_zero is False and the computed transaction amount is exactly
    zero, the activation is similarly advanced and None is returned. Zero-amount
    transactions can arise from (a) float64 balance underflow on hyperactive
    nodes — after enough repeated 0.8x decay the balance rounds to 0 and emits
    a stream of zero transactions, which are simulation artifacts; or (b)
    BetaBinomial / Binomial draws that legitimately yield zero, which the
    caller may or may not want to record.
    '''
    # Select next active node
    now, node_i = hq.heappop(activations)

    # Select target node via the configured sampling strategy
    node_j = sample_target(sampler, node_i, rng)

    # Skip-the-activation semantics: advance the heap but emit no transaction
    if sampler.sample_self and node_j == node_i and not record_self:
        next_activation = activate(now, nodes[node_i]["act"], nodes[node_i]["iet"], rng=rng)
        hq.heappush(activations, (next_activation, node_i))
        return None

    # Pay the selected node a share of the available balance
    p = nodes[node_i]["spr"]
    s = kwargs.get("s", None) # BetaBin precision (the SM's xi); None falls back to Binomial
    tau = kwargs.get("tau", None) # logit-normal dispersion
    dispersion = kwargs.get("dispersion", None) # 'logitnormal' selects pay_random_share_logitn

    if dispersion == "logitnormal" and tau is not None:
        amount = pay_random_share_logitn(node_i, node_j, balances, p, tau, rng=rng)
    elif s is not None:
        amount = pay_random_share(node_i, node_j, balances, p, s, rng=rng)
    else:
        amount = pay_share(node_i, node_j, nodes[node_i]["spr"], balances)

    # Skip-the-record semantics: advance the heap but emit no transaction.
    # The amount==0 mutations in pay_*() are no-ops, so balances are unchanged.
    if amount == 0 and not record_zero:
        next_activation = activate(now, nodes[node_i]["act"], nodes[node_i]["iet"], rng=rng)
        hq.heappush(activations, (next_activation, node_i))
        return None

    # Update the next activation time for the source node
    next_activation = activate(now, nodes[node_i]["act"], nodes[node_i]["iet"], rng=rng)
    hq.heappush(activations, (next_activation, node_i))

    # Return transaction details
    return {
        "timestamp": now,
        "source": node_i,
        "target": node_j,
        "amount": amount,
        "source_bal": balances[node_i],
        "target_bal": balances[node_j]
    }


def interact(nodes, activations, sampler, rng=np.random.default_rng(), *, record_self=False):
    '''
    Simulate the next interaction using a TargetSampler.

    Returns None when sampler.sample_self is True, record_self is False, and a
    self-draw occurred (the activation is still advanced).
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    node_j = sample_target(sampler, node_i, rng)
    # update the next activation time for the node
    next = activate(now, nodes[node_i]["act"], nodes[node_i]["iet"], rng=rng)
    hq.heappush(activations, (next, node_i))
    if sampler.sample_self and node_j == node_i and not record_self:
        return None
    return {"timestamp": now,
            "source": node_i,
            "target": node_j}


def run_interactions(N,T):
    '''
    Run the model to generate T interactions, printed to stdout
    '''
    # initialize the model
    nodes = create_nodes(N)
    sampler = build_target_sampler(nodes, sample_self=True)
    activations = initialize_activations(nodes)
    # print the output header
    header = ["timestamp","source","target","amount","source_bal","target_bal"]
    print(",".join(header))
    # run the model
    for i in range(T):
        interaction = interact(nodes, activations, sampler, record_self=True)
        print(",".join([str(interaction[term]) for term in header]))


def run_transactions(N,T):
    '''
    Run the model to generate T transactions, printed to stdout
    '''
    # initialize the model
    nodes = create_nodes(N)
    sampler = build_target_sampler(nodes, sample_self=True)
    activations = initialize_activations(nodes)
    balances = initialize_balances(nodes)
    # print the output header
    header = ["timestamp","source","target","amount","source_bal","target_bal"]
    print(",".join(header))
    # run the model
    for i in range(T):
        transaction = transact(nodes, activations, sampler, balances, record_self=True)
        print(",".join([str(transaction[term]) for term in header]))
