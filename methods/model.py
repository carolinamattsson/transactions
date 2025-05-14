#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import heapq as hq
from decimal import Decimal
from scipy.stats import betabinom

from methods.dists import paired_samples, random_unifs, scale_pareto, scale_pwl

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

def initialize_activations(nodes, mean_iet = 1):
    '''
    Initialize the activation heap for the given nodes
    '''
    # create a min heap of activations, keyed by the activation time
    # activations = [(activate(0,nodes[node]["act"],nodes[node]["iet"]), node) for node in nodes] 
    activations = [(activate(0,nodes[node]["act"],nodes[node]["iet"],mean_iet), node) for node in nodes] #added mean_iet
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


def initialize_transition_matrix(nodes, self_selection=False):
    '''
    Precompute transition matrix with or without self-avoiding selection.
    '''
    N = len(nodes)
    attractivities = np.array([nodes[node]["att_pot"] for node in nodes])
    transition_matrix = np.zeros((N, N))
    if self_selection:
        for i in range(N):
            available_nodes = attractivities
            norm_factor = np.sum(available_nodes)
            transition_matrix[i, :] = available_nodes / norm_factor  # Normalize all nodes
    else:
        for i in range(N):
            available_nodes = np.delete(attractivities, i)  # Remove self
            norm_factor = np.sum(available_nodes)
            transition_matrix[i, :i] = available_nodes[:i] / norm_factor  # Before self
            transition_matrix[i, i+1:] = available_nodes[i:] / norm_factor  # After self

    return transition_matrix


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
        txn_size = Decimal(txn_size).scaleb(exp) # integer to decimal (e.g.: 1234 -> 12.34)
    else:
        txn_size = balances[node_i]*rng.beta(beta_a,beta_b)
    # process the transaction
    balances[node_i] -= txn_size
    balances[node_j] += txn_size
    # return the transaction details
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


def transact(nodes, activations, transition_matrix, balances, rng=np.random.default_rng(), **kwargs):
    '''
    Simulate the next transaction using a precomputed transition matrix.
    '''
    # Select next active node
    now, node_i = hq.heappop(activations)

    # Select target node using the precomputed transition matrix
    node_j = rng.choice(len(nodes), p=transition_matrix[node_i])

    # Pay the selected node a share of the available balance
    p = nodes[node_i]["spr"]
    s = kwargs.get("s", None) # Default to None, i.e. Binomial, if not provided
    if s is not None:
        amount = pay_random_share(node_i, node_j, balances, p, s, rng=rng)
    else:
        amount = pay_share(node_i, node_j, nodes[node_i]["spr"], balances)
    
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


def interact(nodes,activations,transition_matrix,rng=np.random.default_rng()):
    '''
    Simulate the next interaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # Select target node using the precomputed transition matrix
    node_j = rng.choice(len(nodes), p=transition_matrix[node_i])
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"],nodes[node_i]["iet"],rng=rng)
    hq.heappush(activations,(next, node_i))
    # return the transaction, the updated balances, and the updated activations
    return {"timestamp":now,
            "source":node_i,
            "target":node_j}


def run_interactions(N,T):
    '''
    Run the model to generate T interactions, printed to stdout
    '''
    # initialize the model
    nodes = create_nodes(N)
    transitions = initialize_transition_matrix(nodes, self_selection=True)
    activations = initialize_activations(nodes)
    # print the output header
    header = ["timestamp","source","target","amount","source_bal","target_bal"]
    print(",".join(header))
    # run the model
    for i in range(T):
        interaction = interact(nodes,activations,transitions)
        print(",".join([str(interaction[term]) for term in header]))


def run_transactions(N,T):
    '''
    Run the model to generate T transactions, printed to stdout
    '''
    # initialize the model
    nodes = create_nodes(N)
    transitions = initialize_transition_matrix(nodes, self_selection=True)
    activations = initialize_activations(nodes)
    balances = initialize_balances(nodes)
    # print the output header
    header = ["timestamp","source","target","amount","source_bal","target_bal"]
    print(",".join(header))
    # run the model
    for i in range(T):
        transaction = transact(nodes,activations,transitions,balances)
        print(",".join([str(transaction[term]) for term in header]))
