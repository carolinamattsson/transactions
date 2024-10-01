#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import heapq as hq
from decimal import Decimal

from methods.dists import paired_samples, random_unifs, scale_pareto, scale_pwl

def create_nodes(N, activity=1, attractivity=1, spending=0.5, burstiness=1, mean_iet=1):
    '''
    Initialize node attributes from the given lists.
    '''
    # Input sanitation
    def list_len_N(value, name):
        if isinstance(value, (int, float, Decimal)):
            return np.full(N, value)
        elif isinstance(value, (list, np.ndarray)):
            assert len(value) == N, f"Please give a list or array that is the length of N for '{name}'."
            return np.array(value)
        else:
            raise TypeError(f"The '{name}' attribute must be a single value or a list/array of length N.")
    # Ensure that the attributes are the correct length
    activity = list_len_N(activity, 'activity')
    attractivity = list_len_N(attractivity, 'attractivity')
    spending = list_len_N(spending, 'spending')
    burstiness = list_len_N(burstiness, 'burstiness')
    # Confirm that the attributes are valid
    assert sum(activity) > 0, "The sum of the activity values must be greater than 0."
    assert sum(attractivity) > 0, "The sum of the attractivity values must be greater than 0."
    assert all([0 < spend <= 1 for spend in spending]), "The spending values must be between 0 and 1, exclusive of zero."
    assert all([0 < burst for burst in burstiness]), "The burstiness values must be greater than zero."
    # scale activity to get the desired mean inter-event time
    assert mean_iet > 0, "The mean inter-event time must be greater than 1."
    mean_act = 1/mean_iet
    activity = mean_act*activity
    # scale attractivities to sum to one
    total_att = sum(attractivity)
    attractivity = attractivity/total_att
    # create dictionary of nodes
    # TODO: This could be changed into a networkx graph to allow for more complex interactions
    nodes = {i:{} for i in range(N)}
    for node in nodes:
        nodes[node]["act"] = activity[node]
        nodes[node]["att"] = attractivity[node]
        nodes[node]["pay"] = spending[node]
        nodes[node]["iet"] = (1./(math.gamma(1+1/burstiness[node])),burstiness[node])
    # return the node dictionary
    return nodes

def initialize_activations(nodes):
    '''
    Initialize the activation heap for the given nodes
    '''
    # create a min heap of activations, keyed by the activation time
    activations = [(activate(0,nodes[node]["act"],nodes[node]["iet"]), node) for node in nodes]
    hq.heapify(activations)
    return activations

def initialize_attractivities(nodes):
    '''
    Initialize the attractiveness dictionary for the given nodes
    '''
    # return the attractivity dictionary, keyed by node
    attractivities = {node:nodes[node]["att"] for node in nodes}
    return attractivities

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

def activate(now,activity,distribution,rng=np.random.default_rng()):
    '''
    Get the next activation time for the given node
    '''
    # draw inter-event time from the relevant distribution
    scale_act = 1/activity # invert the activity
    shape_iet, scale_iet = distribution
    next = now + scale_act*scale_iet*rng.weibull(a=shape_iet)
    return next

def select(attractivities,rng=np.random.default_rng()):
    '''
    Select a node to transact with
    '''
    # select target node
    # TODO: This would need to be changed with a networkx graph to allow for more complex interactions
    node_j = rng.choice(list(attractivities.keys()), p=list(attractivities.values()))
    return node_j

def pay_beta(node_i, node_j, params, balances, rng=np.random.default_rng()):
    '''
    Pay the given node
    '''
    # sample transaction weight
    beta_a, beta_b = params
    if isinstance(balances[node_i],Decimal):
        bal = float(balances[node_i])
        decimals = -balances[node_i].as_tuple().exponent
        edge_w = np.round(bal*rng.beta(beta_a,beta_b),decimals)
        edge_w = Decimal(f"{edge_w:.{decimals}f}")
    else:
        edge_w = balances[node_i]*np.random.beta(beta_a,beta_b)
    # process the transaction
    balances[node_i] -= edge_w
    balances[node_j] += edge_w
    # return the transaction details
    return edge_w

def pay_fraction(node_i, node_j, frac, balances):
    '''
    Pay the given node
    '''
    # sample transaction weight
    if isinstance(balances[node_i],Decimal):
        bal = float(balances[node_i])
        decimals = -balances[node_i].as_tuple().exponent
        edge_w = np.round(bal*frac,decimals)
        edge_w = Decimal(f"{edge_w:.{decimals}f}")
    else:
        edge_w = balances[node_i]*frac
    # process the transaction
    balances[node_i] -= edge_w
    balances[node_j] += edge_w
    # return the transaction details
    return edge_w

def pay_share(node_i, node_j, prob, balances):
    '''
    Pay the given node
    '''
    # sample transaction weight
    decimals = -balances[node_i].as_tuple().exponent
    txn_size = np.random.binomial(balances[node_i]*10**decimals,prob)/10**decimals
    txn_size = Decimal(f"{txn_size:.{decimals}f}")
    # process the transaction
    balances[node_i] -= txn_size
    balances[node_j] += txn_size
    # return the transaction size
    return txn_size

def interact(nodes,activations,attractivities,iet=np.random.exponential):
    '''
    Simulate the next interaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"],nodes[node_i]["iet"])
    hq.heappush(activations,(next, node_i))
    # return the transaction, the updated balances, and the updated activations
    return {"timestamp":now,
            "source":node_i,
            "target":node_j}

def transact(nodes,activations,attractivities,balances,method="share"):
    '''
    Simulate the next transaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # pay the target node
    if method=="beta":
        amount = pay_beta(node_i, node_j, nodes[node_i]["pay"], balances)
    elif method=="fraction":
        amount = pay_fraction(node_i, node_j, nodes[node_i]["pay"], balances)
    else:
        amount = pay_share(node_i, node_j, nodes[node_i]["pay"], balances)
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"],nodes[node_i]["iet"])
    hq.heappush(activations,(next, node_i))
    # return the transaction, the updated balances, and the updated activations
    return {"timestamp":now,
            "source":node_i,
            "target":node_j,
            "amount":amount,
            "source_bal":balances[node_i],
            "target_bal":balances[node_j]}

def run_interactions(N,T):
    '''
    Run the model to generate T interactions, printed to stdout
    '''
    # initialize the model
    nodes = create_nodes(N)
    activations = initialize_activations(nodes)
    attractivities = initialize_attractivities(nodes)
    # print the output header
    header = ["timestamp","source","target","amount","source_bal","target_bal"]
    print(",".join(header))
    # run the model
    for i in range(T):
        interaction = interact(nodes,activations,attractivities)
        print(",".join([str(interaction[term]) for term in header]))

def run_transactions(N,T):
    '''
    Run the model to generate T transactions, printed to stdout
    '''
    # initialize the model
    nodes = create_nodes(N)
    activations = initialize_activations(nodes)
    attractivities = initialize_attractivities(nodes)
    balances = initialize_balances(nodes)
    # print the output header
    header = ["timestamp","source","target","amount","source_bal","target_bal"]
    print(",".join(header))
    # run the model
    for i in range(T):
        transaction = transact(nodes,activations,attractivities,balances)
        print(",".join([str(transaction[term]) for term in header]))
