#!/usr/bin/env python
# coding: utf-8

import numpy as np
import heapq as hq
from decimal import Decimal

from methods.dists import random_unifs, scale_pareto, scale_pwl

def create_nodes(N, activity="constant", fitness="constant", same_sample=False, same_params=False, mean_iet=1, **kwargs):
    '''
    Initialize activity and fitness values for N nodes, according to the specified distributions.
    Specifying no arguments will result in all nodes having 1 for activity and fitness.
        Alternatives are: 'pareto' and 'uniform' and 'pwl' for these distributions.
    By default, the activity and fitness values are independently sampled.
        Specify a copula and its parameters to sample correlated values from the respective distributions (see dists.py).
        Or, specify same_sample=True to use the same sample for both distributions.
    By default, the parameters for the distributions are given separately in the format: {param}_act + {param}_fit.
        Specify same_params=True to use the same parameters for both distributions.
    By default, the mean inter-event time is 1.
        Specify a different mean inter-event time to scale the activity values accordingly.
        The fitness values are scaled to be probabilities.
    # leave open the possibility of adding global network constraints (e.g. SBM, ABM)
    '''
    assert mean_iet > 0, "The mean inter-event time must be greater than 1."
    # create activity and fitness spreads, together or separately
    unifs = {}
    if same_sample:
        unifs['act'] = np.random.random(N)
        unifs['fit'] = unifs['act']
    else:
        # unless a copula and its parameters are specified, the sampled distributions are independent
        unifs['act'], unifs['fit'] = random_unifs(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
    # assign the parameters for the distributions
    params = {}
    # scale up to the desired distributions
    vects = {}
    # different steps for the different distributions
    for dist, label in [(activity, "act"), (fitness, "fit")]:
        if dist=="constant":
            vects[label] = np.ones(N)
        elif dist=="uniform":
            vects[label] = unifs[label]
        elif dist=="pareto":
            if same_params:
                params[label] = kwargs['beta']
            else:
                params[label] = kwargs[f"beta_{label}"]
            vects[label] = scale_pareto(unifs[label], beta=params[label])
        elif dist=="pwl":
            if same_params:
                params[label] = {k: kwargs[k] for k in kwargs.keys() & {'beta', 'loc', 'scale'}}
            else:
                params[label] = {k: kwargs[k] for k in kwargs.keys() & {f"beta_{label}", f"loc_{label}", f"scale_{label}"}}
            vects[label] = scale_pwl(unifs[label], **params[label])
        else:
            raise ValueError("Activity and fitness distributions must be 'pareto' or 'pwl' or 'unif' or 'const'.")
    # scale activity to get the desired mean inter-event time
    mean_act = 1/mean_iet
    vects["act"] = mean_act*vects["act"]
    # scale fitness to sum to one
    total_fit = sum(vects["fit"])
    vects["fit"] = vects["fit"]/total_fit
    # create dictionary of nodes
    nodes = {i:{} for i in range(N)}
    for node in nodes:
        nodes[node]["act"] = vects["act"][node]
        nodes[node]["attr"] = vects["fit"][node]
    # return the node dictionary
    return nodes

def initialize_activations(nodes,iet=np.random.exponential):
    '''
    Initialize the activation heap for the given nodes
    '''
    # create a min heap of activations, keyed by the activation time
    activations = [(activate(0,nodes[node]["act"],iet), node) for node in nodes]
    hq.heapify(activations)
    return activations

def initialize_attractivities(nodes):
    '''
    Initialize the attractivities dictionary for the given nodes
    '''
    # return the attractivity dictionary, keyed by node
    attractivities = {node:nodes[node]["attr"] for node in nodes}
    return attractivities

def initialize_balances(nodes,balances=lambda x: 100*np.ones(x),decimals=2):
    '''
    Initialize the balances for the given nodes
    ''' 
    # create a dictionary of balances, keyed by node
    bal_vect = balances(len(nodes))
    if decimals is not None:
        bal_vect = np.round(bal_vect,decimals)
        bal_vect = [Decimal(f"{bal:.{decimals}f}") for bal in bal_vect]
    balances = {node:bal_vect[node] for node in nodes}
    return balances

def activate(now,activity,distribution):
    '''
    Get the next activation time for the given node
    '''
    # draw inter-event time from the relevant distribution
    scale_iet = 1/activity # invert the activity
    next = now + scale_iet*distribution()  # need to pass the parameter(s)
    return next

def select(attractivities):
    '''
    Select a node to transact with
    '''
    # select target node
    node_j = np.random.choice(list(attractivities.keys()), p=list(attractivities.values()))
    return node_j

def pay_beta(node_i, node_j, balances, beta_a = 1, beta_b = 1):
    '''
    Pay the given node
    '''
    # sample transaction weight
    if isinstance(balances[node_i],Decimal):
        bal = float(balances[node_i])
        decimals = -balances[node_i].as_tuple().exponent
        edge_w = np.round(bal*np.random.beta(beta_a,beta_b),decimals)
        edge_w = Decimal(f"{edge_w:.{decimals}f}")
    else:
        edge_w = balances[node_i]*np.random.beta(beta_a,beta_b)
    # process the transaction
    balances[node_i] -= edge_w
    balances[node_j] += edge_w
    # return the transaction details
    return edge_w

def pay_fraction(node_i, node_j, balances, frac = 0.5):
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

def pay_share(node_i, node_j, balances, rate = 0.5):
    '''
    Pay the given node
    '''
    # sample transaction weight
    decimals = -balances[node_i].as_tuple().exponent
    edge_w = np.random.binomial(balances[node_i]*10**decimals,rate)/10**decimals
    edge_w = Decimal(f"{edge_w:.{decimals}f}")
    # process the transaction
    balances[node_i] -= edge_w
    balances[node_j] += edge_w
    # return the transaction details
    return edge_w

def interact(nodes,activations,attractivities,iet=np.random.exponential):
    '''
    Simulate the next interaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"],iet)
    hq.heappush(activations,(next, node_i))
    # return the transaction, the updated balances, and the updated activations
    return {"timestamp":now,
            "source":node_i,
            "target":node_j}

def transact(nodes,activations,attractivities,balances,iet=np.random.exponential,method="share",rate=0.5,frac=0.5,beta_a=1,beta_b=1):
    '''
    Simulate the next transaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # pay the target node
    if method=="beta":
        amount = pay_beta(node_i, node_j, balances, beta_a=beta_a, beta_b=beta_b)
    elif method=="fraction":
        amount = pay_fraction(node_i, node_j, balances, frac=frac)
    else:
        amount = pay_share(node_i, node_j, balances, rate=rate)
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"],iet)
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
