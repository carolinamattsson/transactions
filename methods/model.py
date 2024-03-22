#!/usr/bin/env python
# coding: utf-8

import numpy as np
import heapq as hq
from decimal import Decimal

from methods.dists import random_pareto, random_paretos, random_pwl, random_pwls_perturb, random_unifs, random_pwls

def create_nodes(N, activity="const", coupling=False, **kwargs):
    '''
    Initialize the underlying of size N with:
    Activity distr:  f(x,β) = β / x^(β+1) with possible scale & shift (see scipy.stats.pareto)
    if theta: activity/attractivity joint distribution with an available copula (see methods.dists.random_unifs)
    # leave open the possibility of adding global network constraints (e.g. SBM, ABM)
    '''
    # create activity and attractivity vectors    
    if not coupling:
        if activity=="pareto":
            act_vect = random_pareto(N, **{k: kwargs[k] for k in kwargs.keys() & {'beta', 'mean_iet'}})
            attr_vect = act_vect
        elif activity=="pwl":
            act_vect = random_pwl(N, **{k: kwargs[k] for k in kwargs.keys() & {'beta', 'loc', 'scale'}})
            attr_vect = act_vect
        elif activity=="unif":
            act_vect = np.random.random(N)
            attr_vect = act_vect
        elif activity=="const":
            act_vect = (1/kwargs['mean_iet'])*np.ones(N) if 'mean_iet' in kwargs else np.ones(N)
            attr_vect = act_vect
        else:
            raise ValueError("Activity distribution must be 'pareto' or 'pwl' or 'unif' or 'const'.")
    else:
        if activity=="pareto":
            act_vect, attr_vect = random_paretos(N, **{k: kwargs[k] for k in kwargs.keys() & {'betas', 'means_iet', 'copula', 'reversed', 'theta', 'resample'}})
        if activity=="pwl":
            act_vect, attr_vect = random_pwls(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
        elif activity=="unif":
            act_vect, attr_vect = random_unifs(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
        else:
            raise ValueError("Activity distribution must be 'pareto' or 'pwl' or 'unif'. Cannot use 'const' with coupling.")
    # create dictionary of nodes
    nodes = {i:{} for i in range(N)}
    for node in nodes:
        nodes[node]["act"] = act_vect[node]
        nodes[node]["attr"] = attr_vect[node]
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
    # attrativity sums to one, keyed by node
    total = sum([nodes[node]["attr"] for node in nodes])
    attractivities = {node:nodes[node]["attr"]/total for node in nodes}
    return attractivities

def initialize_balances(nodes,balances=lambda x: 100*np.ones(x),decimals=None):
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
    mean_iet = 1/activity # invert the activity
    next = now + mean_iet*distribution()  # need to pass the parameter(s)
    return next

def select(attractivities):
    '''
    Select a node to transact with
    '''
    # select target node
    node_j = np.random.choice(list(attractivities.keys()), p=list(attractivities.values()))
    return node_j

def pay_fraction(node_i, node_j, balances, beta_a = 1, beta_b = 1):
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

def pay_share(node_i, node_j, balances, rate=0.5):
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

def transact(nodes,activations,attractivities,balances,iet=np.random.exponential,rate=0.5):
    '''
    Simulate the next transaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # pay the target node
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
