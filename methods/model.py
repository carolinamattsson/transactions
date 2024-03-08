#!/usr/bin/env python
# coding: utf-8

import numpy as np
import heapq as hq

from methods.dists import random_pwl, random_pwls_perturb, random_unifs, random_pwls

def create_nodes(N, scaling="const", coupling=False, **kwargs):
    '''
    Initialize the underlying of size N with:
    activity distr:  f(x,β) = β / x^(β+1) with possible scale & shift (see scipy.stats.pareto)
    if theta: activity/attractivity joint distribution with an available copula (see methods.dists.random_unifs)
    # leave open the possibility of adding global network constraints (e.g. SBM, ABM)
    '''
    # create activity and attractivity vectors    
    if not coupling:
        if scaling=="pwl":
            act_vect = random_pwl(N, **{k: kwargs[k] for k in kwargs.keys() & {'beta', 'loc', 'scale'}})
            attr_vect = act_vect
        elif scaling=="unif":
            act_vect = np.random.random(N)
            attr_vect = act_vect
        elif scaling=="const":
            act_vect = np.ones(N)
            attr_vect = act_vect
        else:
            raise ValueError("Scaling must be 'pwl' or 'unif' or 'const'.")
    else:
        if scaling=="pwl":
            act_vect, attr_vect = random_pwls(N, **kwargs)
        elif scaling=="unif":
            act_vect, attr_vect = random_unifs(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
        else:
            raise ValueError("Scaling must be 'pwl' or 'unif'. Cannot use 'const' with coupling.")
    # create dictionary of nodes
    nodes = {i:{} for i in range(N)}
    for node in nodes:
        nodes[node]["act"] = act_vect[node]
        nodes[node]["attr"] = attr_vect[node]
    # return the node dictionary
    return nodes

def initialize_activations(nodes,**kwargs):
    '''
    Initialize the activation heap for the given nodes
    '''
    # create a min heap of activations, keyed by the activation time
    activations = [(activate(0,nodes[node]["act"],**kwargs), node) for node in nodes]
    hq.heapify(activations)
    return activations

def initialize_attractivities(nodes,**kwargs):
    '''
    Initialize the attractivities dictionary for the given nodes
    '''
    # attrativity sums to one, keyed by node
    total = sum([nodes[node]["attr"] for node in nodes])
    attractivities = {node:nodes[node]["attr"]/total for node in nodes}
    return attractivities

def initialize_balances(nodes,monies=100,distribution=np.ones,**kwargs):
    '''
    Initialize the balances for the given nodes
    ''' 
    # create a dictionary of balances, keyed by node
    bal_vect = monies*distribution(len(nodes),**kwargs)
    balances = {node:bal_vect[node] for node in nodes}
    return balances

def activate(now,scale,distribution=np.random.exponential,**kwargs):
    '''
    Get the next activation time for the given node
    '''
    # draw inter-event time from the relevant distribution
    next = now + distribution(scale=scale,**kwargs)  # need to pass the parameter(s)
    return next

def select(attractivities):
    '''
    Select a node to transact with
    '''
    # select target node
    node_j = np.random.choice(list(attractivities.keys()), p=list(attractivities.values()))
    return node_j

def pay(node_i, node_j, balances, distribution=np.random.beta, beta_a=1, beta_b=1):
    '''
    Pay the given node
    '''
    # sample transaction weight
    edge_w = balances[node_i]*distribution(beta_a,beta_b) # need to pass the parameters
    # process the transaction
    balances[node_i] -= edge_w
    balances[node_j] += edge_w
    # return the transaction details
    return edge_w

def interact(nodes,activations,attractivities):
    '''
    Simulate the next interaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"])
    hq.heappush(activations,(next, node_i))
    # return the transaction, the updated balances, and the updated activations
    return {"timestamp":now,
            "source":node_i,
            "target":node_j}

def transact(nodes,activations,attractivities,balances,**kwargs):
    '''
    Simulate the next transaction
    '''
    # select next active node from the heap
    now, node_i = hq.heappop(activations)
    # have the node select a target to transact with
    node_j = select(attractivities)
    # pay the target node
    amount = pay(node_i, node_j, balances, **kwargs) 
    # update the next activation time for the node
    next = activate(now,nodes[node_i]["act"])
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
