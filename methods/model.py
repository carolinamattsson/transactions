#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx 

from methods.dists import random_pwl, random_pwls_perturb, random_unifs, random_pwls

def initialize_ADmodel(N, scaling="pwl", coupling=False, **kwargs):
    '''
    Initialize AD network of size N with:
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
        else:
            raise ValueError("Scaling must be 'pwl' or 'unif'.")
    else:
        if scaling=="pwl":
            act_vect, attr_vect = random_pwls(N, **kwargs)
        elif scaling=="unif":
            act_vect, attr_vect = random_unifs(N, **{k: kwargs[k] for k in kwargs.keys() & {'copula', 'reversed', 'theta', 'resample'}})
        else:
            raise ValueError("Scaling must be 'pwl' or 'unif'.")
    # create list of nodes; attractivity is normalized; activity is scaled to model time
    nodes = [(i,{"act":act,"attr":attr,"attr_prob":attr_prob}) for i, act, attr, attr_prob in zip(range(N), act_vect, attr_vect, attr_vect/np.sum(attr_vect))]
    # create the network
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    return G

def activate(distribution=np.random.exponential,**kwargs):
    '''
    Get the next activation time for the given node
    '''
    # draw inter-event time from the relevant distribution
    iit = distribution(**kwargs) # need scale=scale for exponential
    return iit

def select(nodes):
    '''
    Select a node to transact with
    '''
    # select target node
    node_j = np.random.choice(nodes.keys, p=nodes.values)
    return node_j

def pay(balances, node_i, node_j ,distribution=np.random.beta, **kwargs):
    '''
    Pay the selected node
    '''
    # sample transaction weight
    edge_w = balances[node_i]*distribution(**kwargs) # need beta_a, beta_b for beta
    # process the transaction
    balances[node_i] -= edge_w
    balances[node_j] += edge_w
    # return the transaction details
    return edge_w

def transact(activations,balances):
    '''
    Update the model by one step
    '''
    # select next active node from the heap
    activation_time, node_i = activations.pop()
    # have the node select a target to transact with
    node_j = select(nodes) ### need to pass the probability distribution
    # pay the target node
    edge_w = pay(balances, node_i, node_j, beta_a, beta_b) ### need to pass the parameters
    # update the next activation time for the node
    activations.push(activate(), node_i) ### need to pass the parameter
    # return the transaction details
    return {"timestamp":edge_t,
            "source":node_i,
            "target":node_j,
            "amount":edge_w,
            "source_bal":bal[node_i],
            "target_bal":bal[node_j]}


def generate_ADtnet(init_G, tmax=7): # let's do a week's worth of transactions at once
    '''
    Run the AD model of the given set of nodes for the given length of time 
    Node activations are a poisson process, with a rate given by the activity of the node
    # leave open the possibility of using a memory network of some sort
    Return the AD temporal network
    '''
    # don't mess with the input graph
    G = init_G.copy()
    # retrieve the attractiveness values
    nodes, attr_prob = zip(*G.nodes('attr_prob'))
    # loop through nodes, adding their activations
    for node_i, node_act in G.nodes('act'):
        t_tot = 0
        scale = 1/node_act
        while(True):
            # exponential inter-event time
            t_tot += np.random.exponential(scale=scale)   # can we get the whole series of activations at once? ..idk
            if t_tot  > tmax:
                break
            # select target node
            node_j = np.random.choice(nodes, p=attr_prob)
            # add target node to 
            G.add_edge(node_i,node_j,time=t_tot)       # is there a reason to dis-allow self-loops?
    # return the temporal graph
    return G

def model(G, bal, beta_a=1, beta_b=1): # use a Beta distribution w/ default uniform fraction
    '''
    Run through the AD temporal network and allocate weights compatible with a transaction process
    The initial balance can/should be given
        -- the default is to give everyone the same amount of money
    The share of the balance to send in a transaction is drawn from a Beta distribution
        -- the default values for the parameters make it a uniform fraction
    Yeild the transaction data
    '''
    # Initilize balances that are not given to be 0
    missings = G.nodes - set(bal.keys())
    init_bal = {node_i:0 for node_i in missings} # we can't start with all zeros, bal is REQUIRED
    bal.update(init_bal)
    
    # retrieve the edges and their time
    temporal_edges = sorted(G.edges.data('time'), key=lambda nnt: nnt[2])
    
    # Loop though to sample for transaction weight and update balances
    for node_i, node_j, edge_t in temporal_edges:
        # sample transaction weight
        edge_w = bal[node_i]*np.random.beta(beta_a,beta_b) # update later # faster to generate all at once? 
        # return the transaction details
        yield (node_i,node_j,edge_t,edge_w)
        # process the transaction
        bal[node_i] -= edge_w
        bal[node_j] += edge_w