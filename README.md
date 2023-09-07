# Transactions

This is a mechanistic, stochastic, generative model for financial transactions as recorded within a hypothetical universal payment system. The simplest possible version of the model: a group of N identical nodes activate as a memoryless point process in continuous time, send a transaction to a random other node, and fund this transaction with a random fraction of their present account balance.

The model itself has three modules:
* ACTIVATION. This module simulates node activation, given the present time and a node's activity. 
* SELECTION. This module simulates the selection of a target for a transaction, given the attractivity of all nodes.
* PAYMENT. This module simulates a payment, given two nodes and the present account balances.

The three modules are strung together in `transact`, which simulates the next transaction. Notably, this involves storing the next node activation for each node in a min heap keyed by the timestamp so that transactions are simulated in time order.

The system being modelled consists of N nodes, for now with simply a value for each of activity and attractiveness. Initializing the model populates the heap with the first activations, normalizes the attractivity, and gives each node an initial balance.

Using `run_default(N,T)` initializes the model with N nodes and uses it to simulate T transactions. This is simply an example of how to call the model, as it prints transactions straight to stout.