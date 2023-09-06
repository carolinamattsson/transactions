# Transactions

This is a mechanistic, stochastic, generative model for financial transactions as recorded within a hypothetical universal payment system. The simplest possible version of the model: a group of N identical nodes activate at a uniform rate in continuous time, send a transaction to a random other node, and fund this transaction with a random fraction of their present account balance.

The model itself has three modules:
* ACTIVATION. This module simulates node activation, given a node and its activity. 
* SELECTION. This module simulates the selection of a target for a transaction, given an activated node.
* PAYMENT. This module simulates a payment, given two nodes and the present account balances.

The three modules are strung together in ` `. Notably, this involves storing the next node activation for each node in a min heap by timestamp so that the present account balances can be used in the TRANSACTION and (crucially) PAYMENT step.

Using ` ` initializes the model and runs it. It is straightforward to run the model over many sequential chucks of time (e.g. simulating 52 weeks) when the activity distribution is poissonian (otherwise, it would need to be engineered differently).

Each step of the model has various input parameters; the defaults produce the simplest possible version.

Running the model requires a vector of initial balances. Initilaizing the model requires  



