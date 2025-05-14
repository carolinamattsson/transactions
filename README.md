# Transactions

This is a mechanistic, stochastic, generative model for financial transactions as recorded within a hypothetical universal payment system. The simplest possible version of the model is this: a group of N identical nodes activate as a memoryless point process in continuous time, send a transaction to a random other node, and fund this transaction with a sampled share of their present account balance.

The model itself has three modules:
* `activate` - this module simulates node activation, given the present time and a node's activity. 
* `select` - this module simulates selection of a target for a transaction, given node attractivities.
* `pay` - this module simulates a payment, given two nodes and the present account balances.

The three modules are strung together in `transact`, which simulates the next transaction. Notably, this involves storing the next node activation for each node in a min heap keyed by the timestamp so that transactions are simulated in time order.

The system being modelled consists of N nodes, for now with simply a value for each of activity and attractiveness. Initializing the model populates the heap with the first activations, normalizes the attractivity, and gives each node an initial balance.

The notebook `example.ipynb` contains a simple example of how to use the model. The model is run in a loop, simulating transactions until a specified time limit is reached. The results can be stored in csv or a DataFrame, which can be used for further analysis or visualization.

It is also possible to run the model in batch mode from the command line using `simulations.py`. See `tutorial.ipynb` for instructions.