import os
import sys
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import psutil
import matplotlib.pyplot as plt
import gc

# Add directories to Python path
projdir = os.path.abspath(os.getcwd())
sys.path.insert(0, projdir)
#sys.path.extend([
#    os.path.abspath('/home/ccellerini/adtxns/modular_model/transactions/methods'),
#    os.path.abspath('/home/ccellerini/adtxns/modular_model/transactions'),
#    os.path.abspath('/home/ccellerini/adtxns/modular_model'),
#])

import src.model as model
import src.dists as dists

# Convert metadata to JSON-serializable format
def convert_metadata_to_serializable(metadata):
    return {
        key: (int(value) if isinstance(value, np.integer) else
              float(value) if isinstance(value, np.floating) else
              value)
        for key, value in metadata.items()
    }

# Generate parameter grid
def create_parameter_grid(params,seed=None):
    """
    Generate a grid of parameters for batch simulations.
    """
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
     # Generate a list of additional seeds
    rng = np.random.default_rng(seed=seed)
    seeds = rng.integers(low=0, high=2**32 - 1, size=len(combinations))
    # Add the seeds to the parameter combinations
    for i, combination in enumerate(combinations):
        combination['seed'] = seeds[i]
    return combinations

# Generate activity and attractivity samples
def generate_activity_attractivity(N, copula_type, copula_param, reversed_copula, activity_generator, attractivity_generator):
    unif_act, unif_att = dists.paired_samples(
        N,
        params={
            'copula': copula_type,
            'theta': copula_param,
            'reversed': reversed_copula
        }
    )
    vect_act = activity_generator(unif_act)
    vect_att = attractivity_generator(unif_att)
    return vect_act, vect_att


def run_simulation(params, saved=None, seed=None):
    """
    Run a single simulation.

    Parameters
    ----------
    params : dict
        Dictionary containing all necessary parameters for the simulation.
    saved : int, optional
        Number of transactions to save after the burn-in period. Defaults to the total number of transactions.

    Returns
    -------
    transactions_df : pd.DataFrame
        DataFrame containing the saved transactions.
    metadata : dict
        Dictionary containing metadata about the simulation.
    execution_time : float
        Total execution time of the simulation.
    nodes_df : pd.DataFrame
        DataFrame containing node parameters.
    """
    start_time = time.time()

    # Extract parameters
    seed = params["seed"]
    s = params["s"]
    decimals = params["decimals"]
    N = params["N"]
    T = params["T"]
    D = params["D"]
    SIZE_SCALE = params["SIZE_SCALE"]
    LENGTH_SCALE = params["LENGTH_SCALE"]
    MEAN_IET = params["MEAN_IET"]
    burstiness = params["burstiness"]

    # Set random seed for reproducibility
    # TODO: Figure out how to get the distributions to use the seed
    #    np.random.seed(seed)
    #    dists.set_seed(seed) # Copilot halucinated this but maybe it is needed
    rng = np.random.default_rng(seed)

    # Extract distributions and generators
    sprate_type, sprate_params, sprate_generator = params["spending_rate"]
    inbal_type, inbal_params, inbal_generator = params["initial_balance"]
    copula_type, copula_param, reversed_copula = params["copula"]
    activity_type, activity_params, activity_generator = params["activity_distribution"]
    attractivity_type, attractivity_params, attractivity_generator = params["attractivity_distribution"]

    # Generate activity and attractivity vectors
    vect_act, vect_att = generate_activity_attractivity(
        N=N,
        copula_type=copula_type,
        copula_param=copula_param,
        reversed_copula=reversed_copula,
        activity_generator=activity_generator,
        attractivity_generator=attractivity_generator
    )

    # Generate spending rates and initial balances
    spending_rate = sprate_generator(N)
    initial_bal = inbal_generator(N)

    # Initialize the model
    # todo: transition matrix return by its own method
    nodes = model.create_nodes(
        N, activity=vect_act, attractivity=vect_att,
        spending=spending_rate, mean_iet=MEAN_IET, burstiness=burstiness
    )
    transitions = model.initialize_transition_matrix(nodes)
    activations = model.initialize_activations(nodes)
    balances = model.initialize_balances(nodes, balances=initial_bal, decimals=decimals)

    # Prepare to write transactions
    header = ["timestamp", "source", "target", "amount", "source_bal", "target_bal"]

    #todo: move saved to params
    if saved is None:
        saved = params['T']
    transactions_list = [None] * saved
    burn_in_period = T - saved

    # Run the model
    for i in range(T):
        transaction = model.transact(nodes, activations, transitions, balances, method='random_share', s=s, rng=rng)
        if i >= burn_in_period:
            transactions_list[i-burn_in_period] = [transaction[term] for term in header]

    # Convert transactions to DataFrame
    transactions_df = pd.DataFrame(transactions_list, columns=header)
    nodes_df = pd.DataFrame.from_dict(nodes, orient='index').reset_index()

    # Compile metadata
    metadata = {
        'seed': seed,
        'sprate_type': sprate_type,
        'sprate_params': sprate_params,
        'inbal_type': inbal_type,
        'inbal_params': inbal_params,
        'activity_type': activity_type,
        'activity_params': activity_params,
        'attractivity_type': attractivity_type,
        'attractivity_params': attractivity_params,
        'N': N,
        'T': T,
        'D': D,
        's': s,
        'decimals': decimals,
        'SIZE_SCALE': SIZE_SCALE,
        'LENGTH_SCALE': LENGTH_SCALE,
        'MEAN_IET': MEAN_IET,
        'burstiness': burstiness,
        'copula_type': copula_type,
        'copula_param': copula_param
    }

    execution_time = time.time() - start_time
    return transactions_df, metadata, execution_time, nodes_df

# Print memory usage
def print_memory_usage(label):
    """
    Logs memory usage for debugging.

    Parameters
    ----------
    label : str
        Label to prepend to the memory usage message.

    Returns
    -------
    memory_info.rss : int
        Memory usage in bytes (Resident Set Size).

    Notes
    -----
    This function is useful for tracking memory usage in scripts, e.g. during long simulations.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"{label}: {memory_info.rss / (1024 ** 2):.2f} MB (Resident Set Size)")
    return memory_info.rss

# Clear memory
def clear_memory():
    """
    Triggers garbage collection to free up unused memory and prints a confirmation message.
    """

    gc.collect()
    print("Memory cleared.")

# Run batch simulations
def batch_runner(params_grid, output_dir, saved=None):
    """
    Executes batch simulations using the provided parameter grid.

    Parameters
    ----------
    params_grid : list
        List of parameter sets for each simulation.
    saved : int, optional
        Number of transactions to save, defaults to None.
    output_dir : str, optional
        Directory for saving simulation outputs, defaults to a warning directory.

    Notes
    -----
    This function runs simulations in a batch, saves the transaction and node data,
    and logs metadata and memory usage.
    """

    # Start the timer for the entire loop
    loop_start = time.time()

    # Initialize empty list to track memory usage
    memory = []

    # Define file path for metadata summary
    metadata_file = os.path.join(output_dir, "metadata_summary.json")

    # Initialize empty list to track metadata summary
    metadata_summary = []

    # Iterate over each set of parameters
    for idx, params in enumerate(tqdm(params_grid, desc="Batch Simulations")):
        try:
            # If the number of transactions to save was not specified,
            # use the value from the current parameter set
            if saved is None:
                saved = params['T']

            # Generate unique identifier for simulation
            simulation_id = f'{idx}'
            time_id = f'{int(time.time())}'

            # Run the simulation
            transactions_df, metadata, execution_time, nodes_df = run_simulation(params, saved=saved)

            # Add simulation ID to metadata
            metadata["simulation_id"] = simulation_id

            # Save transactions
            transaction_file = os.path.join(output_dir, f"{simulation_id}.parquet")
            transactions_df.to_parquet(transaction_file, index=False)

            nodes_file = os.path.join(output_dir, f'nodes_{simulation_id}.csv')
            nodes_df.to_csv(nodes_file, index=False)

            # Save metadata for this simulation
            metadata_file_path = os.path.join(output_dir, f"{simulation_id}_metadata.json")
            with open(metadata_file_path, 'w') as f:
                json.dump(convert_metadata_to_serializable(metadata), f, indent=4)

            # Track metadata summary
            metadata_summary.append({
                "simulation_id": simulation_id,
                "transaction_file": transaction_file,
                "metadata_file": metadata_file_path,
                "time id": time_id,
                "execution_time": execution_time,
            })

            print(f"Completed simulation: {params}")
            print(f"Execution time: {execution_time:.2f} seconds")
            memory.append(print_memory_usage(f"End of {simulation_id}"))

            # Clear memory after each loop
            clear_memory()

        except Exception as e:
            print(f"Error in simulation {idx}: {e}")
            continue

    # Save metadata summary
    with open(metadata_file, 'w') as f:
        json.dump(metadata_summary, f, indent=4)

    # Plot memory usage
    plt.figure()
    plt.plot(memory, marker='o', label='Memory Usage (MB)')
    plt.xlabel('Simulation Index')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Simulations')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(os.path.join(output_dir, "memory_usage_plot.png"))

    print("All simulations completed.")
    print(f"Total execution time: {time.time() - loop_start:.2f} seconds")

