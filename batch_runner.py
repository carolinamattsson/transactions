import os
import sys
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from scipy.stats import uniform
import psutil
import matplotlib.pyplot as plt
import gc

# Add directories to Python path
projdir = os.path.abspath(os.getcwd())
sys.path.insert(0, projdir)
sys.path.extend([
    
    os.path.abspath('/home/ccellerini/adtxns/modular_model/transactions/methods'),
    os.path.abspath('/home/ccellerini/adtxns/modular_model/transactions'),
    os.path.abspath('/home/ccellerini/adtxns/modular_model'),
])

import methods.model as model
import methods.dists as dists

np.random.seed(42)
rng = np.random.default_rng(seed=42)


# Convert metadata to JSON-serializable format
def convert_metadata_to_serializable(metadata):
    return {
        key: (int(value) if isinstance(value, np.integer) else
              float(value) if isinstance(value, np.floating) else
              value)
        for key, value in metadata.items()
    }

# Generate parameter grid
def create_parameter_grid(params):
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations

# Generate activity and attractivity samples
# def generate_activity_attractivity(
#     N, 
#     copula_type, 
#     copula_param, 
#     reversed_copula, 
#     activity_generator, 
#     attractivity_generator, 
#     rng=None  # ← NEW: allow external RNG for reproducibility
# ):
#     if rng is None:
#         rng = np.random.default_rng()  # ← NEW: fallback RNG if none provided

#     # ← CHANGED: now we pass rng explicitly to paired_samples
#     unif_act, unif_att = dists.paired_samples(
#         N,
#         params={
#             'copula': copula_type,
#             'theta': copula_param,
#             'reversed': reversed_copula
#         },
#         rng=rng  # ← NEW: ensures reproducibility
#     )

#     vect_act = activity_generator(unif_act)
#     vect_att = attractivity_generator(unif_att)
#     return vect_act, vect_att

def generate_activity_attractivity(N, copula_type, copula_param, reversed_copula, activity_generator, attractivity_generator):
    '''
    Sample uniform correlated values via paired_samples, then transform via PPFs.
    '''
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



def run_simulation(params, saved=None, rng=None):  # ← NEW: accepts external RNG
    start_time = time.time()

    if rng is None:
        rng = np.random.default_rng(seed=42)  # ← NEW: fallback seed for reproducibility

    # Extract simulation parameters
    s = params["s"]
    decimals = params["decimals"]
    N = params["N"]
    T = params["T"]
    D = params["D"]
    SIZE_SCALE = params["SIZE_SCALE"]
    LENGTH_SCALE = params["LENGTH_SCALE"]
    MEAN_IET = params["MEAN_IET"]
    burstiness = params["burstiness"]

    # Extract all random-dependent generators
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
        attractivity_generator=attractivity_generator,
        # rng=rng  # ← NEW: ensure reproducibility of copula sampling,#* for the future
    )

    # Generate spending rate and initial balance vectors
    spending_rate = sprate_generator(N, rng=rng)   # ← NEW: pass rng to lambda
    initial_bal = inbal_generator(N, rng=rng)      # ← NEW: pass rng to lambda

    # Initialize model
    nodes, transition_matrix = model.create_nodes(
        N,
        activity=vect_act,
        attractivity=vect_att,
        spending=spending_rate,
        mean_iet=MEAN_IET,
        burstiness=burstiness
    )
    acts = model.initialize_activations(nodes)
    balances = model.initialize_balances(nodes, balances=initial_bal, decimals=decimals)

    # Prepare for transaction recording
    if saved is None:
        saved = T
    burn_in_period = T - saved
    transactions_list = [None] * saved
    header = ["timestamp", "source", "target", "amount", "source_bal", "target_bal"]

    # Run transaction loop
    for i in range(T):
        transaction = model.transact(
            nodes, 
            acts, 
            transition_matrix, 
            balances,
            method='random_share',
            s=s
        )
        if i >= burn_in_period:
            index = i - burn_in_period  # ← NEW: fill forward, from 0 to saved-1
            transactions_list[index] = [transaction[term] for term in header]

    # Convert results to DataFrames
    transactions_df = pd.DataFrame(transactions_list, columns=header)
    nodes_df = pd.DataFrame.from_dict(nodes, orient='index')
    nodes_df = nodes_df.sort_index().reset_index()  # ← force deterministic row order


    # Build metadata dictionary
    metadata = {
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
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"{label}: {memory_info.rss / (1024 ** 2):.2f} MB (Resident Set Size)")
    return memory_info.rss

# Clear memory
def clear_memory():
    gc.collect()
    print("Memory cleared.")

# Run batch simulations
def batch_runner(params_grid,saved=None):
    
    loop_start = time.time()
    memory = []
    metadata_file = os.path.join(output_dir, "metadata_summary.json")
    metadata_summary = []

    for idx, params in enumerate(tqdm(params_grid, desc="Batch Simulations")):
        try:
            if saved is None:
                saved = params['T']
            # Unique identifier for simulation
            # simulation_id = f"sim_{idx}_{int(time.time())}"
            simulation_id = f'{idx}'
            time_id = f'{int(time.time())}'

            # Run simulation
            transactions_df, metadata, execution_time, nodes_df = run_simulation(params, saved=saved)

            # Add simulation ID to metadata
            metadata["simulation_id"] = simulation_id

            # Save transactions
            transaction_file = os.path.join(output_dir, f"{simulation_id}.parquet")
            transactions_df.to_parquet(transaction_file, index=False)

            nodes_file = os.path.join(output_dir,f'nodes_{simulation_id}.csv')
            nodes_df.to_csv(nodes_file,index=False)

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

from scipy.stats import pareto



# Main execution
# s low values + burstiness 


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    # Constants as parameters
    SIZE_SCALE = 1
    LENGTH_SCALE = 6
    MEAN_IET = 43706315  # 1 Sarafu time unit in seconds
    N = 25_000 // SIZE_SCALE
    T = int(500_000 * LENGTH_SCALE)
    # T = 2
    D = T / MEAN_IET #useless
    saved = 500_000

    # test = 'y'
    test = input('Is this a test ? (y/n)')

    if test == 'y':
        N = 500
        T = 50
        saved = 20
    print(f"Iteration : {T:_}")
    print(f"save : {saved:_}")
    # Define spending rates


# Define spending rates
    spending_rate_list = [
        ("uniform", [0, 1], lambda N: rng.uniform(1e-16, 1, N)), #*('name',[parameters], actual distribution as lambda of N)
        # ("beta", [0.4,0.6], lambda N: rng.beta(0.4,0.6,N))
    ]

    # Define initial balances
    initial_bal_list = [
        # ("constant", [100], lambda N: 100 * np.ones(N)),
        # ("uniform", [0, 1], lambda N: rng.uniform(1e-16, 1, N)),
        # ("pareto", [0.9], lambda N: 15*rng.pareto(0.9,N)),
        ("constant", [1000], lambda N: 1000 * np.ones(N)),
        # ("lognormal",[200,1], lambda N: 200*rng.lognormal(1,size=N)),
        # ("uniform", [0,2000], lambda N: rng.uniform(1e-16,2000,N)),         

    ]

    # Define activity and attractivity distributions
    activity_distributions = [
        ("powlaw", [1.85, 1, 1838], lambda unif: dists.powlaw_ppf(1.85, 1, 1838)(unif)),
        # ('uniform',[0,1], lambda unif: unif),
        # ('cost', [1], lambda N:np.ones(N)),

    ]


    attractivity_distributions = [
        ("powlaw", [1.87, 1, 2118], lambda unif: dists.powlaw_ppf(1.87, 1, 2118)(unif)),
        # ('uniform',[0,1], lambda unif: unif),
        # ('cost', [1], lambda N:np.ones(N)),

    ]
     #type, param, reverse
    copulas = [
        ('joe', 3.15, False),
        # ('joe', 1, False),
    ]



    parameter_dict = {
        # "s": [2,5000],#np.logspace(0, 3, 6).astype(int),#[1],
        # "s": [2*N],
        # "s": [0.1,2,5,20,100,2000,10000],
        # "s": [1,N,5*N],
        # "s": [0.2,2,20,200,2000,N//2,N,5*N],
        # "s": [1,250,5000,N,N*1000,N*100_000],
        # "s": [1,5,10,20,50,100,250],
        "s":[1,2,3,5,10], #! always use lists
        # "spending_rate": [dist[2](N) for dist in spending_rate_list],
        # "spending_rate_name": [dist[0] for dist in spending_rate_list],
        "spending_rate": spending_rate_list,
        # "initial_bal": [dist[2] for dist in initial_bal_list],
        # "initial_bal_name": [dist[0] for dist in initial_bal_list],
        "initial_balance": initial_bal_list,
        "decimals": [3],
        # "decimals": [1,2,3,4,5],
        "copula": copulas, 
        "activity_distribution": activity_distributions,
        "attractivity_distribution": attractivity_distributions,
        "N": [N],
        "T": [T],
        "D": [D],
        "SIZE_SCALE": [SIZE_SCALE],
        "LENGTH_SCALE": [LENGTH_SCALE],
        "MEAN_IET": [MEAN_IET],
        # "burstiness": np.logspace(-1, 2, 6),
        # "burstiness": [1],
        # "burstiness": [0.05,0.5,1,5],
        "burstiness": [0.25, 0.5, 0.75, 1, 5],
    }
     # continue_running = input('Did You create the output dir? (y/n)')
    continue_running='y'
    if continue_running == 'n':
        print('Please create the output dir')
        # output_dir = input('Please enter the output dir:')
        exit()
    # Ensure output directory exists
    if test == 'y':
        output_dir = "/home/ccellerini/adtxns/files/final/009/test"
    else:
        output_dir = "/home/ccellerini/adtxns/files/final/009"
        print(output_dir)
        input('continue?')
    # output_dir = 'simulations/adtxns/test/00_abc'
    os.makedirs(output_dir, exist_ok=True)


    # Generate parameter grid
    parameter_grid = create_parameter_grid(parameter_dict)

    # Run batch simulations
    batch_runner(parameter_grid,saved)
