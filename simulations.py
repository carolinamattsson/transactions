import methods.execution as execution
import methods.dists as dists
import methods.model as model
import os 
import sys
import numpy as np
import argparse

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a batch of transaction simulations.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (default: False)")
    parser.add_argument("--output_dir", type=str, default="output/", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=None, help="Specify a random seed")

    # Parse arguments
    args = parser.parse_args()
    test = args.test
    output_dir = args.output_dir
    seed = args.seed #  from the given one

    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)
    # If you want the seed to work, probably you need to create these generators inside run_simulation
    # My suggestion is to pass this original seed into the parameter grid and make one per simulation

    # Constants as parameters
    SIZE_SCALE = 1
    LENGTH_SCALE = 6
    MEAN_IET = 43706315  # 1 Sarafu time unit in seconds
    N = 25_000 // SIZE_SCALE
    T = int(500_000 * LENGTH_SCALE)
    # T = 2
    D = T / MEAN_IET #useless
    saved = 500_000

    if test:
        output_dir = output_dir + "test/"
        N = 500
        T = 50
        saved = 20

    print(f"Iteration : {T:_}")
    print(f"Save : {saved:_}")
    print(f"Random seed: {seed}")

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
        # "seed": seed,
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
        "burstiness": [0.5, 0.75, 1, 1.5, 3],
    }

    if not test:
        print(output_dir)
        input('Continue?')  # Pause for confirmation in full run mode

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate parameter grid
    parameter_grid = execution.create_parameter_grid(parameter_dict,seed=seed)

    # Run batch simulations
    execution.batch_runner(parameter_grid,output_dir,saved=saved)
