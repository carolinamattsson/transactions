import os
import sys
import numpy as np
import argparse

# Allow `python src/simulations.py` from the repo root by putting the repo root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import src.execution as execution
import src.dists as dists

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a batch of transaction simulations.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (default: False)")
    parser.add_argument("--output-dir", type=str, default="output/", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master seed; each combination gets a derived (seed_setup, seed_run) pair.")
    parser.add_argument("--seed-setup", type=int, default=None,
                        help="Pin the setup seed (population draw) across every combination.")
    parser.add_argument("--seed-run", type=int, default=None,
                        help="Pin the run seed (transaction realization) across every combination.")

    # Parse arguments
    args = parser.parse_args()
    test = args.test
    output_dir = args.output_dir
    seed = args.seed
    seed_setup = args.seed_setup
    seed_run = args.seed_run

    # Constants as parameters. MEAN_IET is set to one Sarafu time unit so the
    # default grid is comparable to the paper; override for other datasets.
    SIZE_SCALE = 1
    LENGTH_SCALE = 6
    MEAN_IET = 43706315  # 1 Sarafu time unit in seconds
    N = 25_000 // SIZE_SCALE
    T = int(500_000 * LENGTH_SCALE)
    saved = 500_000

    if test:
        output_dir = output_dir + "test/"
        N = 500
        T = 50
        saved = 20

    print(f"Iteration : {T:_}")
    print(f"Save : {saved:_}")
    print(f"Master seed: {seed} | seed_setup pin: {seed_setup} | seed_run pin: {seed_run}")

    # Generators take (N, rng) so each call uses the per-simulation setup_rng.
    spending_rate_list = [
        ("uniform", [0, 1], lambda N, rng: rng.uniform(1e-16, 1, N)), #*('name',[parameters], actual distribution as lambda of (N, rng))
        # ("beta", [0.4,0.6], lambda N, rng: rng.beta(0.4,0.6,N))
    ]

    # Define initial balances
    initial_bal_list = [
        # ("constant", [100], lambda N, rng: 100 * np.ones(N)),
        # ("uniform", [0, 1], lambda N, rng: rng.uniform(1e-16, 1, N)),
        # ("pareto", [0.9], lambda N, rng: 15*rng.pareto(0.9,N)),
        ("constant", [1000], lambda N, rng: 1000 * np.ones(N)),
        # ("lognormal",[200,1], lambda N, rng: 200*rng.lognormal(1,size=N)),
        # ("uniform", [0,2000], lambda N, rng: rng.uniform(1e-16,2000,N)),
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

    # Every value must be a list — create_parameter_grid takes the product over them.
    parameter_dict = {
        "xi": [1, 2, 3, 5, 10],
        "spending_rate": spending_rate_list,
        "initial_balance": initial_bal_list,
        "decimals": [3],
        "copula": copulas,
        "activity_distribution": activity_distributions,
        "attractivity_distribution": attractivity_distributions,
        "N": [N],
        "T": [T],
        "SIZE_SCALE": [SIZE_SCALE],
        "LENGTH_SCALE": [LENGTH_SCALE],
        "MEAN_IET": [MEAN_IET],
        "burstiness": [0.5, 0.75, 1, 1.5, 3],
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate parameter grid
    parameter_grid = execution.create_parameter_grid(
        parameter_dict, seed=seed, seed_setup=seed_setup, seed_run=seed_run
    )

    # Run batch simulations
    execution.batch_runner(parameter_grid,output_dir,saved=saved)
