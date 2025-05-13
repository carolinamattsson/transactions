import methods.execution as execution
import methods.dists as dists
import methods.model as model
import os 
import sys
import numpy as np

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
        output_dir = "files/new_version/000/test"
    else:
        output_dir = "files/new_version/000"
        print(output_dir)
        input('continue?')
    # output_dir = 'simulations/adtxns/test/00_abc'
    os.makedirs(output_dir, exist_ok=True)


    # Generate parameter grid
    parameter_grid = execution.create_parameter_grid(parameter_dict)

    # Run batch simulations
    execution.batch_runner(parameter_grid,saved,output_dir)
