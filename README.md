# transactions

A mechanistic, stochastic, generative model of financial transactions as random walks on activity-driven temporal networks. Companion code for:

> Mattsson, C. E., Cellerini, C., Ojer, J., & Starnini, M. (2026). *Modeling financial transactions via random walks on temporal networks*. arXiv:2602.20713 [physics.soc-ph]. https://doi.org/10.48550/arXiv.2602.20713

The simplest version of the model: a group of $N$ nodes activate as a memoryless point process in continuous time, each sends a transaction to a random other node, and funds it with a sampled share of its present account balance. Enforcing fund conservation gives stationary distributions for balances and transaction sizes. When calibrated against real, public, transaction data the model largely reproduces the relevant distributions and matches the expected pattern of strong inflow/outflow correlation.

## Model structure

The model is three modules strung together in `transact`:

- `activate` — when does a node fire next? (memoryless or Weibull-burstiness inter-event times)
- `select` — given a source, who does it pay? (attractivity-weighted via a precomputed `TargetSampler`)
- `pay` — how much? There are several options, both for discrete and for continous balances.

Activations live in a min-heap keyed by timestamp so transactions are simulated in time order.

The `pay` module covers both **continuous balances** (floats) and **discrete balances** (`Decimal`, so payments come out as integer multiples of a chosen precision — e.g. cents). Three transaction-size rules are available:

- **Fraction** (continuous, no `xi`): the transaction is $p \cdot \mathrm{balance}$ — a deterministic share equal to the source's spending propensity $p$ (the paper's per-node $s$).
- **Fixed-probability sample** (discrete, no `xi`): $\mathrm{Binomial}(\mathrm{balance}, p)$ — each integer unit of the source balance jumps with probability $p$.
- **Variable-probability sample** (with precision `xi`): the jump probability $q$ is itself drawn from $\mathrm{Beta}(p \cdot \xi,\ (1-p) \cdot \xi)$, realized as $q \cdot \mathrm{balance}$ in the continuous case and as $\mathrm{Beta\text{-}Binomial}(\mathrm{balance}, p \cdot \xi, (1-p) \cdot \xi)$ in the discrete case. This is the overdispersed pay rule introduced in the paper.

There are a few knobs that aren't visible in the simplest setup. **Self-loops** can be admitted or excluded from the sampling support (`sample_self`), and if admitted, can be either emitted as transactions or treated as activations that just advance the clock (`record_self`). **Storage** of the per-source target distribution scales: at or below `matrix_n_threshold` (default 20,000), targets are drawn via searchsorted on a precomputed $(N, N)$ cumulative-row matrix; above it, the sampler falls back to a shared $(N,)$ cumulative attractivity vector with rejection. **Zero-amount transactions** (which can arise from legitimate BetaBin/Binomial draws or from float64 balance underflow on hyperactive nodes) can be emitted or skipped (`record_zero`).

**Input construction uses draws**, not hardcoded values: per-node spending propensity $h(s)$, the joint activity/attractivity distribution (optionally via a copula), and initial balances each come from generators. The simulator takes **two independent seeds** — `seed_setup` (the population draw: $h(s)$, activity, attractivity, initial balances) and `seed_run` (the realization: first activations and the `transact()` loop). Fixing setup and varying run gives multiple realizations over the same nodes; fixing run and varying setup studies the same realization-RNG across different populations.

## Install

The project ships a conda environment file. Using [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main):

```
conda env create -f environment.yml
conda activate txns
```

## Quickstart

[`example.ipynb`](example.ipynb) runs the model in a loop until a time limit is reached and stores the resulting transactions in a DataFrame.

## Batch mode

[`src/simulations.py`](src/simulations.py) runs a parameter grid from the command line. Run it from the repo root:

```
python src/simulations.py --help
python src/simulations.py --test                  # small, fast sanity run
python src/simulations.py --output-dir output/    # full run
```

[`tutorial.ipynb`](tutorial.ipynb) walks through configuring the parameter grid and reading the outputs.

## Repository layout

```
src/                 model + batch runner
  model.py             activate / select / pay / transact
  dists.py             distributions (power law, copula, etc.)
  execution.py         single-run + batch driver
  simulations.py       CLI entry point
numerical/           numerical solutions for P(m), P(w)
berka/               empirical analysis on the Berka dataset
sarafu/              pointers for analysis on the Sarafu dataset (not runnable)
example.ipynb        minimal usage example
tutorial.ipynb       batch-mode walkthrough
environment.yml      conda environment
```

## How to cite

If you use this software, please cite both the software release and the preprint. [`CITATION.cff`](CITATION.cff) contains both. The software DOI is minted per-release via Zenodo from the GitHub tag.

## License

[MIT](LICENSE).
