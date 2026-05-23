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
python src/simulations.py --output_dir output/    # full run
```

[`tutorial.ipynb`](tutorial.ipynb) walks through configuring the parameter grid and reading the outputs.

## Repository layout

```
src/                 model + batch runner
  model.py             activate / select / pay / transact
  dists.py             distributions (power law, copula, etc.)
  execution.py         single-run + batch driver
  simulations.py       CLI entry point
  utils.py
numerical/           numerical solutions for P(m), P(w)
berka/               empirical analysis on the Berka dataset
sarafu/              empirical analysis on the Sarafu dataset
example.ipynb        minimal usage example
tutorial.ipynb       batch-mode walkthrough
environment.yml      conda environment
```

## How to cite

If you use this software, please cite both the software release and the preprint. [`CITATION.cff`](CITATION.cff) contains both. The software DOI is minted per-release via Zenodo from the GitHub tag.

## License

[MIT](LICENSE).
