# HeteroCollude.jl

This project studies pricing agents in a repeated market. Three firms set prices again and again and each firm uses a different kind of learning method. The firms never communicates.

This project studies -
> Can learning agents end up with persistently high prices through repeated interaction, even without communication or central control?

## Why it matters

Pricing software is used in many markets. Some people worry that such software could keep prices high without any agreement between companies. This is also a topic in AI safety, where researchers study harmful behavior that can come from many autonomous agents acting together.

This project is a small-scale, controlled experiment on that topic. It is a simplified model, and tts results describe the algorithms and the market used here. 


## The model

- **Market:** repeated Bertrand competition with three firms and identical products.
- **Demand:** `D(p) = max(0, 100 - p)`. The lowest price wins the market. Ties share it equally.
- **Costs and prices:** marginal cost is 10. Prices go from 10 to 90 in steps of 5.
- **Benchmarks:** competitive price is 10. Monopoly price is 55.
- **Information:** each agent sees only its own action, profit and quantity, and the lowest rival price.

## The agents

| Agent | Idea |
|---|---|
| Q-learning | Learns a value table for prices, with random exploration |
| Evolutionary | Keeps a small group of pricing policies and mutates the best ones |
| Adaptive rule | Moves its price up or down based on its own recent profit |

The rule-based agent also has random noise and different parameters per agent. This reduces the chance that identical rules move prices together by accident.

## Project layout

```
src/            Model and agents
  environment.jl   Market settings, demand, benchmarks
  market.jl        Market clearing, profits, welfare
  agents.jl        Shared agent interface
  qlearning.jl     Q-learning agent
  evolutionary.jl  Evolutionary agent
  rulebased.jl     Adaptive rule agent
  simulation.jl    Populations, seeds, running simulations
  metrics.jl       Collusion index, persistence, statistics
  diagnostics.jl   Forced-undercut test
test/           Unit tests
experiments/    Scripts that run experiments and make figures
results/        Output files (CSV tables and figures)
```

### Experiment scripts

| Script | What it does |
|---|---|
| `experiments/smoke_run.jl` | Quick check of the four main populations |
| `experiments/homogeneous.jl` | Experiments 1 to 3: three agents of the same type |
| `experiments/heterogeneous.jl` | Experiment 4: one agent of each type, compared with the first three |
| `experiments/sensitivity.jl` | Experiment 5: all compositions and robustness checks |
| `experiments/visualize.jl` | Makes figures from the saved CSV files |
| `experiments/common.jl` | Shared helper code (not run directly) |

## How to run


**1. Set up**

```bash
cd HeteroCollude.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add(["DataFrames", "CSV", "Plots"])'
```

**2. Run the tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**3. Run the experiments**

```bash
julia --project=. experiments/homogeneous.jl
julia --project=. experiments/heterogeneous.jl
julia --project=. experiments/sensitivity.jl
```

Each script saves CSV files in `results/`. The full runs can take several minutes.

For a fast trial, use quick mode (4 replications, 2,000 periods):

```bash
HETERO_QUICK=1 julia --project=. experiments/homogeneous.jl
```

**4. Make the figures**

```bash
julia --project=. experiments/visualize.jl
```

Figures are saved in `results/figures/` as PNG and SVG. Run the experiment scripts first, because the figures read their CSV files.

## Output files

| File | Content |
|---|---|
| `*_runs.csv` | One row per replication, with all metrics |
| `*_summary.csv` | Mean, spread and 95% interval across replications |
| `*_trajectories.csv` | Average price and welfare over time, for plots |
| `exp4_comparisons.csv` | Tests of the mixed population against the controls |

## How to reproduce results

- **Default settings:** 10,000 periods per run and 30 replications per configuration.
- **Seeds:** the base seed is 2026. It creates 30 replication seeds, and the same list is used for every population. A run depends only on its seed, so the same code gives the same output.
- **Same setup:** use the same Julia and package versions. The random number generator can differ between Julia versions.
- **Steps:** run the three experiment scripts without quick mode, then `visualize.jl`.
- **Settings:** change `DEFAULT_PERIODS` and `DEFAULT_REPLICATIONS` in `src/environment.jl`, or pass values to `SimulationConfig`.

## Limits

- The market is small and simple: three symmetric firms, a linear demand curve and a coarse price grid.
- The agents are basic learning methods with fixed settings.
- A high collusion index is only a price-based measure. It does not prove collusion.
- The tests that look for coordination have limited power.