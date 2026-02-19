# MSCAI Agents Project

Game-playing agents for classic matrix games (Fictitious Play) plus reinforcement-learning (RL) baselines for stochastic games (including `TerrainGame` and `StochasticSwitchingDominanceGame`).

## What's in the repo

- **Fictitious Play (FP)**: best-response dynamics with optional mixed-strategy action selection.
- **RL baselines**:
  - Independent Q-learning (`IndependentQLearner`)
  - Minimax-Q for 2-action zero-sum games (`MinimaxQLearner`)
- **Games**:
  - Matrix games: Matching Pennies, Prisoner's Dilemma, Anti-Coordination, Almost RPS
  - Stochastic switching-dominance game: `StochasticSwitchingDominanceGame`
  - Stochastic terrain sensor game: `TerrainGame`

## Requirements

- Python **3.10+** (some modules use `X | None` type syntax)
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```

## Running experiments

All experiment entrypoints are under `experiments/` and save outputs under `results/`.

### FP vs FP

```bash
python experiments/main_fp_vs_fp.py
```

Outputs (per game run):
- `results/fp_vs_fp/<Game>_<YYYY-MM-DD_HH-MM-SS>/report.txt`
- `results/fp_vs_fp/<Game>_<YYYY-MM-DD_HH-MM-SS>/results.csv`

### RL vs FP

```bash
python experiments/main_rl_vs_fp.py --steps 20000 --seed 0 --switch_p 0.2 --alpha 0.2 --gamma 0.95 --eps 0.1 --fp_strategy pure
```

Outputs:
- `results/rl_vs_fp/<Game>_<YYYY-MM-DD_HH-MM-SS>/report.txt`
- `results/rl_vs_fp/<Game>_<YYYY-MM-DD_HH-MM-SS>/results.csv`
- `results/rl_vs_fp/<Game>_<YYYY-MM-DD_HH-MM-SS>/args.json`
- `results/rl_vs_fp/<Game>_<YYYY-MM-DD_HH-MM-SS>/data.npz`

### FP vs RL

```bash
python experiments/main_fp_vs_rl.py --steps 20000 --seed 0 --switch_p 0.2 --alpha 0.2 --gamma 0.95 --eps 0.1 --fp_strategy pure
```

Outputs:
- `results/fp_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/report.txt`
- `results/fp_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/results.csv`
- `results/fp_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/args.json`
- `results/fp_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/data.npz`

### RL vs RL

```bash
python experiments/main_rl_vs_rl.py --steps 20000 --seed 0 --switch_p 0.2 --alpha 0.2 --gamma 0.95 --eps 0.1
```

Outputs:
- `results/rl_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/report.txt`
- `results/rl_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/results.csv`
- `results/rl_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/args.json`
- `results/rl_vs_rl/<Game>_<YYYY-MM-DD_HH-MM-SS>/data.npz`

## Building a summary (plots + metrics)

To generate an aggregated folder with plots/metrics for every game across all experiments:

```bash
python experiments/build_summary.py
```

Outputs under:
- `results/summary/README.md`
- `results/summary/INDEX.md`
- `results/summary/summary.csv`

## Using agents in code

```python
import numpy as np
from agents.agent_fp import FictitousPlayAgent as FictitiousPlayAgent

payoff_matrix = np.array([[1, -1], [-1, 1]])
agent = FictitiousPlayAgent(
    payoff_matrix=payoff_matrix,
    action_space=2,
    opponent_action_space=2,
    strategy_type="mixed",  # or "pure"
)
```

## Project structure (current)

```
mscai-agents-project/
  agents/
    agent_fp.py
    agent_rl_q.py
    agent_rl_minimaxq.py
  experiments/
    main_fp_vs_fp.py
    main_fp_vs_rl.py
    main_rl_vs_fp.py
    main_rl_vs_rl.py
    build_summary.py
  games/
    matching_pennies.py
    prisoners_dilemma.py
    anti_coordination.py
    almost_rock_paper_scissors.py
    stochastic_switching_dominance.py
    terrain_sensor.py
  results/
    fp_vs_fp/
    fp_vs_rl/
    rl_vs_fp/
    rl_vs_rl/
    summary/
  requirements.txt
  README.md
```
