# MSCAI Agents Project

## Abstract

This project implements game-playing agents using game-theoretic learning algorithms. The Fictitious Play agent learns to play strategic games by maintaining and updating beliefs about opponent behavior.

**Fictitious Play** is a learning dynamics where an agent:
1. **Initializes beliefs** about the opponent's strategy distribution
2. **Plays a best response** to the assessed opponent strategy
3. **Observes the opponent's actual play** and updates beliefs accordingly
4. **Repeats steps 2-3** to adapt and converge toward equilibrium

This approach is particularly effective in many two-player games and serves as a foundation for understanding more sophisticated multi-agent learning algorithms.

---

## Key Features

- **Mixed Strategy Support**: Agents play probabilistic mixed strategies using softmax temperature-scaled action selection
- **Multiple Game Types**: Includes Prisoner's Dilemma, Matching Pennies, Anti-Coordination, and Almost Rock-Paper-Scissors
- **Comprehensive Metrics**: Tracks and visualizes:
  - Cumulative and average payoffs
  - Action probability distributions
  - Belief convergence
  - Regret over game runs
  - Nash equilibria identification
  - Outcome statistics

---

## Installation

### Prerequisites
- Python 3.7 or higher

### Setup Instructions

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd mscai-agents-project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running Complete Analysis

Execute the main analysis script to generate reports, plots, and GIFs:
```bash
python scripts/play_fp_vs_fp.py
```

This generates for each game:
- **PNG plot** with 4 subplots:
  - Regret evolution
  - Nash equilibria & convergence
  - Agent 1 action distribution convergence
  - Agent 2 action distribution convergence
- **Text report** with detailed statistics including:
  - Payoff matrices
  - Action probability distributions
  - Final beliefs
  - Complete payoff history

### Using the Agent in Your Code

```python
from agents.agent_fp import FictitousPlayAgent
import numpy as np

# Define payoff matrix
payoff_matrix = np.array([
    [1, -1],
    [-1, 1]
])

# Create agent with mixed strategy
agent = FictitousPlayAgent(
    payoff_matrix=payoff_matrix,
    action_space=2,
    opponent_action_space=2,
    strategy_type="mixed"  # or "pure" for best response only
)

# Play a round
opponent_action = 0
my_action = agent.play()
agent.observe(opponent_action)

# Get current metrics
print(f"Action: {my_action}")
print(f"Belief: {agent.get_belief()}")
print(f"Mixed Strategy: {agent.get_mixed_strategy()}")
```

---

## Project Structure

```
mscai-agents-project/
├── agents/
│   └── agent_fp.py              # Fictitious Play agent implementation
├── games/
│   ├── __init__.py
│   ├── prisoners_dilemma.py      # Prisoner's Dilemma game
│   ├── matching_pennies.py       # Matching Pennies game
│   ├── anti_coordination.py      # Anti-Coordination game
│   └── almost_rock_paper_scissors.py  # Almost RPS game
├── scripts/
│   └── play_fp_vs_fp.py          # Main analysis script (runs agents and generates reports)
├── results/                       # Output directory (auto-created)
│   └── *.txt                     # Text reports
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Games Included

### 1. Prisoner's Dilemma
Two players choose between Cooperate and Defect. Mutual cooperation yields good payoffs, but defection against a cooperator is tempting.

### 2. Matching Pennies
A zero-sum game where Player 1 wins if both play the same action, Player 2 wins if different.

### 3. Anti-Coordination
Players prefer to play different actions from each other.

### 4. Almost Rock-Paper-Scissors
Modified rock-paper-scissors with 3 actions and symmetric payoffs, creating mixed strategy equilibria.

---

## Output Files

### Reports (`.txt`)
Detailed game statistics including:
- Game configuration and payoff matrices
- Cumulative and average payoffs
- **Action probability distributions** for each agent
- Belief convergence metrics
- Complete action and payoff history

---

## Algorithm Details

### Fictitious Play with Mixed Strategy

1. **Belief Update**: Empirical distribution of opponent's observed actions
   - $b_t(a) = \frac{1 + \sum_{s=1}^{t-1} \mathbb{1}[a_s = a]}{n + t - 1}$

2. **Mixed Strategy Selection**: Softmax over expected payoffs
   - $\pi_t(a) = \frac{e^{u(a,b_t) / \tau}}{\sum_{a'} e^{u(a',b_t) / \tau}}$
   - Where $\tau$ is temperature parameter (controls exploration)

3. **Convergence**: Under certain conditions, empirical play converges to Nash equilibrium

---

## Configuration

### Temperature Parameter
The softmax temperature in `agent_fp.py` controls exploration vs exploitation:
- Lower values → closer to best response (greedy)
- Higher values → more uniform exploration

Default: `temperature = 0.1`

---

## Requirements

- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Static visualizations
- `Pillow` - Image processing for GIF creation
