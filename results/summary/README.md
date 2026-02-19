# Summary (plots & metrics)

Ο φάκελος `results/summary/` είναι ένα ενιαίο σημείο αναφοράς για **plots** και **metrics** από **όλα τα παιχνίδια** και από **όλα τα experiments** (π.χ. `fp_vs_fp`, `rl_vs_fp`, `rl_vs_rl`).

## Πώς δημιουργείται

Τρέξε:

```bash
python experiments/build_summary.py
```

Παράγει:
- `results/summary/INDEX.md` (λίστα με links προς κάθε run)
- `results/summary/summary.csv` (συγκεντρωτικός πίνακας)
- ανά run: `metrics.json`, `empirical_strategies.csv`, `action_probabilities.csv`, `joint_action_probabilities.csv`, `timeseries.csv` και τα plots.

## Labels (actions)

Για τους τίτλους/legends στα plots και για τους πίνακες, τα actions “μεταφράζονται” σε ετικέτες:

| Game key | 0 | 1 | 2 |
|---|---|---|---|
| `matching_pennies` | Heads | Tails | — |
| `anti_coordination` | Strategy A | Strategy B | — |
| `almost_rock_paper_scissors` | Rock | Paper | Scissors |
| `prisoners_dilemma` | Cooperate | Defect | — |
| `switching_dominance` | A0 | A1 | — |
| `terrain_sensor` | (numeric) | (numeric) | (numeric) |

## Definitions (metrics)

Οι ορισμοί εδώ κρατούν το **νόημα** (όχι copy-paste) και χρησιμοποιούνται από το script που φτιάχνει το summary.

- **Empirical strategy**: η κατανομή συχνοτήτων των actions ενός παίκτη σε όλο το run (probabilities που αθροίζουν σε 1).
- **Entropy (H)**: μέτρο “απροσδιοριστίας” της empirical strategy. Υπολογίζεται ως `H(p) = -Σ p_i log(p_i)` (με φυσικό λογάριθμο).
- **Exploitability (zero-sum)**: πόσο “εκμεταλλεύσιμη” είναι η εμπειρική στρατηγική σε παιχνίδι μηδενικού αθροίσματος, ως **minimax gap**: καλύτερη μονομερής απόκριση του P1 απέναντι στο `y` μείον το χειρότερο που μπορεί να του κάνει ο P2 απέναντι στο `x`.
- **Exploitability (general-sum)**: άθροισμα των δύο μονομερών “κερδών από απόκλιση”:  
  `exploit_total = (BR1 - u1) + (BR2 - u2)`, όπου `u1, u2` είναι οι αναμενόμενες απολαβές των δύο παικτών κάτω από τις empirical strategies.
- **Rolling exploitability curve**: exploitability ανά γύρο, υπολογισμένη πάνω σε rolling empirical strategies (παράθυρο `W`, με μέγεθος `min(W, t)` στους πρώτους γύρους).
- **t_exploit_lt_0_10 / t_exploit_lt_0_05**: ο **πρώτος** γύρος `t` όπου το rolling exploitability πέφτει κάτω από `0.10` / `0.05` (όπου αυτό είναι διαθέσιμο).

## Plots

Αυτά είναι τα plots που παράγονται στο summary:

- `plot_payoff_running_mean.png`: running mean payoff ανά παίκτη (P1/P2).
- `plot_action_proportions.png`: rolling proportions των actions και για τους 2 παίκτες (με window `W`).
- `plot_terrain_movement.gif`: μόνο για `TerrainGame` — **ένα** animated heatmap που δείχνει “κίνηση” και των 2 παικτών μαζί (rolling window `W`). Κόκκινο=P2 (path density), πράσινο=P1 (sensor density), κίτρινο=επικάλυψη.
- `plot_exploitability.png`: rolling exploitability (μόνο όταν ορίζεται/υπολογίζεται).
- `plot_regret.png`: regret ανά χρόνο (P1 & P2 στο ίδιο figure), όταν υπάρχει στο `results.csv`.
- `plot_joint_action_heatmap.png`: heatmap των τελικών πιθανοτήτων των κινήσεων (joint empirical distribution πάνω στα (a,b)).
