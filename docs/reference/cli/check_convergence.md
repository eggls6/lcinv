# `check_convergence`

*Convergence report for finished runs, agreement between repeats.*

This command reads its arguments directly (no option parser).

```text
Convergence report for finished runs, and agreement between repeat runs.

    python check_convergence.py results/K03Q21N                    # one global run
    python check_convergence.py results/X_seed1 results/X_seed2    # + reproducibility
    python check_convergence.py results/X_hybrid                   # hybrid (NUTS) run

Global runs are read from nested_results.pkl + summary.json; hybrid runs from
posterior_spin_state_*.npz. Repeat runs agree when ln Z differs by less than ~2 combined sigma and the
period, jitter and prograde fraction overlap within their errors.
```
