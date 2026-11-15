# Examples

Every example on these pages was run for the documentation with lcinv 0.2.0, and the outputs
shown are the real ones, copied from the terminal or the results folder. Most use short sampler
settings so that they finish in minutes on a laptop CPU; the production runs of the same data
are on [Reference results](reference-results.md).

| Example | Shows | Data | Runtime (2 CPU cores) |
|---|---|---|---|
| [Quickstart](../getting-started/quickstart.md) | a complete global inversion, stopping early and resuming | K03Q21N | about 20 min |
| [ADES survey photometry](ades-566130.md) | MPC ADES input, 30 s exposures, a fast rotator and the period-range trap | 566130 | about 15 min |
| [Period search from Python](period-search-api.md) | `prepare()` and the multiband Fourier periodogram | K03Q21N | seconds |
| [Likelihood and shape from Python](convex-api.md) | `ConvexInversion`, a pole map and a reconstructed shape | K03Q21N | under a minute |
| [Hybrid model](hybrid-k03q21n.md) | joint spin and shape sampling in each spin state | K03Q21N | about 20 min |
| [Working with finished runs](post-processing.md) | convergence reports, OBJ export, viewers, model comparison | both | seconds |
| [Batch processing](batch.md) | a target table and a `--dry-run` plan | both | seconds |
| [Contact binary versus ellipsoid](synthetic-binary.md) | what a two-lobe body does to a lightcurve | synthetic | 30 s |
| [Reference results](reference-results.md) | full-length runs to compare against | both | (shipped) |

## The bundled data

| | K03Q21N (2003 QN21) | 566130 |
|---|---|---|
| file | `data/K03Q21N.lc` (plain lightcurve with geometry) | `data/566130_ades.csv` (MPC ADES) |
| observer | Rubin Observatory (X05) | Rubin Observatory (X05); the file also holds 10 other stations, 2007–2026 |
| points used | 334 in g, r, i | 195 in Lg, Lr, Li (`--stations X05`) |
| span | five nights, 2026 Feb 23–27 | five nights, 2025 Apr 28 – May 5 |
| rotation period | 3.248 h | 0.178 h (10.7 min) |
| feature | large amplitude (0.40 mag), equator-on | 30 s exposures (`--exptime 30`; the file has no `exp` values) are 5 % of the period |

The Horizons geometry of both is cached in `data/`, so no example needs the network.

## Running the examples yourself

Run every command from the repository folder. Scripts in `examples/docs/` import the package,
so put the repository folder on the path:

```bash
PYTHONPATH=. python examples/docs/period_search.py
```

Without a GPU, add `--device cpu` to the model scripts (all commands here do). Outputs differ
from run to run in the last digits: the nested sampler is stochastic, and the shortened runs
have larger errors than the reference runs.
