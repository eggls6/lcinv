# `compare_models`

*Evidence-based comparison of runs on the same data.*

```text
python compare_models.py [-h] [--out OUT] [--no-recompute]
                         folders [folders ...]
```

```text
Honest comparison of all models fitted to one target, with a recommendation.

  python compare_models.py results/566130_X05_cl results/566130_X05_binary results/566130_X05_hybrid
  python compare_models.py results/566130_X05*  --out results/566130_comparison

Every folder may be a convex run (run_inversion.py), a contact-binary run (run_binary.py) or
a hybrid run (run_hybrid.py); a complexity_scan.json inside a convex folder adds its rows.

For each model the table gives
  - goodness of fit of the best fit: N, effective number of parameters p_eff, reduced chi^2
    with the quoted errors and with the fitted jitter, residual rms (lcinv.fitstats);
  - the evidence ln Z, its difference to the best model and the Jeffreys / Kass-Raftery
    reading, and the posterior probability of each model for equal prior odds;
  - convergence, the evidence method (nested sampling, importance sampling, Laplace) and
    the required cohesion (Sanchez & Scheeres 2014) where a diameter was given.

Only runs on identical data (same data fingerprint: epochs, magnitudes, errors, bands) are
compared; others are listed but excluded. Evidences from importance sampling with low ESS are
lower bounds and are not used for probabilities, but they enter the recommendation. Laplace
evidences (hybrid) are approximate: an uncertainty of 2.0 is assumed (two runs of the same hybrid
model in different pole parametrisations differ by ~3).

Outputs: <out>.md (table + recommendation), <out>.csv, <out>.json (default <first folder>/model_comparison).
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `folders` | – | results folders (convex, binary or hybrid runs) of the same target |

## Options

| Option | Default | Description |
|---|---|---|
| `--out`&nbsp;<code>OUT</code> | – | output path without extension |
| `--no-recompute` | off | do not rebuild models for old runs without stored fit statistics |

<small>3 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
