# `complexity_scan`

*Evidence versus shape resolution (lmax) and prior scale (tau0).*

```text
python complexity_scan.py [-h] [--lmax LMAX [LMAX ...]]
                          [--tau0 TAU0 [TAU0 ...]] [--n N] [--seed SEED]
                          [--starts STARTS] [--map-iter MAP_ITER]
                          [--n-inner N_INNER] [--full] [--print-commands]
                          outdir
```

```text
Data-driven model complexity: how much shape detail do the data support?

  python complexity_scan.py results/566130_X05_cl --lmax 2 3 4 5 6           # convex run
  python complexity_scan.py results/566130_X05_cl --lmax 3 4 6 --tau0 0.3 0.6 1.2
  python complexity_scan.py results/566130_X05_hybrid --lmax 3 4 5 6 7       # hybrid run
  python complexity_scan.py results/566130_X05_cl --lmax 6 --full            # nested runs

The evidence decides: each complexity (spherical-harmonic degree lmax, and for the convex model
optionally the prior scale tau0 of the shape coefficients) is a separate model, and ln Z
already contains the Occam penalty for the extra coefficients. More detail is supported only
if ln Z increases.

Convex runs (no resampling). The spin-state posterior of the finished run is re-used:
    Z' / Z = E_post[ L'(theta) / L(theta) ]
(importance sampling with n posterior draws; L is the Laplace-marginalised likelihood, so each
lmax is fully marginalised over its own shape coefficients). The effective sample size of the
weights says whether this is reliable; when the posteriors at the two complexities differ a
lot (ESS fraction < 0.1), the estimate is biased low and a full run is needed (--full prints
or runs the commands, results in <outdir>_lmax<L>/, which compare_models.py picks up).

Hybrid runs: per spin state MAP + Laplace evidence of the radius-field model at each lmax
(the Matern hyperparameters sigma_f and ell are already inferred from the data within each
run, which is the hybrid model's own data-driven smoothing).

Outputs in the folder: complexity_scan.json, fig_complexity.png, and a table on screen.
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `outdir` | – | finished convex (run_inversion.py) or hybrid (run_hybrid.py) results folder |

## Options

| Option | Default | Description |
|---|---|---|
| `--lmax`&nbsp;<code>LMAX [...]</code> | `2 3 4 5 6` | spherical-harmonic degrees to compare |
| `--tau0`&nbsp;<code>TAU0 [...]</code> | – | convex: shape prior scales to scan |
| `--n`&nbsp;<code>N</code> | `200` | convex: posterior draws for importance sampling |
| `--seed`&nbsp;<code>SEED</code> | `0` | random seed of the posterior subsample |
| `--starts`&nbsp;<code>STARTS</code> | `4` | hybrid: MAP starts per spin state |
| `--map-iter`&nbsp;<code>MAP_ITER</code> | `1500` | hybrid: L-BFGS iterations |
| `--n-inner`&nbsp;<code>N_INNER</code> | `400` | convex: shape-optimisation iterations for the re-evaluated settings |
| `--full` | off | convex: run full nested sampling for each setting |
| `--print-commands` | off | convex: only print the full-run commands |

<small>10 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
