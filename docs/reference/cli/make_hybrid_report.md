# `make_hybrid_report`

*Rebuild all hybrid figures and viewers without resampling.*

```text
python make_hybrid_report.py [-h] [--global-dir GLOBAL_DIR] [--lc LC]
                             [--designation DESIGNATION] [--lmax LMAX]
                             [--nlat NLAT] [--period PERIOD]
                             [--period-sigma PERIOD_SIGMA]
                             outdir
```

```text
Rebuild the hybrid report (figures + per-spin-state HTML viewers) from saved posteriors.

  python make_hybrid_report.py results/K03Q21N_hybrid [--global-dir results/K03Q21N]

Reads posterior_spin_state_*.npz (or legacy posterior_mode*.npz) and, if present,
summary_hybrid.json (for spin-state weights and
settings). Useful after a run was interrupted or to restyle figures without resampling.
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `outdir` | – | hybrid results folder (run_hybrid.py) |

## Options

| Option | Default | Description |
|---|---|---|
| `--global-dir`&nbsp;<code>GLOBAL_DIR</code> | – | global run, for the global pole posterior (default: from summary_hybrid.json) |
| `--lc`&nbsp;<code>LC</code> | – | lightcurve file (default: from summary_hybrid.json) |
| `--designation`&nbsp;<code>DESIGNATION</code> | – | target designation (default: from summary_hybrid.json) |
| `--lmax`&nbsp;<code>LMAX</code> | `5` | radius-field degree (default: from summary_hybrid.json) |
| `--nlat`&nbsp;<code>NLAT</code> | `16` | grid latitude nodes (default: from summary_hybrid.json) |
| `--period`&nbsp;<code>PERIOD</code> | – | period prior centre [h] (default: from summary_hybrid.json) |
| `--period-sigma`&nbsp;<code>PERIOD_SIGMA</code> | – | period prior width [h] (default: from summary_hybrid.json) |

<small>8 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
