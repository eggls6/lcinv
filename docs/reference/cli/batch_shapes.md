# `batch_shapes`

*Whole pipeline for every row of a target table.*

```text
python batch_shapes.py [-h] [--outroot OUTROOT] [--data-pattern DATA_PATTERN]
                       [--data-column DATA_COLUMN]
                       [--period-factor PERIOD_FACTOR] [--window WINDOW]
                       [--download] [--download-dir DOWNLOAD_DIR]
                       [--refresh-data] [--ades-version {2017,2022}]
                       [--hybrid] [--binary] [--binary-args BINARY_ARGS]
                       [--size-source {H,table}] [--albedo ALBEDO]
                       [--albedo-from-table] [--no-sbdb] [--sbdb-dir SBDB_DIR]
                       [--cohesion-max PA] [--density DENSITY] [--complexity]
                       [--complexity-args COMPLEXITY_ARGS] [--no-compare]
                       [--jobs JOBS] [--npool NPOOL] [--device {cpu,gpu}]
                       [--common-args COMMON_ARGS]
                       [--inversion-args INVERSION_ARGS]
                       [--hybrid-args HYBRID_ARGS] [--ab-from-table]
                       [--no-scale-obj] [--timeout-h TIMEOUT_H]
                       [--query QUERY] [--only ONLY [ONLY ...]]
                       [--limit LIMIT] [--force] [--dry-run]
                       table
```

```text
Batch shape modelling for a table of targets (pandas DataFrame).
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `table` | – | CSV/Parquet/Feather/pickle/Excel with columns num, period_h, ... |

## Options

| Option | Default | Description |
|---|---|---|
| `--outroot`&nbsp;<code>OUTROOT</code> | `results/batch` | folder for the per-target results and batch_summary.csv |
| `--data-pattern`&nbsp;<code>DATA_PATTERN</code> | – | photometry path template with {num}, {desig}, {desig_nospace} or {name}; repeatable, first existing file wins (default data/{num}_ades.csv, data/{num}.csv, data/{num}.lc) |
| `--data-column`&nbsp;<code>DATA_COLUMN</code> | – | table column holding the photometry path per row |
| `--period-factor`&nbsp;<code>PERIOD_FACTOR</code> | `2.0` | rotation period = factor x period_h (2: table periods are single-peaked) |
| `--window`&nbsp;<code>WINDOW</code> | `0.05` | fractional half-width of the period search around the rotation period |
| `--hybrid` | off | also run the hybrid model |
| `--binary` | off | also fit the contact-binary model (run_binary.py) and compare its evidence |
| `--binary-args`&nbsp;<code>BINARY_ARGS</code> | `""` | extra options for run_binary.py, e.g. "--nlive 400" |
| `--jobs`&nbsp;<code>JOBS</code> | `1` | targets processed in parallel |
| `--npool`&nbsp;<code>NPOOL</code> | `2` | likelihood workers per global run |
| `--device`&nbsp;<code>DEVICE</code> | `cpu` | JAX device of the model scripts One of `cpu`, `gpu`. |
| `--common-args`&nbsp;<code>COMMON_ARGS</code> | `""` | options for both run_inversion.py and run_hybrid.py, e.g. "--stations X05 --observer local --exptime 30 --fit-cl" |
| `--inversion-args`&nbsp;<code>INVERSION_ARGS</code> | `""` | extra options for run_inversion.py, e.g. "--nlive 300" |
| `--hybrid-args`&nbsp;<code>HYBRID_ARGS</code> | `""` | extra options for run_hybrid.py, e.g. "--chains 4" |
| `--ab-from-table` | off | use a/b from the table as the a/b prior bound |
| `--no-scale-obj` | off | do not scale OBJ files to the size estimate (Flag: turns the behaviour off.) |
| `--timeout-h`&nbsp;<code>TIMEOUT_H</code> | – | per-step time limit for global/hybrid runs |
| `--query`&nbsp;<code>QUERY</code> | – | pandas query to select rows, e.g. "chi2_red &lt; 2" |
| `--only`&nbsp;<code>ONLY [...]</code> | – | process only these targets (num values) |
| `--limit`&nbsp;<code>LIMIT</code> | – | process only the first N selected rows |
| `--force` | off | rerun targets that already finished |
| `--dry-run` | off | print the commands only |

## MPC download

| Option | Default | Description |
|---|---|---|
| `--download` | off | download ADES observations from the MPC for targets without a local file |
| `--download-dir`&nbsp;<code>DOWNLOAD_DIR</code> | `data/mpc` | where downloaded files are stored (&lt;num&gt;_ades.csv); also searched for data |
| `--refresh-data` | off | download again even if a downloaded file exists (new observations) |
| `--ades-version`&nbsp;<code>ADES_VERSION</code> | `2022` | ADES version requested from the MPC One of `2017`, `2022`. |

## Size (scales OBJ files, used for cohesion)

| Option | Default | Description |
|---|---|---|
| `--size-source`&nbsp;<code>SIZE_SOURCE</code> | `H` | H: CNEOS size estimate D = 10^(3.1236 - 0.5 log10 p - 0.2 H) km (default); table: the diameter_km column One of `H`, `table`. |
| `--albedo`&nbsp;<code>ALBEDO</code> | `0.15` | geometric albedo for the H-based size |
| `--albedo-from-table` | off | use the table's albedo_assumed column where present |
| `--no-sbdb` | off | do not look up H in the JPL Small-Body Database when the table has no H column (Flag: turns the behaviour off.) |
| `--sbdb-dir`&nbsp;<code>SBDB_DIR</code> | `data/sbdb` | cache of SBDB responses |

## Physics, complexity and model comparison

| Option | Default | Description |
|---|---|---|
| `--cohesion-max`&nbsp;<code>PA</code> | – | penalise shapes/spins needing more cohesion than this (Sanchez & Scheeres 2014), using the size estimate (and the table's density column if present) |
| `--density`&nbsp;<code>DENSITY</code> | `1700.0` | bulk density [kg/m^3] when the table has no density column |
| `--complexity` | off | run complexity_scan.py on every global run (evidence vs lmax / tau0) |
| `--complexity-args`&nbsp;<code>COMPLEXITY_ARGS</code> | `--lmax 3 4 6 8 --tau0 0.6 1.2` | options passed to complexity_scan.py (quoted) |
| `--no-compare` | off | skip compare_models.py (model_comparison.md per target) (Flag: turns the behaviour off.) |

```text
Arguments after the options are not allowed; pass script options via --common-args / --inversion-args / --hybrid-args (quoted).
```

<small>37 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
