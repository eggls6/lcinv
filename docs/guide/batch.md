# Batch runs and MPC download

`batch_shapes.py` runs the whole pipeline for every row of a table of targets: global inversion,
optionally the hybrid and binary models, convergence report, complexity scan, model comparison,
OBJ export scaled to a size estimate, and 3D viewers.

```bash
python batch_shapes.py targets.csv --data-pattern "data/{num}_ades.csv" \
    --common-args "--stations X05 --observer local --exptime 30 --fit-cl" \
    --inversion-args "--nlive 300" --hybrid --hybrid-args "--chains 4 --warmup 1000 --samples 1500" \
    --outroot results/batch --jobs 2 --npool 4
```

## The target table

Any pandas-readable table: CSV, Parquet, Feather, pickle or Excel.

| Column | Required | Meaning |
|---|---|---|
| `num` | yes | target, e.g. `2013_DV23` or `722574`; `_` becomes a space for Horizons |
| `period_h` | yes | period from a period search, in hours |
| `H`, `H_V` or `Hmag` | no | absolute magnitude for the size estimate (else looked up in the SBDB) |
| `diameter_km` | no | used with `--size-source table`, or when no H is available |
| `density` | no | bulk density for the cohesion calculation (else `--density`) |
| `albedo_assumed` | no | used with `--albedo-from-table` |
| `a`, `b` | no | with `--ab-from-table`, a/b is the lower bound of the a/b prior |
| `file` / `path` | no | per-row photometry path, with `--data-column` |

All other columns are carried through to `batch_summary.csv` unchanged.

**Period convention.** The lightcurve of an elongated body has two maxima per rotation, so a
single-peaked Fourier period is half the rotation period. `--period-factor` (default 2) converts
the table period into the rotation period that is searched, within ±`--window` (default 5 %).
Use `--period-factor 1` if `period_h` is already the double-peaked rotation period.
`par_to_targets.py` converts a folder of `.par` files into such a table.

## Where the photometry comes from

- `--data-pattern` with `{num}`, `{desig}`, `{desig_nospace}` or `{name}`; repeatable, the first
  existing file wins;
- `--data-column`: a table column holding a path per row;
- `--download`: fetch ADES observations from the MPC Observations API
  (`https://data.minorplanetcenter.net/api/get-obs`) for targets without a local file, cached as
  `<--download-dir>/<num>_ades.csv` (default `data/mpc`) in the layout of an MPC ADES export.

The downloaded files contain every station; add `--stations X05` to `--common-args` to use only
Rubin photometry. `--refresh-data` downloads again (after a new data release); combine it with
`--force` to recompute the shape models. Standalone:

```bash
python -m lcinv.mpc 2013_DV23 722574 --outdir data/mpc
```

## Output and resuming

Each target gets `<outroot>/<num>/` with `pipeline.log`, `status.json`, `convergence.txt` and the
run folders; `<outroot>/batch_summary.csv` has one row per target: the input columns plus fitted
period, pole, prograde fraction, jitter, \(\ln Z\), convergence verdicts, size used and the
model-comparison verdict.

Finished targets are skipped on reruns (`--force` reruns them) and finished stages are reused, so
an interrupted batch resumes where it stopped.

| Option | Effect |
|---|---|
| `--dry-run` | print the commands only |
| `--query "chi2_red < 2"` | select rows with a pandas query |
| `--only 722574 2013_DV23` | only these targets |
| `--limit N` | the first N selected rows |
| `--timeout-h H` | time limit per global or hybrid step |
| `--jobs`, `--npool` | targets in parallel, likelihood workers per run |
| `--hybrid`, `--binary`, `--complexity` | extra stages |
| `--no-compare` | skip `compare_models.py` |
| `--size-source`, `--albedo`, `--no-sbdb` | size estimate for OBJ scaling and cohesion |

From Python: `from batch_shapes import run_batch; run_batch(df, ...)`. The
[batch example](../examples/batch.md) shows a dry run on the bundled data, and the
[`batch_shapes.py` reference](../reference/cli/batch_shapes.md) lists every option.
