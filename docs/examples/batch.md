# Batch processing

`batch_shapes.py` turns a table of targets into a set of finished analyses. Before committing
hours of computing, `--dry-run` prints every command the batch would run. This example builds a
two-target table from the bundled data and checks the plan.

## The table

`examples/docs/targets.csv`:

```text
--8<-- "examples/docs/targets.csv"
```

The periods are **single-peaked** Fourier periods, half the rotation periods (0.178 h and
3.248 h), as catalogue tables usually list them; `--period-factor 2` (the default) doubles them.
566130 has no H, and the SBDB lookup is switched off (`--no-sbdb`), so it gets no size;
K03Q21N has H = 16.

## The dry run

```bash
python batch_shapes.py examples/docs/targets.csv \
    --data-pattern "data/{num}_ades.csv" --data-pattern "data/{num}.lc" \
    --common-args "--observer local --fit-cl" --inversion-args "--nlive 300" --hybrid \
    --outroot runs/batch --dry-run --no-sbdb
```

<div class="run-output" markdown>

```text
--8<-- "examples/docs/batch_dry_run.txt"
```

</div>

<small>The real output prints absolute paths; they are shortened here to paths relative to the
repository folder.</small>

## What it shows

- **Data lookup.** Both `--data-pattern`s are tried in order: 566130 finds its ADES file,
  K03Q21N its plain lightcurve. Every run gets `--designation` from `num`.
- **Period window.** The rotation period \(2 \times\) `period_h` is searched within ±5 %:
  0.16908–0.18688 h for 566130 and 3.0858–3.4107 h for K03Q21N. No full 1–30 h search, so no
  danger of the alias trap shown in the [ADES example](ades-566130.md#what-happens-with-the-default-search-range).
- **Size.** For K03Q21N, H = 16 and albedo 0.15 give
  \(D = 10^{3.1236 - 0.5\log_{10}0.15 - 0.2\cdot16} = 2.165\) km, passed as `--diameter-km` to
  the models (cohesion report) and to `export_obj.py`.
- **Stages.** Global inversion, hybrid model, convergence checks, OBJ export and viewers for
  both layers, then a model comparison per target.

Dropping `--dry-run` runs the plan, with one folder per target under `runs/batch/` and a
`batch_summary.csv` at the top. An interrupted batch picks up where it stopped. Options:
[Batch runs and MPC download](../guide/batch.md).
