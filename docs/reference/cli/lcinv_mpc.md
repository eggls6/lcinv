# `lcinv.mpc`

*Download ADES observations from the MPC.*

```text
python -m lcinv.mpc [-h] [--outdir OUTDIR] [--ades-version {2017,2022}]
                 [--refresh]
                 targets [targets ...]
```

```text
Download ADES observations from the MPC
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `targets` | – | names, numbers or designations (2013_DV23 or '2013 DV23') |

## Options

| Option | Default | Description |
|---|---|---|
| `--outdir`&nbsp;<code>OUTDIR</code> | `data/mpc` | folder for the CSV files |
| `--ades-version`&nbsp;<code>ADES_VERSION</code> | `2022` | ADES version requested from the MPC One of `2017`, `2022`. |
| `--refresh` | off | download again even if the file exists |

<small>4 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
