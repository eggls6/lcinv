# `lcinv.size`

*Diameter from H (CNEOS size estimator).*

```text
python -m lcinv.size [-h] [--H H] [--albedo ALBEDO] [--cache-dir CACHE_DIR]
                  targets [targets ...]
```

```text
Diameter from H (CNEOS size estimator), H from JPL SBDB
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `targets` | – | numbers or designations |

## Options

| Option | Default | Description |
|---|---|---|
| `--H`&nbsp;<code>H</code> | – | use this H instead of looking it up |
| `--albedo`&nbsp;<code>ALBEDO</code> | `0.15` | geometric albedo (CNEOS reference range 0.05-0.30) |
| `--cache-dir`&nbsp;<code>CACHE_DIR</code> | `data/sbdb` | cache of SBDB answers |

<small>4 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
