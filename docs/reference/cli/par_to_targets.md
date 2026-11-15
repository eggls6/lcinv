# `par_to_targets`

*Convert a .par target list into a batch table.*

```text
python par_to_targets.py [-h] [-o OUTPUT] [--like LIKE] folder
```

```text
Collect period-search results (*.par, with *.obsfull) into a targets table laid out like
targets.csv, so batch_shapes.py can use it.

    python par_to_targets.py par_files/ -o targets_from_par.csv [--like targets.csv]

Every column that can be read or recomputed from the .par / .obsfull files is filled in.
Columns produced by later steps of the pipeline (period-rival analysis, spectral
classification, H_V / albedo / diameter, strength) are left empty. If you supply albedos
or diameters, a and b follow from the elongation (see axes_from_diameter).
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `folder` | – | folder with *.par files (and *.obsfull) |

## Options

| Option | Default | Description |
|---|---|---|
| `-o`, `--output`&nbsp;<code>OUTPUT</code> | `targets_from_par.csv` | output table (CSV) |
| `--like`&nbsp;<code>LIKE</code> | – | existing targets.csv: compare objects present in both |

<small>3 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
