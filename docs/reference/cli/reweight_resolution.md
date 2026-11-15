# `reweight_resolution`

*Facet-resolution check by importance reweighting.*

```text
python reweight_resolution.py [-h] [--nfacets NFACETS] [--n N] outdir
```

```text
Importance-reweight posterior samples with a finer facet discretisation.

  python reweight_resolution.py results/K03Q21N --nfacets 800 --n 300

The production run uses 400 Gaussian-image facets. Here equal-weight posterior samples are
re-evaluated with a finer model, w_i ~ exp(lnL_fine - lnL_coarse), and the effective sample
size and the shift of pole/period summaries are reported. A high ESS with negligible shifts
means the discretisation does not affect the posterior.
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `outdir` | – | finished run_inversion.py folder |

## Options

| Option | Default | Description |
|---|---|---|
| `--nfacets`&nbsp;<code>NFACETS</code> | `800` | facets of the finer model |
| `--n`&nbsp;<code>N</code> | `300` | posterior samples to reweight |

<small>3 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
