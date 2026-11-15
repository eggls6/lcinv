# `make_uncertainty_viewer`

*3D viewer of the shape uncertainty (radius spread).*

```text
python make_uncertainty_viewer.py [-h] [--n N] [--ndir NDIR] outdir
```

```text
Mean convex shape coloured by posterior radius uncertainty (self-contained HTML, three.js).

  python make_uncertainty_viewer.py results/K03Q21N [--n 150]
      -> results/K03Q21N/shape_uncertainty_viewer.html

Steps
  1. Reconstruct the convex polyhedron of every posterior shape draw (spin-state posterior
     sample + Laplace draw of the shape coefficients) by solving the Minkowski problem.
  2. Register the draws: the rotational phase is a gauge freedom, so each body may be rotated
     arbitrarily about its spin axis (z). Radii on a latitude-longitude grid make a rotation
     about z a circular shift in longitude; each draw is rotated to best match the reference
     (first the max-likelihood shape, then the ensemble mean; two passes).
  3. Radius r(u) from the centroid along ~2500 directions u for every registered draw:
     mean, standard deviation and 16/50/84th percentiles per direction.
  4. Render the mean-radius surface coloured by sigma_r (viridis), with optional +-1 sigma
     envelopes, the spin axis and the maximum-inertia axis of the mean shape.

Radii are in units of the volume-equivalent sphere radius R_eq (all draws have volume 1).
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `outdir` | – | finished run_inversion.py folder |

## Options

| Option | Default | Description |
|---|---|---|
| `--n`&nbsp;<code>N</code> | `150` | number of posterior shape draws |
| `--ndir`&nbsp;<code>NDIR</code> | `2562` | surface directions for the mesh |

<small>3 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
