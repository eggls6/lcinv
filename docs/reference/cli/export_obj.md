# `export_obj`

*Watertight .obj meshes of the posterior shapes.*

```text
python export_obj.py [-h] [--n N] [--frame {body,principal}]
                     [--diameter-km DIAMETER_KM | --scale SCALE] [--seed SEED]
                     [--H H] [--albedo ALBEDO] [--size-from-H]
                     outdir
```

```text
Export posterior shapes as Wavefront .obj files (global or hybrid results).

    python export_obj.py results/K03Q21N                 # global (convex) run
    python export_obj.py results/K03Q21N_hybrid_quat     # hybrid run, every spin state
    python export_obj.py results/X --n 50 --diameter-km 1.2 --frame principal

Global run (nested_results.pkl), written to <outdir>/obj/:
    shape_maxlike.obj   maximum-likelihood convex shape
    shape_mean.obj      Minkowski mean: the convex body whose facet areas are the posterior
                        mean areas (always a valid convex shape)
    draw_###.obj        posterior draws (spin sample + Laplace draw of the shape)

Hybrid run (posterior_spin_state_*.npz), written to <outdir>/obj/spin_state_<k>/:
    shape_mean.obj, shape_median.obj   mean / median radius at each mesh direction
    shape_p16.obj, shape_p84.obj       16th / 84th percentile radius surfaces (+-1 sigma envelopes)
    draw_###.obj                       NUTS draws

Frames: --frame body (default) keeps the body frame of the inversion: z = spin axis, and the
rotational phase is zero at t_ref (given in the file header), so draws are mutually aligned.
--frame principal rotates each shape to its own principal axes of inertia (x: smallest
moment = longest axis, z: largest moment), which suits 3D printing or comparison with other
models but discards the spin orientation.

Units: by default every shape is scaled to a volume-equivalent radius R_eq = 1 (the volume of
a unit sphere). Lightcurves alone do not fix the size: --diameter-km D scales to a
volume-equivalent diameter of D km (e.g. from H and an assumed albedo); --scale s sets R_eq = s.
Each export replaces the .obj files of the previous one in the same folder.
Every file is centred on its centre of mass (uniform density) and has outward-facing triangles.
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `outdir` | – | results folder of run_inversion.py or run_hybrid.py |

## Options

| Option | Default | Description |
|---|---|---|
| `--n`&nbsp;<code>N</code> | `20` | number of posterior draws to export |
| `--frame`&nbsp;<code>FRAME</code> | `body` | body: z = spin axis, phase 0 at t_ref (draws aligned); principal: each shape in its own principal-axis frame One of `body`, `principal`. |
| `--diameter-km`&nbsp;<code>DIAMETER_KM</code> | – | volume-equivalent diameter [km] |
| `--scale`&nbsp;<code>SCALE</code> | `1.0` | volume-equivalent radius in output units |
| `--seed`&nbsp;<code>SEED</code> | `0` | seed for choosing the hybrid draws |
| `--H`&nbsp;<code>H</code> | – | absolute magnitude: scale to the CNEOS size estimate D = 10^(3.1236 - 0.5 log10 p - 0.2 H) km |
| `--albedo`&nbsp;<code>ALBEDO</code> | `0.15` | geometric albedo for --H / --size-from-H |
| `--size-from-H` | off | look H up in the JPL Small-Body Database and scale to the CNEOS size estimate |

<small>9 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
