# Contact-binary model

Some asteroids are two lobes resting on each other. A convex model can only approximate them, and
the lightcurve of a contact binary has features, sharp minima from one lobe hiding or shadowing
the other, that no convex body produces. `run_binary.py` fits that hypothesis directly and
reports its evidence against the convex model.

```bash
python run_binary.py --global-dir results/566130_X05_cl --outdir results/566130_X05_binary
```

## The model

Two triaxial ellipsoid lobes in contact (`lcinv.binary`), with

| Parameter | Prior (uniform) | Option |
|---|---|---|
| lobe size ratio \(q\) | 0.3–1.0 | `--q-range` |
| lobe axis ratio \(b/a\) (shared) | 0.5–1.0 | `--b-range` |
| lobe axis ratio \(c/b\) (shared) | 0.5–1.0 | `--cb-range` |
| separation \(s\): centre distance in units of \(a_1 + a_2\) | 0.7–0.98 | `--s-range` |
| pole, period, jitter, optionally \(\ln c_L\) | as in the convex model | |
| rotational phase \(\varphi_0\) | uniform | |

That is 9 or 10 sampled parameters. \(s = 1\) means touching; smaller values overlap.

Each lobe is convex, so a facet can only be hidden or shadowed by the **other** lobe. lcinv
computes this analytically (ray-ellipsoid closest approach, with soft edges of width `--soft`,
0.02) and removes facets inside the other lobe (the neck). Against an independent z-buffer
render of the union mesh the brightness agrees to 1–10 mmag rms (`tests/test_binary.py`).
Exposure smearing and the scattering law are the same as in the convex model.

## Comparing with the convex model

With `--global-dir`, the data selection (file, stations, observer, exposure time, clipped
outliers), the period prior and the \(c_L\) setting come from that convex run. Both evidences
then use the same data, period and jitter priors and the same profiled band offsets, and
`summary.json` reports `evidence_comparison`: \(\ln Z_\mathrm{binary} - \ln Z_\mathrm{convex}\),
read as favoured above 2.3 and strongly favoured above 5.

!!! warning "Different shape priors"
    The binary model has uniform priors on four lobe parameters; the convex model marginalises
    24 harmonic coefficients by Laplace. Small evidence differences between the two deserve
    caution. [Model comparison](model-comparison.md) puts all runs of a target in one table.

Without a global run, give the data and the period:

```bash
python run_binary.py data/566130_ades.csv --stations X05 --observer local --exptime 30 \
       --period 0.1779775 --period-sigma 2e-6 --outdir results/566130_binary
```

## Outputs

Like a global run: `summary.json`, `nested_results.pkl`, `shapes/*.obj` (watertight union
meshes, unit volume-equivalent radius, or km with `--diameter-km`), `fig_pole_posterior.png`,
`fig_corner.png`, `fig_lightcurves.png`, `fig_folded_lightcurve.png`, `shape_viewer.html`.
`check_convergence.py` works on the folder, and `run_binary.py --replot --outdir DIR` rebuilds
the products without sampling (`make_viewer.py`, `export_obj.py` and `make_folded_plots.py` do
this automatically for binary folders). In a batch: `batch_shapes.py --binary`.

For 566130 the binary model is strongly disfavoured: `evidence_comparison` gives
\(\Delta\ln Z = -9.2\) against the convex run it was built on, and −12 against the best convex
setting of the [complexity scan](../examples/reference-results.md#566130-model-comparison). The
[synthetic example](../examples/synthetic-binary.md) shows what a real contact binary's
lightcurve looks like next to an ellipsoid's.

All options: [`run_binary.py` reference](../reference/cli/run_binary.md).
