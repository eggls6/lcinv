# Python API

lcinv is a set of command-line scripts on top of an importable package. Everything the scripts do
is available from Python: preparing data, the period search, the likelihoods of all three models,
the samplers, the shape reconstruction and the reports. These pages are generated from the
docstrings and signatures of lcinv 0.2.0.

Run Python from the repository folder, put it on the path (`PYTHONPATH=/path/to/lcinv`), or install it with `pip install -e .`.
Import `lcinv.jaxenv` and call its setup before anything imports JAX if you need to choose the
device; the scripts do this through `--device`.

## By task

| Task | Modules |
|---|---|
| Read photometry | [`lcinv.data`](data.md) (plain lightcurves), [`lcinv.ades`](ades.md) (MPC ADES), [`lcinv.mpc`](mpc.md) (download) |
| Geometry | [`lcinv.ephem`](ephem.md) (Horizons, light time, caching) |
| Everything a model needs | [`lcinv.setup_data`](setup_data.md): `prepare()` returns the data dictionary |
| Period and amplitude | [`lcinv.period`](period.md) (NumPy only) |
| Global layer | [`lcinv.convex`](convex.md) (model and likelihood), [`lcinv.sampler`](sampler.md) (nested sampling, checkpoints, shape ensemble) |
| Shapes | [`lcinv.minkowski`](minkowski.md) (Gaussian image to polyhedron, OBJ) |
| Model variants | [`lcinv.smear`](smear.md) (exposure smearing), [`lcinv.binary`](binary.md) (contact binary) |
| Local layer | [`lcinv.hybrid`](hybrid.md) |
| Physics | [`lcinv.size`](size.md) (diameter from H), [`lcinv.cohesion`](cohesion.md) |
| Diagnostics | [`lcinv.fitstats`](fitstats.md), [`lcinv.diagnostics`](diagnostics.md) |
| Finished runs | [`lcinv.rebuild`](rebuild.md), [`lcinv.plots`](plots.md) |
| Devices | [`lcinv.jaxenv`](jaxenv.md) |

## A minimal session

```python
import lcinv                                   # before jax: GPU preallocation off
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion

D = prepare("data/K03Q21N.lc", "K03Q21N", location="X05", cache_dir="data")
M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"],
                    D["e_sun"], D["e_obs"], lmax=4, nfacets=400)
print(M.loglike([60.0, 70.0, 3.24836, -4.07]))  # (lambda, beta) deg, P h, ln jitter
```

From here, [Period search from Python](../../examples/period-search-api.md) and
[Likelihood and shape from Python](../../examples/convex-api.md) are complete, tested scripts.

## Conventions

- Times are Julian Dates, light-time corrected; periods are in hours; the spin parameters
  `theta` are \((\lambda, \beta)\) in degrees, \(P\) in hours and the jitter as \(\ln\sigma\) in mag.
- Ecliptic coordinates are J2000; the pole is \((\lambda, \beta)\) of the spin axis, and
  \(\beta > 0\) is prograde.
- Shapes have unit volume and the spin axis along \(z\); sizes come from `--diameter-km` or
  [`lcinv.size`](size.md).
