"""lcinv: Bayesian lightcurve inversion of asteroids from survey photometry.

Two layers share one data preparation:

* the **global layer** samples the spin state (pole, sidereal period, jitter, optionally the
  Lambert weight and the shape-prior scale) with dynesty nested sampling, marginalising a convex
  shape by a Laplace approximation (`run_inversion.py`);
* the **hybrid layer** samples spin and a non-convex radius-field shape jointly with NUTS in each
  spin state the global layer found (`run_hybrid.py`).

Modules
-------
Data and geometry
    `data` (lightcurve files), `ades` (MPC ADES), `mpc` (download from the MPC),
    `ephem` (JPL Horizons, light time, caching), `setup_data` (``prepare()``: everything a model
    needs).
Period and amplitude
    `period` (multiband Fourier periodogram, refinement, amplitude, a/b bound).
Models
    `convex` (global-layer model and likelihood), `smear` (exposure smearing),
    `binary` (contact-binary model), `hybrid` (local layer, a subpackage).
Inference and shapes
    `sampler` (nested sampling, checkpoints, shape ensemble), `minkowski` (Gaussian image to
    polyhedron, OBJ).
Physics and diagnostics
    `size` (diameter from H), `cohesion` (Sanchez & Scheeres 2014), `fitstats` (goodness of fit),
    `diagnostics` (convergence reports).
Finished runs and figures
    `rebuild` (models from results folders), `plots`, `folding`.
Devices
    `jaxenv` (JAX platform and GPU memory; import before JAX).

Documentation: https://USER.github.io/lcinv/
"""
__version__ = "0.2.0"

import os as _os

# Disable JAX's default GPU preallocation (75% of device memory per process), which makes
# spawned likelihood workers fail with CUDA_ERROR_OUT_OF_MEMORY. Effective only if lcinv is
# imported before jax; entry scripts call lcinv.jaxenv.early_setup() for full control.
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
