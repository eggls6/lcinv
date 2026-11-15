# Installation

lcinv is a set of scripts around the `lcinv` Python package. There is nothing to build: clone the
repository, install the dependencies, and run the scripts from the repository folder.

## Requirements

- Python 3.10 or newer (the examples in these pages were run with Python 3.13).
- The scientific stack listed in `requirements.txt` and `pyproject.toml`: NumPy, SciPy (1.15
  or newer), pandas, Matplotlib, JAX, Optax, dynesty, NumPyro, ArviZ, h5netcdf, h5py, Astropy,
  astroquery and corner.
- Network access to JPL Horizons the first time a target is run. Geometry is cached in
  `--cache-dir`, so later runs work offline. Both bundled targets ship with their caches.

## Install

```bash
git clone https://github.com/USER/lcinv.git
cd lcinv
python -m venv .venv && source .venv/bin/activate     # or a conda environment
pip install -r requirements.txt
```

To import `lcinv` from anywhere, also install the package itself in editable mode
(`pip install -e .`); the scripts are always run from the repository folder.

Check the installation:

```bash
python -c "import lcinv; print(lcinv.__version__)"     # 0.2.0
python run_inversion.py --help | head -5
python -m pytest tests -q                              # the test suite, about 2 minutes
```

!!! note "ArviZ NetCDF export"
    `h5netcdf` is only needed to write the hybrid posteriors as ArviZ NetCDF files
    (`posterior_spin_state_*.nc`). Without it the posteriors are still saved as `.npz`, which is
    what every lcinv script reads.

## GPU support

JAX runs on CPU out of the box. For an NVIDIA GPU, install the CUDA build of JAX
([JAX installation guide](https://docs.jax.dev/en/latest/installation.html)), for example
`pip install -U "jax[cuda12]"`.

Use the GPU where it pays off. The global layer evaluates a small convex model many times, so
CPU workers in parallel usually beat a GPU there. The hybrid layer's NUTS sampler is where a GPU
helps. A typical split on a workstation:

```bash
python run_inversion.py data/X.lc --designation X --device cpu --npool 8 --outdir results/X
python run_hybrid.py   data/X.lc --designation X --global-dir results/X --device gpu --chains 4 \
       --outdir results/X_hybrid
```

[GPU and performance](../guide/performance.md) explains the memory settings, worker threads and
how to pick `--npool`.

!!! warning "Importing lcinv from your own code"
    JAX reserves about 75 % of GPU memory in every process unless told otherwise, which makes the
    likelihood workers fail with `CUDA_ERROR_OUT_OF_MEMORY`. lcinv switches preallocation off,
    but only if it is imported **before** `jax`:

    ```python
    import lcinv            # first: sets XLA_PYTHON_CLIENT_PREALLOCATE=false
    import jax
    ```

    Or call `lcinv.jaxenv.configure(device="cpu")` before importing JAX, or set
    `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the shell.

## Building these docs

The documentation lives in `docs/` with `mkdocs.yml` in the project root.

```bash
pip install -r requirements-docs.txt
mkdocs serve                        # live preview at http://127.0.0.1:8000
python tools/gen_cli_docs.py        # after changing any command-line option
```

Pushing to `main` publishes the site to GitHub Pages through `.github/workflows/docs.yml`.

## Troubleshooting

`CUDA_ERROR_OUT_OF_MEMORY` when the global run starts its workers
:   Leave `--worker-device` at `cpu` (the default) or cap the memory per process with
    `--gpu-mem-fraction 0.2`. See [GPU and performance](../guide/performance.md).

`INTERNAL: RET_CHECK failure ... xla/service/gpu/...` during sampling
:   This is an internal error of JAX's GPU compiler, not of lcinv or your data; it has been seen
    with JAX 0.12 on NVIDIA L40S cards. Run on CPU with `--device cpu --worker-device cpu`, or
    try another `jax`/`jaxlib` version for GPU runs. The first line of every run prints the
    backend in use (`JAX ... backend=cpu`).

Horizons queries fail or time out
:   Check network access, then rerun: finished queries are cached and not repeated. For ADES
    data, `--observer local` needs only the heliocentric table (one query instead of one per
    station).

`--designation is required for non-ADES lightcurves`
:   Plain lightcurve files carry no designation; pass `--designation`. ADES files supply it from
    `permID`/`provID`.
