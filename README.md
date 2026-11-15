# lcinv

Bayesian lightcurve inversion of asteroids from sparse survey photometry: pole, sidereal
rotation period and shape, with uncertainties.

- A **global layer** samples the spin state with dynesty nested sampling and marginalises a
  convex shape by a Laplace approximation. It finds every pole solution and its evidence,
  including the mirror solutions a single apparition cannot separate.
- A **hybrid layer** then samples spin and a non-convex radius-field shape jointly with NUTS
  inside each solution, with a principal-axis (inertia) prior.
- A **contact-binary model**, exposure smearing for fast rotators, a fitted scattering law,
  model comparison by evidence and a cohesion limit come with it.

It reads Rubin-style lightcurve files and MPC ADES files directly and gets the viewing geometry
from JPL Horizons.

**Documentation:** https://USER.github.io/lcinv/ (sources in [`docs/`](docs/)).

## Install

```bash
git clone https://github.com/USER/lcinv.git
cd lcinv
pip install -r requirements.txt        # or: pip install -e ".[netcdf,test]"
```

Python 3.10 or newer. JAX runs on CPU out of the box; for NVIDIA GPUs install the CUDA build of
JAX. See [Installation](https://USER.github.io/lcinv/getting-started/installation/).

## Quickstart

The bundled Rubin data of 2003 QN21 (K03Q21N) and their Horizons geometry are in `data/`, so
this runs offline: a five-minute first pass, its continuation and the hybrid model, about 40
minutes in all on a two-core laptop.

```bash
python run_inversion.py data/K03Q21N.lc --designation K03Q21N --location X05 --cache-dir data \
    --nlive 60 --dlogz 1.0 --maxcall 4000 --npool 2 --device cpu --outdir runs/K03Q21N_quick
python run_inversion.py data/K03Q21N.lc --designation K03Q21N --location X05 --cache-dir data \
    --nlive 60 --dlogz 1.0 --npool 2 --device cpu --outdir runs/K03Q21N_quick \
    --resume --extra-calls 20000
python run_hybrid.py data/K03Q21N.lc --designation K03Q21N --location X05 --cache-dir data \
    --global-dir runs/K03Q21N_quick --chains 2 --warmup 150 --samples 150 --starts 2 \
    --map-iter 800 --device cpu --outdir runs/K03Q21N_quick_hybrid
```

The [Quickstart](https://USER.github.io/lcinv/getting-started/quickstart/) walks through the
output; production settings and their results are in
[Reference results](https://USER.github.io/lcinv/examples/reference-results/).

## What is in the repository

| Path | Content |
|---|---|
| `lcinv/` | the Python package (`lcinv.hybrid` is the local layer) |
| `run_inversion.py`, `run_hybrid.py`, `run_binary.py` | the three models |
| `batch_shapes.py`, `compare_models.py`, `complexity_scan.py`, `check_convergence.py`, `export_obj.py`, `make_viewer.py`, ... | batch runs, model comparison, diagnostics and products; every script has `--help` |
| `data/` | bundled photometry (K03Q21N, 566130) with cached Horizons tables |
| `results/` | finished reference runs of both targets |
| `tests/` | tests (`python -m pytest tests`) and development diagnostics |
| `examples/` | example scripts; `examples/docs/` holds those used in the documentation |
| `paper/`, `paper_hybrid/` | methods notes (LaTeX and PDF) |
| `docs/`, `mkdocs.yml`, `tools/` | documentation site (MkDocs Material) |

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests -q
```

The seven test modules take about two minutes on a laptop CPU; what they check is described in
[Validation](https://USER.github.io/lcinv/validation/).

## Documentation site

```bash
pip install -r requirements-docs.txt
mkdocs serve                       # preview at http://127.0.0.1:8000
python tools/gen_cli_docs.py       # after changing command-line options
```

`.github/workflows/docs.yml` builds the site and publishes it to GitHub Pages on every push to
`main` (Settings -> Pages -> Source: GitHub Actions).

## Methods

The methods are described in two notes shipped with the code,
[`paper/lcinv_methods.pdf`](paper/lcinv_methods.pdf) (global layer) and
[`paper_hybrid/hybrid_methods.pdf`](paper_hybrid/hybrid_methods.pdf) (hybrid layer). Please cite
them together with the version of lcinv you used. Changes between versions are listed in
[`CHANGELOG.md`](CHANGELOG.md).
