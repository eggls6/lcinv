# GPU and performance

## Where the time goes

| Stage | Cost driver | Typical |
|---|---|---|
| period search | NumPy least squares per trial period | seconds (single apparition) |
| global layer | likelihood calls × (shape optimisation + Laplace step) | ~20 calls/s per core for 334 points; 4,000–60,000 calls |
| hybrid layer | NUTS gradients, 60–130 leapfrog steps per iteration | ~10 ms per gradient (16 × 32 mesh) |

On the 2-core machine used for the [examples](../examples/index.md), a 4,000-call global run of
K03Q21N took under five minutes, and finishing it to convergence took 19,000 calls in total.

## Devices

lcinv configures JAX before it is imported (`lcinv.jaxenv`):

- **Preallocation is off** (`XLA_PYTHON_CLIENT_PREALLOCATE=false`). By default JAX takes about
  75 % of GPU memory in every process, and the global layer's worker pool would fail with
  `CUDA_ERROR_OUT_OF_MEMORY`. `--gpu-preallocate` restores the JAX default.
- `--device {auto,cpu,gpu}` selects the platform of the main process; `--gpu-mem-fraction F`
  caps the memory per process.
- `--worker-device {cpu,gpu,auto}` (global layer and binary, default `cpu`) sets the platform of
  the pool workers. The convex likelihood is small, so CPU workers in parallel are usually faster
  than several processes queueing on one GPU.
- **CPU workers are single-threaded** (`--worker-threads`, default 1). Otherwise every worker
  starts XLA and BLAS thread pools as large as the machine, and `--npool 16` runs hundreds of
  threads that mostly wait on each other. Keep `npool × worker-threads` at or below the number of
  physical cores.
- **NUTS chains** run in parallel on CPU host devices, or vectorised on one GPU.

Recommended on a GPU workstation:

```bash
python run_inversion.py data/X.lc --designation X --device cpu --npool 8 --outdir results/X
python run_hybrid.py   data/X.lc --designation X --global-dir results/X --device gpu --chains 4 \
       --outdir results/X_hybrid
```

The first line of every run shows what JAX is using:

```text
JAX 0.12.0.1 backend=cpu devices=1 [cpu] preallocate=false
```

## Making the global layer faster

- More workers (`--npool`): each `rslice` worker runs its own slice chain.
- Fewer live points for exploration, more for the final run: `--nlive 100` maps the spin states
  quickly; `--nlive 400 --dlogz 0.1` gives publication-grade results.
- Interrupted or limited runs are not lost: `--resume --extra-calls N` continues from the
  checkpoint.
- Multi-apparition data: restrict the period search with `--search-window`
  ([Period search](period-search.md#data-spanning-several-apparitions)).

## Making the hybrid layer faster

- `--device gpu` with several `--chains` (vectorised on one GPU).
- A smaller mesh (`--nlat`) or `--lmax` for exploration; the mesh error is reported against a
  finer reference.
