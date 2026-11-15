"""JAX device and memory configuration (must take effect before JAX is imported).

Why this exists
---------------
By default JAX preallocates ~75% of GPU memory in every process at first use. The global layer
evaluates its likelihood in a process pool (``--npool``); each spawned worker imports JAX and
tries to grab its own 75% of a GPU the parent already holds, which fails with
``CUDA_ERROR_OUT_OF_MEMORY``. Remedies applied here:

* preallocation is disabled by default (``XLA_PYTHON_CLIENT_PREALLOCATE=false``), so each process
  allocates only what it needs; ``--gpu-mem-fraction`` can cap it instead;
* ``--device {auto,cpu,gpu}`` selects the platform of the main process;
* pool workers get their own platform (default CPU: the convex likelihood is small, so GPU
  kernel-launch latency outweighs the arithmetic and several processes would compete for one GPU);
* CPU pool workers are pinned to ``--worker-threads`` threads each (default 1). Otherwise every
  worker starts an XLA/Eigen and BLAS thread pool the size of the whole machine, and
  ``--npool 16`` runs 16 x ncores threads that mostly wait on each other.

This module must not import JAX: call :func:`early_setup` at the very top of an entry script,
before anything that imports ``jax``.
"""
import contextlib
import os
import sys

_PLATFORM = {"cpu": "cpu", "gpu": "cuda", "cuda": "cuda", "rocm": "rocm"}


def _argval(argv, flag, default=None):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def configure(device="auto", preallocate=False, mem_fraction=None, host_devices=None):
    """Set JAX environment variables. Has no effect if JAX was already imported."""
    if "jax" in sys.modules:
        print("[jaxenv] warning: JAX already imported; device/memory settings may not apply")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    if device and device != "auto":
        os.environ["JAX_PLATFORMS"] = _PLATFORM[device]
    if host_devices and int(host_devices) > 1:
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = (flags + f" --xla_force_host_platform_device_count={int(host_devices)}").strip()


def early_setup(argv=None, host_devices_flag="--chains", default_host_devices=1):
    """Parse --device / --gpu-mem-fraction / --gpu-preallocate (and the host-device count)
    from argv and configure JAX before it is imported."""
    argv = sys.argv if argv is None else argv
    device = _argval(argv, "--device", "auto")
    frac = _argval(argv, "--gpu-mem-fraction", None)
    prealloc = "--gpu-preallocate" in argv
    nhost = _argval(argv, host_devices_flag, default_host_devices) if host_devices_flag else None
    configure(device=device, preallocate=prealloc, mem_fraction=frac, host_devices=nhost)


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _strip_flag(flags, name):
    return " ".join(f for f in flags.split() if not f.startswith(name))


@contextlib.contextmanager
def worker_platform(device="cpu", threads=1):
    """Environment for processes spawned inside the block (spawned children inherit os.environ).

    For CPU workers, ``threads`` limits each worker's XLA and BLAS thread pools (None = no limit).
    """
    keys = ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS") + _THREAD_VARS
    old = {k: os.environ.get(k) for k in keys}
    if device and device != "auto":
        os.environ["JAX_PLATFORMS"] = _PLATFORM[device]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if device == "cpu" and threads:
        flags = os.environ.get("XLA_FLAGS", "")
        for name in ("--xla_cpu_multi_thread_eigen", "intra_op_parallelism_threads",
                     "--xla_force_host_platform_device_count"):
            flags = _strip_flag(flags, name)
        multi = "true" if int(threads) > 1 else "false"
        os.environ["XLA_FLAGS"] = (flags + f" --xla_cpu_multi_thread_eigen={multi}"
                                   f" intra_op_parallelism_threads={int(threads)}").strip()
        for k in _THREAD_VARS:
            os.environ[k] = str(int(threads))
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def describe():
    """One-line description of the active JAX backend (imports JAX)."""
    import jax
    devs = jax.devices()
    return (f"JAX {jax.__version__} backend={jax.default_backend()} devices={len(devs)} "
            f"[{devs[0].device_kind}] preallocate={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'default')}")


def add_cli_arguments(parser, worker=False):
    """Add the device options to an argument parser.

    ``--device``, ``--gpu-mem-fraction``, ``--gpu-preallocate``; with `worker` also
    ``--worker-device`` and ``--worker-threads`` for the likelihood pool of the global layer.
    """
    g = parser.add_argument_group("JAX device and memory")
    g.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                   help="JAX platform of the main process (auto = GPU if available)")
    g.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="cap the GPU memory fraction per process (sets XLA_PYTHON_CLIENT_MEM_FRACTION)")
    g.add_argument("--gpu-preallocate", action="store_true",
                   help="re-enable JAX's default GPU preallocation (off by default)")
    if worker:
        g.add_argument("--worker-device", choices=["cpu", "gpu", "auto"], default="cpu",
                       help="JAX platform of the likelihood pool workers (default cpu; with gpu, "
                            "all workers share one GPU without preallocation)")
        g.add_argument("--worker-threads", type=int, default=1,
                       help="threads per CPU pool worker (default 1; npool x threads should not "
                            "exceed the number of physical cores)")
    return parser
