"""Nested sampling of the spin state with dynesty, checkpoints and the posterior shape ensemble.

The sampled parameters are ``theta = (lambda, beta, P, ln sigma_jit[, ln c_L][, ln tau0])``.
The rotational phase is not sampled: for a free shape it is a gauge freedom (a rotation of the
body about the spin axis), fixed to ``phi0 = 0`` at ``t_ref`` inside the model.

Likelihood calls can run in a pool of spawned worker processes (``npool > 1``); each worker
rebuilds the model from ``model.init_kwargs``, so any model class with that attribute and a
``loglike`` method can be sampled (the convex and the contact-binary models).
"""
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import jax.numpy as jnp
import dynesty
from dynesty import utils as dyfunc

from lcinv import jaxenv
from scipy.special import ndtr, ndtri

PARAM_NAMES = [r"$\lambda$ [deg]", r"$\beta$ [deg]", r"$P$ [h]", r"$\ln\sigma_{\rm jit}$"]
CL_NAME = r"$\ln c_{\rm L}$"
TAU_NAME = r"$\ln\tau_0$"
NDIM = 4  # without the optional ln c_L and ln tau0


def param_names(ndim, fit_cl=None, fit_tau=False):
    """Axis labels of the sampled parameters, for plots.

    Parameters
    ----------
    ndim : int
        Number of sampled parameters.
    fit_cl : bool, optional
        Whether ln c_L is included; default: inferred as "more than 4 parameters" (correct for
        runs without `fit_tau`).
    fit_tau : bool, optional
        Whether ln tau0 is included.

    Returns
    -------
    list of str
    """
    if fit_cl is None:
        fit_cl = ndim - int(fit_tau) > 4
    return PARAM_NAMES + ([CL_NAME] if fit_cl else []) + ([TAU_NAME] if fit_tau else [])


class PriorTransform:
    """Picklable map from the unit cube to `theta`.

    Pole uniform on the sphere (lambda uniform, sin(beta) uniform); period from `period`;
    ln sigma_jit uniform in `ln_jit_range`; optional ln c_L and ln tau0 uniform in their ranges
    (log-uniform c_L and tau0).

    Parameters
    ----------
    period : tuple
        ``("gaussian", P0, sigma, nsig)`` (truncated at +-nsig sigma) or ``("uniform", lo, hi)``,
        in hours.
    ln_jit_range : (float, float), optional
        Range of ln sigma_jit; default jitter 1 to 100 mmag.
    ln_cl_range : (float, float), optional
        Range of ln c_L, or None to not sample it.
    ln_tau_range : (float, float), optional
        Range of ln tau0, or None to not sample it.

    Attributes
    ----------
    ndim : int
        Number of sampled parameters.
    """

    def __init__(self, period, ln_jit_range=(np.log(1e-3), np.log(0.1)), ln_cl_range=None, ln_tau_range=None):
        self.period = period
        self.lo, self.hi = ln_jit_range
        self.cl_range = ln_cl_range
        self.tau_range = ln_tau_range
        self.ndim = NDIM + (1 if ln_cl_range is not None else 0) + (1 if ln_tau_range is not None else 0)

    def P(self, u):
        """Period [h] at unit-cube coordinate `u`."""
        if self.period[0] == "gaussian":
            _, P0, s, nsig = self.period
            a, b = ndtr(-nsig), ndtr(nsig)
            return P0 + s * ndtri(a + u * (b - a))
        _, lo, hi = self.period
        return lo + u * (hi - lo)

    def __call__(self, u):
        """`theta` at unit-cube point `u`."""
        th = [360.0 * u[0],                               # lambda, periodic
              np.degrees(np.arcsin(2.0 * u[1] - 1.0)),    # beta, uniform in sin(beta)
              self.P(u[2]),                               # sidereal period (h)
              self.lo + (self.hi - self.lo) * u[3]]       # ln sigma_jit
        if self.cl_range is not None:                     # ln c_L, log-uniform c_L
            a, b = self.cl_range
            th.append(a + (b - a) * u[4])
        if self.tau_range is not None:                    # ln tau0, log-uniform shape prior scale
            a, b = self.tau_range
            th.append(a + (b - a) * u[len(th)])
        return np.array(th)


# ---------------------------------------------------------------- multiprocessing support
_MODEL = None


def _init_worker(kwargs):
    global _MODEL
    import importlib
    kwargs = dict(kwargs)
    path = kwargs.pop("model_class", "lcinv.convex.ConvexInversion")
    mod, cls = path.rsplit(".", 1)
    _MODEL = getattr(importlib.import_module(mod), cls)(**kwargs)
    if hasattr(_MODEL, "warmup_theta"):
        _MODEL.loglike(_MODEL.warmup_theta())                   # compile in the worker
    else:
        _MODEL.loglike(np.r_[0.0, 0.0, 3.0, -4.0, [np.log(0.1)] * (_MODEL.ntheta - 4)])


def _loglike_worker(theta):
    return _MODEL.loglike(theta)


def _check_pool(npool, worker_threads):
    ncpu = os.cpu_count() or 1
    if npool * max(1, worker_threads or 1) > ncpu:
        print(f"[sampler] warning: npool={npool} x worker_threads={worker_threads} exceeds the "
              f"{ncpu} logical CPUs; workers will compete for cores")


def finalize_from_checkpoint(checkpoint_file, model=None, npool=1, extra_calls=0, dlogz=0.5,
                             worker_device="cpu", worker_threads=1):
    """Restore a dynesty checkpoint, optionally continue, and return the results.

    This is ``--resume``. A checkpoint of a run that ended normally or at ``--maxcall`` already
    contains the final live points; they are removed before continuing and added back at the end.

    Parameters
    ----------
    checkpoint_file : str
        ``<outdir>/dynesty.save``.
    model : object, optional
        The model; needed for parallel continuation (`npool` > 1).
    npool : int, optional
        Worker processes.
    extra_calls : int, optional
        Continue for up to this many likelihood calls (``--extra-calls``); 0 only finalises.
    dlogz : float, optional
        Stopping criterion while continuing.
    worker_device : {'cpu', 'gpu', 'auto'}, optional
        JAX platform of the workers.
    worker_threads : int, optional
        CPU threads per worker.

    Returns
    -------
    dynesty.results.Results
    """
    t0 = time.time()

    def _go(pool):
        s = dynesty.NestedSampler.restore(checkpoint_file, pool=pool)
        if getattr(s, "added_live", False):
            # A run that ended normally (or at --maxcall) saved its checkpoint after adding the
            # final live points to the samples. Take them out again before continuing; they are
            # added back below. Without this dynesty raises "The remaining live points have
            # already been added to the list of samples!".
            s._remove_live_points()
        if extra_calls > 0:
            s.run_nested(dlogz=dlogz, maxcall=int(s.ncall) + int(extra_calls), resume=True,
                         checkpoint_file=checkpoint_file, add_live=False)
        s.add_final_live()
        return s.results

    if npool > 1 and model is not None:
        _check_pool(npool, worker_threads)
        ctx = mp.get_context("spawn")
        with jaxenv.worker_platform(worker_device, worker_threads), \
                ctx.Pool(npool, initializer=_init_worker, initargs=(model.init_kwargs,)) as pool:
            res = _go(pool)
    else:
        res = _go(None)
    print(f"finalised from checkpoint: {res.niter} iterations, {int(np.sum(res.ncall))} calls, "
          f"{time.time() - t0:.0f} s, ln Z = {res.logz[-1]:.2f} +- {res.logzerr[-1]:.2f}")
    return res


def run_nested(model, prior_transform, nlive=300, dlogz=0.1, sample="rslice", seed=42,
               maxcall=None, npool=1, print_progress=True, checkpoint_file=None, worker_device="cpu",
               worker_threads=1, bootstrap=0, slices=None, resume=False):
    """Static nested sampling with dynesty.

    ``sample='rslice'`` (default) proposes by random slice sampling inside the live-point
    ellipsoids. Its cost per iteration (about 5 calls x slices) does not depend on how well the
    ellipsoids fit, unlike ``'unif'`` rejection sampling, which collapses once the posterior
    becomes a thin curved pole ridge (bootstrap warnings, more than 100 calls per iteration).

    Parameters
    ----------
    model : object
        Model with ``loglike(theta)`` and ``init_kwargs``.
    prior_transform : PriorTransform
    nlive : int, optional
        Live points.
    dlogz : float, optional
        Stop when the evidence left in the live points is below this.
    sample : {'rslice', 'rwalk', 'slice', 'unif'}, optional
        dynesty proposal.
    seed : int, optional
    maxcall : int, optional
        Stop after this many likelihood calls.
    npool : int, optional
        Worker processes (spawned); 1 evaluates in the main process.
    print_progress : bool, optional
    checkpoint_file : str, optional
        Save the sampler state here regularly (for ``--resume``).
    worker_device : {'cpu', 'gpu', 'auto'}, optional
        JAX platform of the workers (``--worker-device``).
    worker_threads : int, optional
        CPU threads per worker (``--worker-threads``).
    bootstrap : int, optional
        dynesty bootstrap enlargement of the bounds; only matters for ``'unif'``.
    slices : int, optional
        Slices per proposal for (r)slice.
    resume : bool, optional
        Continue a checkpoint that has not finished.

    Returns
    -------
    dynesty.results.Results
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    kw = dict(ndim=getattr(prior_transform, "ndim", NDIM), nlive=nlive, bound="multi", sample=sample,
              periodic=list(getattr(prior_transform, "periodic", [0])), rstate=rng,
              bootstrap=bootstrap)
    if slices is not None and sample in ("slice", "rslice"):
        kw["slices"] = int(slices)
    restore = bool(resume and checkpoint_file and os.path.exists(checkpoint_file))

    def _go(pool):
        if restore:
            s = dynesty.NestedSampler.restore(checkpoint_file, pool=pool)
            print(f"resuming from {checkpoint_file}: {s.it} iterations, {int(s.ncall)} calls done")
            s.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=print_progress,
                         checkpoint_file=checkpoint_file, resume=True)
        else:
            extra = dict(pool=pool, queue_size=npool) if pool is not None else {}
            s = dynesty.NestedSampler(_loglike_worker, prior_transform, **extra, **kw)
            s.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=print_progress,
                         checkpoint_file=checkpoint_file)
        return s.results

    if npool > 1:
        _check_pool(npool, worker_threads)
        ctx = mp.get_context("spawn")
        with jaxenv.worker_platform(worker_device, worker_threads), \
                ctx.Pool(npool, initializer=_init_worker, initargs=(model.init_kwargs,)) as pool:
            res = _go(pool)
    else:
        # serial: the module-level wrapper (not the jitted bound method) keeps the sampler
        # picklable, so checkpoints work here too
        global _MODEL
        _MODEL = model
        res = _go(None)
    ncall = int(np.sum(res.ncall))
    print(f"nested sampling: {res.niter} iterations, {ncall} likelihood calls "
          f"(efficiency {res.niter / ncall:.1%}), {time.time() - t0:.0f} s, "
          f"ln Z = {res.logz[-1]:.2f} +- {res.logzerr[-1]:.2f}")
    return res


def equal_weight_samples(res, seed=3):
    """Equally weighted posterior samples from nested-sampling results, longitude wrapped to [0, 360).

    Parameters
    ----------
    res : dynesty.results.Results
    seed : int, optional

    Returns
    -------
    ndarray
        Shape (n, ndim).
    """
    w = np.exp(res.logwt - res.logz[-1])
    post = dyfunc.resample_equal(res.samples, w / w.sum(), rstate=np.random.default_rng(seed))
    post[:, 0] %= 360.0  # periodic longitude may be stored unwrapped
    return post


def shape_ensemble(model, samples, n=200, seed=4):
    """Posterior shape ensemble: the shape behind each of `n` posterior samples.

    For each spin state the optimal coefficients are recomputed and one shape is drawn from the
    Laplace approximation ``c ~ N(c_hat, H^-1)``; directions without positive curvature are drawn
    from the prior.

    Parameters
    ----------
    model : ConvexInversion
    samples : ndarray
        Equal-weight posterior samples.
    n : int, optional
        Number of samples to use (``--n-post``).
    seed : int, optional

    Returns
    -------
    dict of ndarray
        ``theta``, ``c_hat``, ``c_draw``, ``ab``, ``bc``, ``beta``, ``logL``, ``chi2``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    out = dict(theta=[], c_hat=[], c_draw=[], ab=[], bc=[], beta=[], logL=[], chi2=[])
    for i in idx:
        th = samples[i]
        o = model.loglike_jit(jnp.asarray(th, float))
        H = np.asarray(o["H"])
        ev, V = np.linalg.eigh(0.5 * (H + H.T))
        # floor: the prior precision of the loosest coefficient (directions the optimiser left
        # with non-positive curvature are drawn from the prior, not with an exploding width)
        floor = 1e-8
        if hasattr(model, "tau_of"):
            floor = max(floor, 1.0 / float(np.max(np.asarray(model.tau_of(jnp.asarray(th, float))))) ** 2)
        ev = np.clip(ev, floor, None)
        c = np.asarray(o["c"]) + V @ (rng.standard_normal(len(ev)) / np.sqrt(ev))
        ab, bc = model.axis_ratios(model.areas(jnp.asarray(c)))
        out["theta"].append(th)
        out["c_hat"].append(np.asarray(o["c"]))
        out["c_draw"].append(c)
        out["ab"].append(float(ab))
        out["bc"].append(float(bc))
        out["beta"].append(np.asarray(o["beta"]))
        out["logL"].append(float(o["logL"]))
        out["chi2"].append(float(o["chi2"]))
    return {k: np.array(v) for k, v in out.items()}


def save(path, **kw):
    """Pickle keyword arguments as a dict to `path` (``nested_results.pkl``)."""
    with open(path, "wb") as f:
        pickle.dump(kw, f)


def load(path):
    """Read a pickle written by `save`.

    Parameters
    ----------
    path : str
        E.g. ``"<run>/nested_results.pkl"``.

    Returns
    -------
    dict
        For a global run: ``samples``, ``logwt``, ``logl``, ``logz``, ``logzerr`` (raw nested
        samples), ``post`` (equal-weight posterior), ``args`` (the run's options) and ``ensemble``
        (from `shape_ensemble`).
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def sky_area_fraction(samples, mass=(0.68, 0.95), ncell=3000):
    """Fraction of the sky covered by the highest-density pole regions.

    Parameters
    ----------
    samples : ndarray
        Posterior samples; columns 0 and 1 are lambda and beta [deg].
    mass : sequence of float, optional
        Probability masses.
    ncell : int, optional
        Number of sky cells (Fibonacci sphere).

    Returns
    -------
    dict
        ``{"68%": f68, "95%": f95}``.
    """
    from lcinv.convex import fibonacci_sphere
    cells = fibonacci_sphere(ncell)
    l, b = np.radians(samples[:, 0]), np.radians(samples[:, 1])
    v = np.c_[np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)]
    idx = np.argmax(v @ cells.T, axis=1)
    cnt = np.sort(np.bincount(idx, minlength=ncell))[::-1]
    cum = np.cumsum(cnt) / cnt.sum()
    return {f"{int(m * 100)}%": float((np.searchsorted(cum, m) + 1) / ncell) for m in mass}
