"""NUTS within each mode (NumPyro, JAX-native) and conversion to ArviZ.

The local log density is pure JAX, so it is sampled directly with NumPyro's NUTS through a
potential function, avoiding a PyTensor wrapper. Chains start from the mode's MAP with
Laplace-covariance jitter, and the dense mass matrix is initialised with the Laplace
covariance and then adapted during warm-up.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS


def choose_chain_method(num_chains):
    """CPU: one XLA host device per chain -> 'parallel'. GPU/TPU: vectorise the chains on one
    device ('vectorized'), rather than falling back to running them one after another."""
    backend = jax.default_backend()
    if backend == "cpu":
        return "parallel" if jax.local_device_count() >= num_chains else "sequential"
    if jax.local_device_count() >= num_chains and num_chains > 1:
        return "parallel"
    return "vectorized"


def run_nuts(model, x_map, cov, num_warmup=500, num_samples=500, num_chains=2, seed=0,
             target_accept=0.9, max_tree_depth=7, init_scale=0.3, progress=True):
    """NUTS sampling of one spin state with NumPyro.

    Chains start near the MAP, spread by `init_scale` along the Laplace covariance, which is also
    the initial dense mass matrix.

    Parameters
    ----------
    model : LocalModel
    x_map : ndarray
    cov : ndarray
        Laplace covariance (`laplace_evidence`).
    num_warmup, num_samples, num_chains : int, optional
        ``--warmup``, ``--samples``, ``--chains``.
    seed : int, optional
    target_accept : float, optional
        ``--target-accept``.
    max_tree_depth : int, optional
        ``--tree-depth``.
    init_scale : float, optional
    progress : bool, optional

    Returns
    -------
    dict
        ``x`` (chain, draw, ndim), ``diverging``, ``num_steps``, ``seconds``, ``chain_method``.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(x_map)))
    inits = np.array([x_map + init_scale * L @ rng.standard_normal(len(x_map)) for _ in range(num_chains)])
    pot = lambda x: -model._logdensity(x)
    kernel = NUTS(potential_fn=pot, dense_mass=True, target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, inverse_mass_matrix=jnp.asarray(cov))
    chain_method = choose_chain_method(num_chains)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method=chain_method, progress_bar=progress)
    t0 = time.time()
    init = jnp.asarray(inits if num_chains > 1 else inits[0])     # one chain: no chain axis
    mcmc.run(jax.random.PRNGKey(seed), init_params=init, extra_fields=("diverging", "num_steps"))
    x = np.asarray(mcmc.get_samples(group_by_chain=True))          # (chain, draw, ndim)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    return dict(x=x, diverging=np.asarray(ex["diverging"]), num_steps=np.asarray(ex["num_steps"]),
                seconds=time.time() - t0, chain_method=chain_method)


def derived(model, x_flat, batch=64):
    """Derived quantities for an (n, ndim) array of draws, evaluated in vmapped batches."""
    F = model.F

    def one(x):
        lam, bet, P = model.spin(x)
        a = model.coeffs(x)
        V, c, I = F.mass_properties(a)
        w, U = jnp.linalg.eigh(I)
        ang = jnp.degrees(jnp.arccos(jnp.clip(jnp.abs(U[2, -1]), 0.0, 1.0)))
        req = (3 * V / (4 * jnp.pi)) ** (1 / 3)
        r_mesh = F.radius_mesh(a) / req
        v = r_mesh[:, None] * F.um - c / req
        ext = v.max(0) - v.min(0)
        point, corr, eta, _ = model.loglike_terms(x)
        out = dict(lam=lam, bet=bet, P=P, jitter=jnp.exp(x[3]), sigma_f=jnp.exp(x[4]),
                   ell=jnp.exp(x[5]), inertia_angle=ang, moments=w / w[-1], extents=ext,
                   r_mesh=r_mesh, loglik=point, eta=eta)
        if getattr(model, "fit_cl", False):
            out["c_L"] = model.c_l(x)
        return out

    f = jax.jit(jax.vmap(one))
    outs = [jax.tree_util.tree_map(np.asarray, f(jnp.asarray(x_flat[i:i + batch])))
            for i in range(0, len(x_flat), batch)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def to_inferencedata(res, der, mode_name):
    """ArviZ InferenceData with posterior, pointwise log-likelihood and sample stats."""
    import arviz as az
    nc, nd = res["x"].shape[:2]
    shp = lambda a: a.reshape(nc, nd, *a.shape[1:])
    post = {k: shp(der[k]) for k in ("lam", "bet", "P", "jitter", "sigma_f", "ell", "inertia_angle", "c_L")
            if k in der}
    post["moments"] = shp(der["moments"])
    post["x"] = res["x"]
    data = {"posterior": post,
            "log_likelihood": {"y": shp(der["loglik"])},
            "sample_stats": {"diverging": res["diverging"], "n_steps": res["num_steps"]}}
    try:
        return az.from_dict(data, name=mode_name)
    except TypeError:
        return az.from_dict(data)


def convergence_summary(res, der):
    """R-hat and bulk ESS for the spin parameters, with a NumPy fallback if ArviZ fails."""
    nc, nd = res["x"].shape[:2]
    out = {}
    for k in ("lam", "bet", "P", "jitter", "sigma_f", "ell", "inertia_angle", "c_L"):
        if k not in der:
            continue
        a = der[k].reshape(nc, nd)
        if k == "lam":  # circular: use deviations from the circular mean
            m = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(a)))))
            a = ((a - m + 180) % 360) - 180
        out[k] = dict(rhat=_rhat(a), ess=_ess(a), mean=float(a.mean()), sd=float(a.std()))
    out["divergences"] = int(res["diverging"].sum())
    out["mean_leapfrog_steps"] = float(res["num_steps"].mean())
    return out


def _rhat(a):
    """Rank-normalised split R-hat (Vehtari et al. 2021), computed directly."""
    from scipy.stats import rankdata, norm
    nc, nd = a.shape
    h = nd // 2
    s = np.concatenate([a[:, :h], a[:, h:2 * h]], 0)
    z = norm.ppf((rankdata(s).reshape(s.shape) - 0.375) / (s.size + 0.25))

    def rh(x):
        m, n = x.shape
        B = n * x.mean(1).var(ddof=1)
        W = x.var(1, ddof=1).mean()
        return float(np.sqrt(((n - 1) / n * W + B / n) / W))
    zf = norm.ppf((rankdata(np.abs(s - np.median(s))).reshape(s.shape) - 0.375) / (s.size + 0.25))
    return max(rh(z), rh(zf))


def _ess(a):
    """Bulk ESS via Geyer's initial monotone sequence on the chain-averaged autocorrelation."""
    nc, n = a.shape
    x = a - a.mean(1, keepdims=True)
    acf = np.zeros(n)
    for c in range(nc):
        f = np.fft.rfft(x[c], 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        acf += ac / ac[0]
    acf /= nc
    s, t = 0.0, 1
    while t + 1 < n:
        p = acf[t] + acf[t + 1]
        if p < 0:
            break
        s += p
        t += 2
    return float(nc * n / (1 + 2 * s))
