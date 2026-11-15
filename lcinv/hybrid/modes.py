"""Spin states from the global posterior, the MAP within each, and its Laplace evidence."""
import numpy as np
import jax
import jax.numpy as jnp
import optax


def unitvec(lam, bet):
    """Unit vectors of longitudes and latitudes in degrees, shape (N, 3)."""
    l, b = np.radians(lam), np.radians(bet)
    return np.c_[np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)]


def find_modes(post_lam, post_bet, r_core=20.0, r_member=45.0, min_frac=0.05, max_modes=4):
    """Group global pole samples into spin states (greedy clustering on the sphere).

    Each step seeds a cluster at the densest sample (most neighbours within `r_core`), takes the
    samples within `r_member` of its centre, and removes them.

    Parameters
    ----------
    post_lam, post_bet : array_like
        Posterior pole samples [deg].
    r_core, r_member : float, optional
        Core and membership radii [deg].
    min_frac : float, optional
        Smallest posterior fraction kept as a spin state.
    max_modes : int, optional

    Returns
    -------
    list of dict
        ``center`` (unit vector), ``lam``, ``bet``, ``frac_global``, ``n``, ``weight_global``.
    """
    v = unitvec(np.asarray(post_lam), np.asarray(post_bet))
    left = np.ones(len(v), bool)
    cos_core, cos_mem = np.cos(np.radians(r_core)), np.cos(np.radians(r_member))
    modes = []
    while left.sum() > min_frac * len(v) and len(modes) < max_modes:
        idx = np.where(left)[0]
        dots = v[idx] @ v[idx].T
        seed = idx[np.argmax((dots > cos_core).sum(1))]
        core = idx[(v[idx] @ v[seed]) > cos_core]
        c = v[core].mean(0)
        c /= np.linalg.norm(c)
        members = idx[(v[idx] @ c) > cos_mem]
        frac = len(members) / len(v)
        if frac < min_frac:
            break
        c = v[members].mean(0)
        c /= np.linalg.norm(c)
        modes.append(dict(center=c, lam=float(np.degrees(np.arctan2(c[1], c[0])) % 360),
                          bet=float(np.degrees(np.arcsin(c[2]))), frac_global=float(frac),
                          n=int(len(members))))
        left[members] = False
    tot = sum(m["frac_global"] for m in modes)
    for m in modes:
        m["weight_global"] = m["frac_global"] / tot
    return modes


def find_map(model, starts, n_iter=2000):
    """Multi-start L-BFGS maximisation of the log density.

    Parameters
    ----------
    model : LocalModel
    starts : sequence of array_like
        Starting points (`LocalModel.initial_points`).
    n_iter : int, optional

    Returns
    -------
    x_map : ndarray
    logp : float
    info : dict
        ``start_logp`` per start and ``grad_norm`` at the MAP.
    """
    f = lambda x: -model._logdensity(x)
    opt = optax.lbfgs()
    vg = optax.value_and_grad_from_state(f)

    @jax.jit
    def run(x0):
        def step(carry, _):
            x, st = carry
            v, g = vg(x, state=st)
            upd, st = opt.update(g, st, x, value=v, grad=g, value_fn=f)
            return (optax.apply_updates(x, upd), st), v
        (x, _), _ = jax.lax.scan(step, (x0, opt.init(x0)), None, length=n_iter)
        return x, f(x)

    best = None
    vals = []
    for x0 in starts:
        x, v = run(jnp.asarray(x0, float))
        v = float(v)
        vals.append(-v)
        if np.isfinite(v) and (best is None or v < best[1]):
            best = (np.asarray(x), v)
    gnorm = float(np.linalg.norm(np.asarray(model.grad(jnp.asarray(best[0])))))
    return best[0], -best[1], dict(start_logp=vals, grad_norm=gnorm)


def laplace_evidence(model, x_map):
    """Laplace approximation of the evidence of a spin state at its MAP.

    Returns
    -------
    dict
        ``lnZ``, ``logp_map``, ``n_nonpositive`` (curvatures <= 0: the optimum is not converged
        and lnZ is unreliable), ``cov`` (inverse Hessian, the NUTS mass matrix) and ``min_eig``.
    """
    H = -np.asarray(jax.hessian(model._logdensity)(jnp.asarray(x_map)))
    H = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    n_bad = int(np.sum(ev <= 0))
    # a non-positive curvature means the optimum is not converged; the clip only keeps the
    # numbers finite and the result is flagged (n_nonpositive > 0) as unreliable
    evc = np.clip(ev, 1e-3, None)
    lp = float(model.logdensity(jnp.asarray(x_map)))
    lnZ = lp + 0.5 * len(x_map) * np.log(2 * np.pi) - 0.5 * np.sum(np.log(evc))
    cov = (V / evc) @ V.T
    return dict(lnZ=float(lnZ), logp_map=lp, n_nonpositive=n_bad, cov=cov, min_eig=float(ev[0]))
