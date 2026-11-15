"""Post-hoc physical checks of the local-layer posterior.

convexity     : exact test on the triangulated surface. For every facet plane, the largest
                signed distance of any vertex in front of it (0 for a convex polyhedron),
                in units of the volume-equivalent radius; plus the fraction of facets with
                a violation above `tol`.
shadowing     : the forward model sums Lommel-Seeliger + Lambert over lit and visible facets,
                which is exact only for convex bodies. For selected draws and epochs we
                ray-trace occlusion of the illumination and viewing rays from each
                contributing facet against all triangles (Moller-Trumbore, JAX-vectorised)
                and report the magnitude error 2.5 log10(L_noshadow / L_shadowed).
sensitivity   : MAP + Laplace evidence of every mode for several prior settings.
"""
import numpy as np
import jax
import jax.numpy as jnp



def convexity(model, x, tol=1e-3):
    """How far a shape departs from its convex hull.

    Parameters
    ----------
    model : LocalModel
    x : array_like
        Parameter vector (e.g. a posterior draw).
    tol : float, optional
        Protrusion above which a facet counts as concave.

    Returns
    -------
    dict
        ``max_protrusion`` (largest distance of a vertex beyond a facet plane, in units of the
        volume-equivalent radius; 0 = convex), ``frac_concave`` and ``area_frac_concave``.
    """
    F = model.F
    a = model.coeffs(jnp.asarray(x))
    v, area, nrm = [np.asarray(q) for q in F.mesh(a)]
    req = (3 * float(F.mass_properties(a)[0]) / (4 * np.pi)) ** (1 / 3)
    p0 = v[np.asarray(F.tri)[:, 0]]
    worst = np.empty(len(nrm))
    for i0 in range(0, len(nrm), 256):
        sl = slice(i0, i0 + 256)
        dist = np.einsum("fk,vk->fv", nrm[sl], v) - np.sum(nrm[sl] * p0[sl], 1)[:, None]
        worst[sl] = dist.max(1)
    worst /= req
    return dict(max_protrusion=float(worst.max()), frac_concave=float(np.mean(worst > tol)),
                area_frac_concave=float(area[worst > tol].sum() / area.sum()))


def _ray_hits(orig, dirn, p0, e1, e2, eps=1e-6):
    """Any-hit test of rays (R,3) against triangles (T,3 each) -> (R,) bool."""
    h = jnp.cross(dirn[:, None, :], e2[None, :, :])
    a = jnp.sum(e1[None] * h, -1)
    f = 1.0 / jnp.where(jnp.abs(a) < 1e-12, 1e-12, a)
    s = orig[:, None, :] - p0[None]
    u = f * jnp.sum(s * h, -1)
    q = jnp.cross(s, e1[None])
    vv = f * jnp.sum(dirn[:, None, :] * q, -1)
    t = f * jnp.sum(e2[None] * q, -1)
    hit = (jnp.abs(a) > 1e-12) & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > eps)
    return jnp.any(hit, axis=1)


def shadowing_error(model, x, epochs=None, n_epochs=40):
    """Magnitude error from ignoring self-shadowing/occlusion for one draw, per epoch."""
    F = model.F
    lam, bet, P = model.spin(jnp.asarray(x))
    a = model.coeffs(jnp.asarray(x))
    v, area, nrm = F.mesh(a)
    tri = F.tri
    p0, p1, p2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    cen = (p0 + p1 + p2) / 3.0
    scale = jnp.mean(jnp.linalg.norm(cen, axis=1))
    n = len(model.t)
    epochs = np.linspace(0, n - 1, min(n_epochs, n)).astype(int) if epochs is None else epochs
    xj = jnp.asarray(x)
    es = model.body_vectors(xj, model.e_sun[epochs], model.t[epochs])
    eo = model.body_vectors(xj, model.e_obs[epochs], model.t[epochs])
    hits = jax.jit(_ray_hits)
    out = []
    for k in range(len(epochs)):
        mu0 = jnp.clip(nrm @ es[k], 0, None)
        mu = jnp.clip(nrm @ eo[k], 0, None)
        S = mu * mu0 * (1.0 / (mu + mu0 + 1e-12) + float(model.c_l(xj)))
        act = np.where(np.asarray(S) > 0)[0]
        orig = cen[act] + 1e-6 * scale * nrm[act]
        sh = hits(orig, jnp.broadcast_to(es[k], orig.shape), p0, e1, e2)
        oc = hits(orig, jnp.broadcast_to(eo[k], orig.shape), p0, e1, e2)
        keep = ~(np.asarray(sh) | np.asarray(oc))
        Lfull = float(jnp.sum(area[act] * S[act]))
        Lsh = float(jnp.sum(area[act][keep] * S[act][keep]))
        out.append(2.5 * np.log10(Lfull / Lsh))
    return np.array(out)


def prior_sensitivity(make_model, modes, settings, map_fn, laplace_fn, starts=4):
    """For each prior setting: MAP pole and Laplace evidence per mode -> weights."""
    rows = []
    for name, kw in settings:
        res = []
        for m in modes:
            M = make_model(m, **kw)
            x, lp, info = map_fn(M, M.initial_points(starts))
            L = laplace_fn(M, x)
            lam, bet, P = [float(q) for q in M.spin(jnp.asarray(x))]
            res.append(dict(mode=m["name"], lam=lam, bet=bet, P=P, lnZ=L["lnZ"],
                            n_nonpositive=L["n_nonpositive"], jitter=float(np.exp(x[3]))))
        lz = np.array([r["lnZ"] for r in res])
        w = np.exp(lz - lz.max())
        w /= w.sum()
        for r, wi in zip(res, w):
            r["weight"] = float(wi)
        rows.append(dict(setting=name, kwargs=kw, modes=res))
    return rows
