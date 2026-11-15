"""Phase-folded (0-4 pi) lightcurve figures for the convex and hybrid models.

Both use lcinv.plots.folded: data are shown as model_ref(phase) + residual, where model_ref
is the model at the fixed reference geometry (Sun/observer directions and phase angle of the
median epoch, r band) and the residual is data minus the model at the point's own geometry,
band offset and phase term. Rotational phase 0 is at the model's t_ref (+ whole rotations).
"""
import numpy as np

from lcinv import plots


def _ref_setup(D, t_ref, P, nb):
    _, t_near, es, eo, al = plots.reference_geometry(D)
    tg, sg, t0 = plots.phase_grid(t_near, t_ref, P)
    n = len(tg)
    r = plots.ref_band(D["bands"])
    Xr = np.r_[np.eye(nb)[r], al]
    return dict(t_near=t_near, tg=tg, sg=sg, t0=t0, es=np.repeat(es[None], n, 0),
                eo=np.repeat(eo[None], n, 0), r=np.full(n, r), al=al, alpha=np.full(n, al), Xr=Xr)


def _texp(D):
    T = np.asarray(D.get("exptime", np.zeros(len(D["t"]))), float)
    return (T, float(np.median(T))) if np.any(T > 0) else (None, None)


def convex(path, D, M, theta_ml, o_ml, ens, pick, title):
    """Global convex model: max-likelihood fit (reference) + posterior draws."""
    texp, tm = _texp(D)
    P = float(theta_ml[2])
    beta_ml = np.asarray(o_ml["beta"])
    resid = D["mag"] - M.predict(theta_ml, D["t"], D["e_sun"], D["e_obs"], D["band_idx"], D["alpha"],
                                 o_ml["c"], beta_ml, exptime=texp)
    g = _ref_setup(D, M.t_ref, P, M.nb)
    pred = lambda th, t, c, b: M.predict(th, t, g["es"], g["eo"], g["r"], g["alpha"], c, b, exptime=tm) \
        - g["Xr"] @ np.asarray(b)
    curves = [pred(theta_ml, g["tg"], o_ml["c"], beta_ml)]
    for j in pick:   # every draw evaluated at the same epochs (the reference rotation)
        curves.append(pred(ens["theta"][j], g["tg"], ens["c_hat"][j], ens["beta"][j]))
    plots.folded(path, D, P, g["t0"], resid, g["sg"], {"posterior draws": curves}, title, alpha_ref=g["al"])


def hybrid(path, D, mode_runs, title, n_curves=25, seed=3):
    """Hybrid model: best draw of the highest-weight mode (reference) + draws of every mode."""
    import jax.numpy as jnp
    from lcinv.hybrid.report import MODE_COL
    main = max(mode_runs, key=lambda m: m["weight"])
    M = main["model"]
    xs = main["res"]["x"].reshape(-1, M.ndim)
    jb = int(np.argmax(main["der"]["loglik"].sum(axis=1)))
    xb, eta_b = xs[jb], np.asarray(main["der"]["eta"][jb])
    X = np.asarray(M.X)
    resid = np.asarray(M.y) - (-2.5 * np.log10(np.asarray(M.brightness(jnp.asarray(xb)))) + X @ eta_b)
    texp, tm = _texp(D)
    P = float(main["der"]["P"][jb])
    nb = X.shape[1] - 1
    g = _ref_setup(D, M.t_ref, P, nb)

    def curve(m, x, eta, Pj):   # every draw evaluated at the same epochs (the reference rotation)
        L = np.asarray(m.brightness(jnp.asarray(x), t=g["tg"], e_sun=g["es"], e_obs=g["eo"], exptime=tm))
        return -2.5 * np.log10(L) + g["Xr"] @ (np.asarray(eta) - eta_b)   # own offsets, relative to the reference

    rng = np.random.default_rng(seed)
    groups, cols = {}, []
    groups[f"{main['name']} (reference)"] = [curve(M, xb, eta_b, P)]
    cols.append(MODE_COL[mode_runs.index(main) % len(MODE_COL)])
    for i, mr in enumerate(mode_runs):
        xm = mr["res"]["x"].reshape(-1, mr["model"].ndim)
        pick = rng.choice(len(xm), size=min(n_curves, len(xm)), replace=False)
        lab = f"{mr['name']} (weight {mr['weight']:.2f})"
        groups.setdefault(lab, [])
        groups[lab] += [curve(mr["model"], xm[j], mr["der"]["eta"][j], mr["der"]["P"][j]) for j in pick]
        cols.append(MODE_COL[i % len(MODE_COL)])
    plots.folded(path, D, P, g["t0"], resid, g["sg"], groups, title, curve_cols=cols, alpha_ref=g["al"])
