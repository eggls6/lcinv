"""Rebuild the data and models of finished runs from their results folders, without resampling.

Every post-processing script uses these, so a results folder is self-contained::

    D, M, s, R = global_run(outdir)            # convex model; keywords override settings
    D, M, s, R = binary_run(outdir)
    D, grid, states, s = hybrid_run(outdir)    # states: [dict(k, name, model, z, info)]

``D`` is the `lcinv.setup_data.prepare` dictionary (with the run's clipping applied), ``M`` the
model, ``s`` the parsed ``summary.json`` and ``R`` the ``nested_results.pkl`` contents.
"""
import json
import os

import numpy as np


def _prepare(a):
    from lcinv import ades
    from lcinv.setup_data import prepare
    get = a.get
    return prepare(get("lc"), get("designation"), location=get("location", "X05"), cache_dir=get("cache_dir", "data"),
                   drop_bands=tuple(b for b in str(get("drop_bands", "u") or "").split(",") if b),
                   ades_opts=ades.opts_from_args(a), observer=get("observer", "station"), **ades.timing_from_args(a))


def run_kind(outdir):
    """Kind of a results folder: ``'convex'``, ``'binary'``, ``'hybrid'`` or None."""
    from lcinv.hybrid import naming
    sp = os.path.join(outdir, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        if s.get("model") == "contact binary":
            return "binary"
        if os.path.exists(os.path.join(outdir, "nested_results.pkl")):
            return "convex"
    if os.path.exists(os.path.join(outdir, "summary_hybrid.json")) and naming.has_hybrid(outdir):
        return "hybrid"
    return None


def global_run(outdir, **override):
    """Data and convex model of a ``run_inversion.py`` folder.

    Parameters
    ----------
    outdir : str
    **override
        Any `lcinv.convex.ConvexInversion` keyword (``lmax``, ``tau0``, ``nfacets``, ``cohesion``,
        ...), e.g. for the complexity scan or the facet-resolution check.

    Returns
    -------
    D : dict
    M : ConvexInversion
    s : dict
        ``summary.json``.
    R : dict
        ``nested_results.pkl``.
    """
    from lcinv import sampler
    from lcinv.convex import ConvexInversion
    from lcinv.setup_data import drop_epochs
    s = json.load(open(os.path.join(outdir, "summary.json")))
    R = sampler.load(os.path.join(outdir, "nested_results.pkl"))
    a = R["args"]
    D = drop_epochs(_prepare(a), s.get("clipped_epochs", []))
    pr = s["priors"]["period"]
    P0 = pr[1] if pr[0] == "gaussian" else 0.5 * (pr[1] + pr[2])
    from lcinv import cohesion
    kw = dict(lmax=a["lmax"], nfacets=a["nfacets"], n_inner=a["n_inner"], n_starts=a["n_starts"],
              n_stage1=a["n_stage1"], tau0=a["tau0"],
              ab_prior=tuple(s["priors"]["ab"]) if s["priors"].get("ab") else None,
              bc_prior=tuple(s["priors"]["bc"]) if s["priors"].get("bc") else None,
              exptime=D["exptime"], smear_nodes=a.get("smear_nodes"), smear_P_h=P0,
              c_lambert=a.get("c_lambert", 0.1), fit_cl=a.get("fit_cl", False),
              cohesion=cohesion.spec_from_args(a) if a.get("cohesion_max") is not None else None,
              fit_tau=a.get("fit_tau", False))
    kw.update(override)
    M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"], **kw)
    return D, M, s, R


def binary_run(outdir):
    """Data and contact-binary model of a ``run_binary.py`` folder; returns ``(D, M, s, R)`` as
    `global_run`.
    """
    import run_binary
    from lcinv import sampler
    R = sampler.load(os.path.join(outdir, "nested_results.pkl"))
    _, a = run_binary.parse(["--outdir", outdir])
    for k, v in R["args"].items():
        setattr(a, k, v)
    gs = json.load(open(os.path.join(a.global_dir, "summary.json"))) if a.global_dir else None
    D, _ = run_binary.load_data(a, gs)
    M = run_binary.build_model(a, D)
    s = json.load(open(os.path.join(outdir, "summary.json")))
    return D, M, s, R


def hybrid_run(outdir, lmax=None):
    """Data, grid and per-spin-state local models of a ``run_hybrid.py`` folder.

    Parameters
    ----------
    outdir : str
    lmax : int, optional
        Rebuild the radius field with another degree (complexity scan).

    Returns
    -------
    D : dict
    grid : lcinv.hybrid.quadrature.SphereGrid
    states : list of dict
        Per spin state: ``k``, ``name``, ``model`` (`lcinv.hybrid.model.LocalModel`), the draws
        ``z`` and ``info`` (its summary record).
    s : dict
        ``summary_hybrid.json``.
    """
    from lcinv.setup_data import drop_epochs
    from lcinv.hybrid.quadrature import make_grid
    from lcinv.hybrid.shape import RadiusField
    from lcinv.hybrid.model import LocalModel, lonlat_to_vec
    from lcinv.hybrid import naming
    from lcinv import cohesion
    s = json.load(open(os.path.join(outdir, "summary_hybrid.json")))
    st = s["settings"]
    D = _prepare(st)
    gdir = st.get("global_dir")
    gs = json.load(open(os.path.join(gdir, "summary.json"))) if gdir and os.path.exists(
        os.path.join(gdir, "summary.json")) else None
    if gs and gs.get("clipped_epochs"):
        D = drop_epochs(D, gs["clipped_epochs"])
    L = int(lmax or st["lmax"])
    grid = make_grid(L, nlat=max(st["nlat"], L + 3), nlon=2 * max(st["nlat"], L + 3))
    P0, sP = s["period_prior"]["P0"], s["period_prior"]["sigma"]
    coh = cohesion.spec_from_args(st) if st.get("cohesion_max") is not None else None
    states = []
    for k, f in naming.posterior_files(outdir):
        info = naming.state_info(s, k)
        lam, bet = info.get("global_center", [None, None])
        z = np.load(f)
        if lam is None:
            lam = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(z["lam"])))))) % 360
            bet = float(np.median(z["bet"]))
        F = RadiusField(grid, nu=st.get("nu", 2.0), ln_sigma_prior=(np.log(st.get("sigma_f", 0.15)), 0.5),
                        ln_ell_prior=(np.log(st.get("ell", 3.0)), 0.5),
                        inertia_offdiag=st.get("inertia_offdiag", 0.005), inertia_order=st.get("inertia_order", 0.005),
                        convexity_scale=st.get("convexity_scale", 0.02) if st.get("convexity_scale", 0.02) > 0 else None)
        cl = st.get("cl_prior", [0.3, 2.0])
        M = LocalModel(D, F, lonlat_to_vec(lam, bet), P0, sP, frame=st.get("frame") or "euler",
                       smear_nodes=st.get("smear_nodes"), c_lambert=st.get("c_lambert", 0.1),
                       fit_cl=st.get("fit_cl", False), ln_cl_prior=(np.log(cl[0]), cl[1]), cohesion=coh)
        states.append(dict(k=k, name=naming.name(k), model=M, z=z, info=info, path=f))
    return D, grid, states, s


def best_hybrid_point(state, refine=True, n_iter=600):
    """MAP of a spin state if stored, else the draw with the highest log density (optionally
    refined).
    """
    z, M = state["z"], state["model"]
    if "x_map" in z.files and z["x_map"].shape[-1] == M.ndim:
        return np.asarray(z["x_map"], float)
    x = z["x"].reshape(-1, z["x"].shape[-1])
    if x.shape[1] != M.ndim:
        raise ValueError("stored draws do not match the rebuilt model")
    idx = np.argsort(-z["loglik"].sum(1))[:20] if "loglik" in z.files else np.arange(min(20, len(x)))
    lp = [float(M.logdensity(xi)) for xi in x[idx]]
    x0 = x[idx[int(np.argmax(lp))]]
    if not refine:
        return x0
    from lcinv.hybrid import modes as md
    xm, _, _ = md.find_map(M, [x0], n_iter=n_iter)     # runs before x_map was stored
    return np.asarray(xm)
