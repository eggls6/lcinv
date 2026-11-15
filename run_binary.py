#!/usr/bin/env python
"""Contact-binary inversion: two ellipsoidal lobes in contact, fitted by nested sampling.

  python run_binary.py --global-dir results/566130_X05_cl --outdir results/566130_X05_binary

With --global-dir the data selection (file, stations, observer, exposure time, clipped
outliers), the period prior and c_L settings are taken from that run_inversion.py run, so
the evidences of the two models are directly comparable (summary.json: evidence_comparison).
Without it give the lightcurve and the period:

  python run_binary.py data/566130_ades.csv --stations X05 --observer local --exptime 30 \\
      --period 0.1779775 --period-sigma 2e-6 --outdir results/566130_binary

Outputs (like run_inversion.py): summary.json, nested_results.pkl, shapes/*.obj (watertight
union meshes, unit volume-equivalent radius, or km with --diameter-km), fig_pole_posterior.png,
fig_corner.png, fig_lightcurves.png, fig_folded_lightcurve.png, shape_viewer.html.
check_convergence.py works on the folder; --replot rebuilds the products of an existing
--outdir without sampling.
"""
import sys

from lcinv import jaxenv

jaxenv.early_setup(sys.argv, host_devices_flag=None)  # before anything imports JAX

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from lcinv import ades as _ades, diagnostics, ephem, plots, sampler, setup_data  # noqa: E402
from lcinv import minkowski as mk  # noqa: E402
from lcinv.binary import ContactBinary, BinaryPrior, param_names  # noqa: E402

DATA_KEYS = ("lc", "designation", "location", "cache_dir", "drop_bands", "stations", "exclude_stations",
             "ades_bands", "ades_default_rms", "ades_sparse_floor", "ades_dense_rms", "ades_max_rms",
             "ades_min_group", "observer", "exptime", "obstime", "smear_nodes")


def parse(argv=None):
    """Parse the options; returns the parser and the namespace."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lc", nargs="?", default=None, help="lightcurve / ADES file (optional with --global-dir)")
    p.add_argument("--global-dir", default=None, help="run_inversion.py output: data, period prior, c_L")
    p.add_argument("--designation", default=None,
                   help="target designation (default: from --global-dir or an ADES file)")
    p.add_argument("--location", default="X05",
                   help="MPC observatory code for plain lightcurves")
    p.add_argument("--cache-dir", default="data",
                   help="folder of cached JPL Horizons tables")
    p.add_argument("--drop-bands", default="u",
                   help="comma-separated bands to remove")
    p.add_argument("--outdir", default="results/binary",
                   help="results folder")
    p.add_argument("--period", type=float, default=None, help="period prior centre [h]")
    p.add_argument("--period-sigma", type=float, default=None, help="period prior sigma [h]")
    p.add_argument("--period-nsig", type=float, default=5.0,
                   help="truncation of the Gaussian period prior, in sigma")
    g = p.add_argument_group("contact-binary shape priors (uniform)")
    g.add_argument("--q-range", type=float, nargs=2, default=[0.3, 1.0], help="lobe size ratio")
    g.add_argument("--b-range", type=float, nargs=2, default=[0.5, 1.0], help="lobe b/a")
    g.add_argument("--cb-range", type=float, nargs=2, default=[0.5, 1.0], help="lobe c/b")
    g.add_argument("--s-range", type=float, nargs=2, default=[0.70, 0.98],
                   help="centre distance / (a1 + a2); 1 = touching, smaller = more overlap")
    p.add_argument("--nfacets", type=int, default=800, help="facets per lobe")
    p.add_argument("--soft", type=float, default=0.02, help="soft-edge width of occlusion and neck")
    p.add_argument("--c-lambert", type=float, default=None, help="fixed c_L (default: from --global-dir or 0.1)")
    p.add_argument("--fit-cl", action="store_true", default=None, help="sample ln c_L")
    p.add_argument("--cl-range", type=float, nargs=2, default=[0.01, 100.0],
                   help="prior range of c_L for --fit-cl (log-uniform)")
    p.add_argument("--nlive", type=int, default=400,
                   help="dynesty live points")
    p.add_argument("--dlogz", type=float, default=0.5,
                   help="stop when the evidence left in the live points is below this")
    p.add_argument("--sample", choices=["rslice", "rwalk", "slice", "unif"], default="rslice",
                   help="dynesty proposal")
    p.add_argument("--slices", type=int, default=None,
                   help="slices per proposal for (r)slice")
    p.add_argument("--maxcall", type=int, default=None,
                   help="stop after this many likelihood calls")
    p.add_argument("--seed", type=int, default=42,
                   help="sampler seed")
    p.add_argument("--npool", type=int, default=2,
                   help="parallel likelihood workers")
    p.add_argument("--resume", action="store_true", help="continue from outdir/dynesty.save if it exists")
    p.add_argument("--n-post", type=int, default=150, help="posterior draws for curves and statistics")
    p.add_argument("--n-obj", type=int, default=30, help="posterior shape OBJ files")
    p.add_argument("--diameter-km", type=float, default=None, help="scale OBJ files to this D_eq [km]")
    p.add_argument("--replot", action="store_true", help="rebuild products of an existing --outdir")
    _ades.add_cli_arguments(p)
    from lcinv import cohesion as _co
    _co.add_cli_arguments(p, diameter=False)
    jaxenv.add_cli_arguments(p, worker=True)
    return p, p.parse_args(argv)


def settings(p, a):
    """Merge data settings from the global run (if given) into the arguments."""
    gs = None
    if a.global_dir:
        gs = json.load(open(os.path.join(a.global_dir, "summary.json")))
        ga = sampler.load(os.path.join(a.global_dir, "nested_results.pkl"))["args"]
        for k in DATA_KEYS:
            if k in ga:
                setattr(a, k, ga[k])
        if a.period is None:
            pr = gs["priors"]["period"]
            if pr[0] == "gaussian":
                a.period, a.period_sigma, a.period_nsig = pr[1], pr[2], pr[3]
            else:
                a.period, a.period_sigma = 0.5 * (pr[1] + pr[2]), (pr[2] - pr[1]) / 2
        if a.fit_cl is None:
            a.fit_cl = bool(ga.get("fit_cl", False))
        if a.c_lambert is None:
            a.c_lambert = ga.get("c_lambert", 0.1)
        print(f"data, period prior and c_L settings taken from {a.global_dir} "
              f"(P = {a.period:.7f} +- {a.period_sigma:.2g} h, fit_cl = {a.fit_cl})")
    a.fit_cl = bool(a.fit_cl)
    a.c_lambert = 0.1 if a.c_lambert is None else a.c_lambert
    if a.lc is None:
        p.error("give the lightcurve file or --global-dir")
    if a.designation is None:
        a.designation = _ades.designation_from_file(a.lc)
        if a.designation is None:
            p.error("--designation is required for non-ADES lightcurves")
    if a.period is None or a.period_sigma is None:
        p.error("give --period and --period-sigma (or --global-dir)")
    return gs


def load_data(a, gs):
    """Data of the run, with the outliers clipped by the global run removed."""
    D = setup_data.prepare(a.lc, a.designation, location=a.location, cache_dir=a.cache_dir,
                           drop_bands=tuple(b for b in str(a.drop_bands or "").split(",") if b),
                           ades_opts=_ades.opts_from_args(a), observer=a.observer, **_ades.timing_from_args(a))
    clipped = (gs or {}).get("clipped_epochs", [])
    if clipped:
        D = setup_data.drop_epochs(D, clipped)
        print(f"dropped {len(clipped)} epoch(s) clipped in the global run")
    return D, clipped


def build_model(a, D):
    """ContactBinary model from the options and data."""
    return ContactBinary(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"],
                         nfacets=a.nfacets, c_lambert=a.c_lambert, fit_cl=a.fit_cl, soft=a.soft,
                         exptime=D["exptime"], smear_nodes=a.smear_nodes, smear_P_h=a.period,
                         cohesion=_cohesion_spec(a))


def _cohesion_spec(a):
    from lcinv import cohesion as co
    return co.spec_from_args(vars(a)) if getattr(a, "cohesion_max", None) is not None else None


def build_prior(a):
    """Prior transform and period-prior specification."""
    pspec = ("gaussian", a.period, a.period_sigma, a.period_nsig)
    return BinaryPrior(pspec, q_range=tuple(a.q_range), b_range=tuple(a.b_range), cb_range=tuple(a.cb_range),
                       s_range=tuple(a.s_range), ln_cl_range=tuple(np.log(a.cl_range)) if a.fit_cl else None), pspec


def derived(M, th):
    """Physical shape descriptors of one parameter vector."""
    cen, ax = (np.asarray(x) for x in M.shape(np.asarray(th, float)))
    v, f = M.union_mesh(th, n=1500)
    ext = v.max(0) - v.min(0)
    return dict(length_over_width=ext[0] / ext[1], width_over_height=ext[1] / ext[2],
                volume_ratio=float(th[5]) ** 3, neck_overlap=1 - float(th[8]))


def products(a, D, M, res, prior, pspec, clipped, gs, T0):
    """Convergence report, shapes, figures, viewer and summary.json of a finished (or --replot)
    run.
    """
    os.makedirs(os.path.join(a.outdir, "shapes"), exist_ok=True)
    from lcinv import cohesion as _co
    _co.resolve_diameter(a)                       # --H / --size-from-H -> a.diameter_km
    names = param_names(a.fit_cl)
    samples = np.asarray(res["samples"])
    conv = diagnostics.nested(res["logwt"], res["logz"], res["logzerr"], a.nlive, beta=samples[:, 1],
                              maxcall_hit=bool(a.maxcall and int(np.sum(res.get("ncall", [0]))) >= a.maxcall))
    print(diagnostics.format_report("nested sampling (contact binary)", conv))
    w = np.exp(np.asarray(res["logwt"]) - res["logz"][-1])
    from dynesty import utils as dyfunc
    post = dyfunc.resample_equal(samples, w / w.sum(), rstate=np.random.default_rng(3))
    post[:, 0] %= 360.0
    post[:, 4] %= 360.0
    i_ml = int(np.argmax(res["logl"]))
    th_ml = np.array(samples[i_ml], float)
    th_ml[0] %= 360.0
    th_ml[4] %= 360.0
    o_ml = M.loglike_jit(th_ml)
    beta_ml = np.asarray(o_ml["beta"])
    rng = np.random.default_rng(7)
    pick = rng.choice(len(post), size=min(a.n_post, len(post)), replace=False)
    betas = [np.asarray(M.loglike_jit(post[j])["beta"]) for j in pick]

    # shapes
    scale = 1.0 if not a.diameter_km else a.diameter_km / 2.0
    unit = "km" if a.diameter_km else "R_eq"
    v_ml, f_ml = M.union_mesh(th_ml)
    hdr = lambda th, what: (f"{what}: lambda={th[0]:.2f} beta={th[1]:.2f} P={th[2]:.7f} h phi0={th[4]:.2f} deg "
                            f"q={th[5]:.3f} b={th[6]:.3f} c/b={th[7]:.3f} s={th[8]:.3f}\n"
                            f"contact-binary model; body frame: z = spin axis, lobes along x; "
                            f"rotational phase phi0 at t_ref JD {M.t_ref:.6f}; units {unit}")
    mk.write_obj(os.path.join(a.outdir, "shapes", "shape_maxlike.obj"), v_ml * scale, f_ml,
                 comment=hdr(th_ml, "max-likelihood sample"))
    obj_v = []
    for k, j in enumerate(pick[:a.n_obj]):
        v, f = M.union_mesh(post[j])
        obj_v.append(v)
        mk.write_obj(os.path.join(a.outdir, "shapes", f"shape_post_{k:03d}.obj"), v * scale, f,
                     comment=hdr(post[j], f"posterior draw {k}"))
    der = [derived(M, post[j]) for j in pick]
    q = lambda x: [float(v) for v in np.percentile(x, [16, 50, 84])]

    # figures
    los = -np.mean(D["e_obs"], axis=0)
    los_l = float(np.degrees(np.arctan2(los[1], los[0])) % 360)
    los_b = float(np.degrees(np.arcsin(los[2] / np.linalg.norm(los))))
    plots.pole_map(os.path.join(a.outdir, "fig_pole_posterior.png"), post,
                   [(los_l, los_b), ((los_l + 180) % 360, -los_b)], map_theta=th_ml)
    try:
        import corner
        fig = corner.corner(post, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            title_fmt=".4g", color=plots.INK)
        fig.savefig(os.path.join(a.outdir, "fig_corner.png"), dpi=110)
    except Exception as e:  # corner is optional
        print("corner plot skipped:", e)
    texp = D["exptime"] if np.any(np.asarray(D["exptime"]) > 0) else None
    tm = float(np.median(texp)) if texp is not None else None
    jd = D["df"]["epoch"].values
    r = plots.ref_band(D["bands"])
    X = np.c_[np.eye(M.nb)[D["band_idx"]], D["alpha"]]
    curves = []
    for k in range(D["night"].max() + 1):
        m = D["night"] == k
        jd_d = setup_data.dense_epochs(jd[m].min() - 0.01, jd[m].max() + 0.01, a.period)
        g = setup_data.plot_geometry(D, jd_d)
        bi = np.full(len(jd_d), r)
        Xd = np.c_[np.eye(M.nb)[bi], g["alpha"]]
        cs = [M.predict(post[j], g["t_emit"], g["e_sun"], g["e_obs"], bi, g["alpha"], b, exptime=tm) - Xd @ beta_ml
              for j, b in zip(pick[:30], betas[:30])]
        curves.append([(g["t_emit"], cs)])
    plots.lightcurves(os.path.join(a.outdir, "fig_lightcurves.png"), D, curves, beta_ml, X)
    try:
        resid = D["mag"] - M.predict(th_ml, D["t"], D["e_sun"], D["e_obs"], D["band_idx"], D["alpha"], beta_ml,
                                     exptime=texp)
        _, t_near, es, eo, al = plots.reference_geometry(D)
        P = float(th_ml[2])
        tg, sg, t0 = plots.phase_grid(t_near, M.t_ref, P)
        n = len(tg)
        Xr = np.r_[np.eye(M.nb)[r], al]
        geo = (tg, np.repeat(es[None], n, 0), np.repeat(eo[None], n, 0), np.full(n, r), np.full(n, al))
        cv = [M.predict(th_ml, *geo, beta_ml, exptime=tm) - Xr @ beta_ml]
        cv += [M.predict(post[j], *geo, b, exptime=tm) - Xr @ beta_ml for j, b in zip(pick[:30], betas[:30])]
        plots.folded(os.path.join(a.outdir, "fig_folded_lightcurve.png"), D, P, t0, resid, sg,
                     {"posterior draws": cv}, f"{ephem.unpack_designation(a.designation)} contact-binary model",
                     alpha_ref=al)
    except Exception as e:  # plotting must never fail a finished run
        print("folded lightcurve plot skipped:", e)

    # fit quality, required cohesion
    from lcinv import fitstats, cohesion as co
    fq = M.fit_quality(th_ml)
    print(fitstats.format_line("best fit (contact binary)", fq))
    w_eq = np.exp(np.asarray(res["logwt"]) - res["logz"][-1])
    ll = np.asarray(res["logl"])
    fq["p_eff_variance"] = float(2 * (np.sum(w_eq * ll ** 2) / w_eq.sum() - (np.sum(w_eq * ll) / w_eq.sum()) ** 2))
    coh = None
    rspec = co.report_spec(vars(a))
    if rspec:
        sig = np.array([float(M.required_cohesion_jax(np.asarray(post[j], float), rspec)) for j in pick])
        coh = co.summarise(sig, rspec)
        coh["note"] = "equivalent ellipsoid of the two lobes (bulk criterion; the neck can need more)"

    # summary
    lnZ = float(res["logz"][-1])
    comp = None
    if gs is not None:
        lz = gs.get("convergence", {}).get("lnZ", gs.get("sampler", {}).get("logZ"))
        if lz is not None:
            dz = lnZ - lz
            comp = dict(lnZ_convex=float(lz), lnZ_binary=lnZ, delta_lnZ_binary_minus_convex=float(dz),
                        bayes_factor_binary_over_convex=float(np.exp(np.clip(dz, -700, 700))),
                        reading=("binary strongly favoured" if dz > 5 else "binary favoured" if dz > 2.3 else
                                 "no clear preference" if dz > -2.3 else "convex favoured" if dz > -5 else
                                 "convex strongly favoured"),
                        note="same data, period prior, jitter prior and profiled band offsets; the shape "
                             "priors differ (Laplace-marginalised harmonic coefficients vs uniform lobe parameters)")
            print(f"evidence: ln Z binary {lnZ:.2f} vs convex {lz:.2f}: delta {dz:+.2f} -> {comp['reading']}")
    ncall = int(np.sum(res.get("ncall", [0])))
    summary = dict(
        model="contact binary", designation=a.designation, horizons_target=ephem.unpack_designation(a.designation),
        n_points=int(len(D["t"])), bands=D["bands"], n_nights=int(D["night"].max() + 1),
        phase_angle_range=[float(D["alpha"].min()), float(D["alpha"].max())],
        priors=dict(period=list(pspec), **prior.describe()),
        sampler=dict(nlive=a.nlive, sample=a.sample, niter=int(res.get("niter", len(samples))), ncall=ncall,
                     logZ=lnZ, logZ_err=float(res["logzerr"][-1]), finalised_from_checkpoint=False, seed=a.seed),
        convergence=conv,
        max_likelihood=dict(names=names, theta=[float(x) for x in th_ml], logL=float(res["logl"][i_ml]),
                            chi2=float(o_ml["chi2"]), t_ref_jd=M.t_ref),
        posterior=dict(P_h=q(post[:, 2]), jitter_mag=q(np.exp(post[:, 3])), beta_deg=q(post[:, 1]),
                       frac_prograde=float(np.mean(post[:, 1] > 0)),
                       pole_hpd_sky_fraction=sampler.sky_area_fraction(post),
                       q=q(post[:, 5]), b=q(post[:, 6]), c_over_b=q(post[:, 7]), c=q(post[:, 6] * post[:, 7]),
                       s=q(post[:, 8]), volume_ratio=q(post[:, 5] ** 3),
                       length_over_width=q([d["length_over_width"] for d in der]),
                       width_over_height=q([d["width_over_height"] for d in der]),
                       **({"c_lambert": q(np.exp(post[:, 9]))} if a.fit_cl else {})),
        fit_quality=fq, cohesion=coh, data_fingerprint=fitstats.data_fingerprint(D),
        evidence_comparison=comp, clipped_epochs=clipped, line_of_sight_lonlat=[los_l, los_b],
        settings=dict(nfacets=a.nfacets, soft=a.soft, c_lambert=a.c_lambert, fit_cl=a.fit_cl,
                      smear_nodes=M.Q, global_dir=a.global_dir),
        runtime_s=time.time() - T0)
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    viewer(a, D, M, th_ml, post, pick, obj_v, v_ml, f_ml, summary)
    print(json.dumps(summary["posterior"], indent=1))
    return summary


def viewer(a, D, M, th_ml, post, pick, obj_v, v_ml, f_ml, summary):
    """shape_viewer.html with the maximum-likelihood union mesh and posterior draws."""
    import make_viewer as mv
    spin = lambda th: dict(lam=float(th[0]) % 360, bet=float(th[1]), P=float(th[2]))
    v0, f0 = M.union_mesh(th_ml, n=1500)                # lighter meshes for the HTML viewer
    shapes = [mv._shape_rec("Max-likelihood", v0, f0, spin(th_ml))]
    for k, j in enumerate(pick[:len(obj_v)]):
        v, f = M.union_mesh(post[j], n=1500)
        shapes.append(mv._shape_rec(f"Posterior draw {k + 1}", v, f, spin(post[j])))
    p = summary["posterior"]
    f3 = lambda x, d: f"{x[1]:.{d}f} <span style=\"color:var(--mute)\">({x[0]:.{d}f}–{x[2]:.{d}f})</span>"
    meta = dict(designation=summary["horizons_target"], npts=summary["n_points"], nnights=summary["n_nights"],
                bands=summary["bands"], alpha=summary["phase_angle_range"], P=p["P_h"], jit=p["jitter_mag"],
                fpro=p["frac_prograde"], eab=p["length_over_width"], ebc=p["width_over_height"],
                kind="Contact-binary model (two ellipsoidal lobes)",
                extra=[["Lobe size ratio q", f3(p["q"], 2)], ["Volume ratio q³", f3(p["volume_ratio"], 2)],
                       ["Separation s", f3(p["s"], 3)], ["Lobe b/a, c/a", f"{p['b'][1]:.2f}, {p['c'][1]:.2f}"]])
    if "c_lambert" in p:
        meta["cl"] = p["c_lambert"]
    mv._write(a.outdir, "shape_viewer.html", shapes, meta, f"{summary['horizons_target']} contact-binary model")


def main(argv=None):
    """Command line of run_binary.py."""
    T0 = time.time()
    p, a = parse(argv)
    if a.replot:
        R = sampler.load(os.path.join(a.outdir, "nested_results.pkl"))
        keep = {k: getattr(a, k) for k in ("outdir", "n_post", "n_obj", "diameter_km", "replot", "H", "albedo",
                                           "size_from_H")}
        for k, v in R["args"].items():
            setattr(a, k, v)
        for k, v in keep.items():
            setattr(a, k, v)
        gs = json.load(open(os.path.join(a.global_dir, "summary.json"))) if a.global_dir else None
        D, clipped = load_data(a, gs)
        M = build_model(a, D)
        prior, pspec = build_prior(a)
        products(a, D, M, R["res"], prior, pspec, clipped, gs, T0)
        print(f"replotted -> {a.outdir}")
        return
    gs = settings(p, a)
    os.makedirs(a.outdir, exist_ok=True)
    D, clipped = load_data(a, gs)
    M = build_model(a, D)
    prior, pspec = build_prior(a)
    print(f"{len(D['t'])} points, {D['night'].max() + 1} nights, bands {D['bands']}; contact-binary model with "
          f"{2 * len(M.tri_np)} facets, {M.Q} exposure node(s), {prior.ndim} parameters: {param_names(a.fit_cl)}")
    print(f"likelihood at a test point: {M.loglike(M.warmup_theta()):.1f}")
    res = sampler.run_nested(M, prior, nlive=a.nlive, dlogz=a.dlogz, sample=a.sample, seed=a.seed,
                             maxcall=a.maxcall, npool=a.npool, slices=a.slices,
                             checkpoint_file=os.path.join(a.outdir, "dynesty.save"), resume=a.resume,
                             worker_device=a.worker_device, worker_threads=a.worker_threads)
    R = dict(samples=np.asarray(res.samples), logwt=np.asarray(res.logwt), logl=np.asarray(res.logl),
             logz=np.asarray(res.logz), logzerr=np.asarray(res.logzerr), ncall=np.asarray(res.ncall),
             niter=int(res.niter))
    sampler.save(os.path.join(a.outdir, "nested_results.pkl"), samples=R["samples"], logwt=R["logwt"],
                 logl=R["logl"], logz=R["logz"], logzerr=R["logzerr"], args=vars(a), res=R, ensemble=None)
    s = products(a, D, M, R, prior, pspec, clipped, gs, T0)
    print(f"done in {time.time() - T0:.0f} s -> {a.outdir}")
    return s


if __name__ == "__main__":
    main()
