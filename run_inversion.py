#!/usr/bin/env python
"""End-to-end Bayesian convex inversion of a single lightcurve file.

  python run_inversion.py data/K03Q21N.lc --designation K03Q21N --location X05 \
      --nlive 400 --period-prior gaussian --ab-prior lower --outdir results/K03Q21N

Steps
  1. geometry from JPL Horizons (astroquery; cached CSV in --cache-dir), light-time corrected
  2. multiband Fourier period search, period uncertainty, amplitude -> a/b prior
  3. JAX convex model, Laplace-marginalised shape, jitter
  4. dynesty nested sampling over (lambda, beta, P, ln sigma_jit)
  5. posterior shape ensemble, Minkowski reconstruction, OBJ files, figures, summary.json
"""
import sys

from lcinv import jaxenv

jaxenv.early_setup(sys.argv, host_devices_flag=None)  # before anything imports JAX

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from lcinv import period as per, plots, sampler, minkowski as mk, ephem
from lcinv import diagnostics  # noqa: E402
from lcinv import folding  # noqa: E402
from lcinv.sampler import sky_area_fraction  # noqa: E402
from lcinv.setup_data import prepare  # noqa: E402
from lcinv import setup_data  # noqa: E402
from lcinv.convex import ConvexInversion  # noqa: E402
from lcinv import cohesion, fitstats  # noqa: E402


def parse(argv=None):
    """Parse the options of run_inversion.py (argv: list of arguments, default sys.argv)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lc",
                   help="lightcurve file: Rubin-style .lc/.csv with epoch, mag, err, band (and "
                        "optional geometry), or an MPC ADES CSV/PSV/XML file")
    p.add_argument("--designation", default=None, help="e.g. K03Q21N (packed) or 2003 QN21")
    p.add_argument("--location", default="X05", help="MPC observatory code (X05 = Rubin)")
    p.add_argument("--cache-dir", default="data",
                   help="folder for cached JPL Horizons tables (reused when they cover the epochs)")
    p.add_argument("--outdir", default="results",
                   help="results folder (created if needed)")
    p.add_argument("--drop-bands", default="u",
                   help="comma-separated bands to remove before fitting (empty string keeps all)")
    # period search
    p.add_argument("--pmin", type=float, default=2.0,
                   help="shortest rotation period accepted [h]; the periodogram starts at min(1 h, "
                        "--pmin), and a deeper minimum below --pmin triggers a warning")
    p.add_argument("--pmax", type=float, default=30.0,
                   help="max rotation period searched [h]")
    p.add_argument("--harmonics", type=int, default=6,
                   help="Fourier harmonics K of the period search (rotation frequency as fundamental)")
    p.add_argument("--sys-err", type=float, default=0.015, help="error floor for period search [mag]")
    p.add_argument("--clip-sigma", type=float, default=5.0,
                   help="drop points beyond this many robust sigma from the Fourier fit (0 = off)")
    # priors
    p.add_argument("--period-prior", choices=["gaussian", "uniform"], default="gaussian",
                   help="prior on P: truncated Gaussian around the period-search result, or uniform "
                        "over +- --period-nsig sigma")
    p.add_argument("--period", type=float, default=None, help="override prior centre [h]")
    p.add_argument("--period-sigma", type=float, default=None, help="override prior sigma [h]")
    p.add_argument("--period-nsig", type=float, default=5.0,
                   help="half-width of the period prior in units of its sigma")
    p.add_argument("--search-window", type=float, default=None, metavar="W",
                   help="search the period only within --period x (1 +- W) instead of the full grid; "
                        "needed for multi-apparition data, whose full grid has millions of trial "
                        "periods")
    p.add_argument("--ab-prior", choices=["none", "lower", "lognormal"], default="lower",
                   help="a/b prior from the amplitude: one-sided lower bound or lognormal")
    p.add_argument("--ab", type=float, default=None, help="override a/b value")
    p.add_argument("--ab-sigma", type=float, default=None, help="sigma of ln(a/b)")
    p.add_argument("--ab-model-err", type=float, default=0.05,
                   help="extra ln(a/b) uncertainty for non-ellipsoidal shapes/scattering")
    p.add_argument("--bc-prior", choices=["none", "lower", "lognormal"], default="lower",
                   help="b/c prior; default: soft lower bound b/c >= 1, i.e. spin about the "
                        "shortest axis (relaxed principal-axis rotator)")
    p.add_argument("--bc", type=float, default=1.0, help="b/c prior value")
    p.add_argument("--bc-sigma", type=float, default=0.05, help="sigma of ln(b/c)")
    p.add_argument("--amp-phase-slope", type=float, default=0.030,
                   help="amplitude-phase slope m [mag/deg] for correcting the amplitude to zero phase "
                        "angle (a/b bound)")
    # model / sampler
    p.add_argument("--lmax", type=int, default=4,
                   help="maximum spherical-harmonic degree of the log facet areas (shape resolution)")
    p.add_argument("--nfacets", type=int, default=400,
                   help="number of Gaussian-image facets (Fibonacci sphere)")
    p.add_argument("--n-inner", type=int, default=None,
                   help="L-BFGS iterations of the shape optimisation per likelihood call "
                        "(default 80; 400 with --fit-tau or tau0 > 0.8, where 80 does not converge)")
    p.add_argument("--n-starts", type=int, default=3,
                   help="starts of the inner shape optimisation per likelihood call; the best is kept")
    p.add_argument("--n-stage1", type=int, default=40,
                   help="L-BFGS iterations of the coarse l <= 2 stage that seeds each start")
    p.add_argument("--tau0", type=float, default=0.6,
                   help="prior scale of the shape coefficients, tau_l = tau0 / l (fixed)")
    p.add_argument("--fit-tau", action="store_true",
                   help="infer tau0 from the data (log-uniform prior over --tau-range): data-driven "
                        "shape complexity within one run")
    p.add_argument("--tau-range", type=float, nargs=2, default=[0.2, 5.0], metavar=("LO", "HI"),
                   help="log-uniform prior range of tau0 for --fit-tau")
    p.add_argument("--c-lambert", type=float, default=0.1,
                   help="Lambert weight c_L of the scattering law S = mu mu0 [1/(mu+mu0) + c_L] (fixed)")
    p.add_argument("--fit-cl", action="store_true",
                   help="sample ln c_L (log-uniform prior over --cl-range) instead of fixing it")
    p.add_argument("--cl-range", type=float, nargs=2, default=[0.01, 100.0], metavar=("LO", "HI"),
                   help="prior range of c_L for --fit-cl (0.01 ~ pure Lommel-Seeliger, 100 ~ pure Lambert)")
    p.add_argument("--nlive", type=int, default=400,
                   help="dynesty live points (more: smoother evidence and posterior, proportionally "
                        "slower)")
    p.add_argument("--dlogz", type=float, default=0.5,
                   help="stop when the evidence left in the live points is below this (in ln Z)")
    p.add_argument("--sample", choices=["rslice", "rwalk", "slice", "unif"], default="rslice",
                   help="dynesty proposal (rslice: robust on thin pole ridges; unif: fast early, "
                        "but collapses on curved posteriors)")
    p.add_argument("--bootstrap", type=int, default=0, help="dynesty bootstrap (only matters for unif)")
    p.add_argument("--slices", type=int, default=None, help="slices per proposal for (r)slice")
    p.add_argument("--maxcall", type=int, default=None,
                   help="stop after this many likelihood calls (finish later with --resume "
                        "--extra-calls)")
    p.add_argument("--seed", type=int, default=42, help="sampler seed (repeat with another seed to check)")
    p.add_argument("--npool", type=int, default=2, help="parallel likelihood workers")
    from lcinv import ades as _ades
    _ades.add_cli_arguments(p)
    from lcinv import cohesion as _co
    _co.add_cli_arguments(p)
    jaxenv.add_cli_arguments(p, worker=True)
    p.add_argument("--resume", action="store_true",
                   help="finish from <outdir>/dynesty.save instead of starting a new run")
    p.add_argument("--extra-calls", type=int, default=0,
                   help="with --resume: continue sampling for this many calls before adding live points")
    p.add_argument("--n-post", type=int, default=150, help="posterior draws for shapes/curves")
    p.add_argument("--n-obj", type=int, default=30, help="posterior shape OBJ files to write")
    a = p.parse_args(argv)
    if a.n_inner is None:
        a.n_inner = 400 if (a.fit_tau or a.tau0 > 0.8) else 80
    if a.designation is None:
        a.designation = _ades.designation_from_file(a.lc)
        if a.designation is None:
            p.error("--designation is required for non-ADES lightcurves")
    return a


def main():
    """Run the global inversion: data, period search, model, nested sampling, shapes, figures and
    summary.
    """
    a = parse()
    print(jaxenv.describe())
    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "shapes"), exist_ok=True)
    T0 = time.time()

    # 1. data + geometry ------------------------------------------------------------------
    from lcinv import ades
    D = prepare(a.lc, a.designation, location=a.location, cache_dir=a.cache_dir,
                ades_opts=ades.opts_from_args(a), observer=a.observer, **ades.timing_from_args(a),
                drop_bands=tuple(b for b in a.drop_bands.split(",") if b))
    print(f"{len(D['t'])} points, bands {D['bands']}, nights {D['night'].max() + 1}; "
          f"geometry vs file: {D['check']}")

    # 2. period search + priors -----------------------------------------------------------
    if a.search_window and not a.period:
        raise SystemExit("--search-window needs --period (the centre of the window)")

    def search(D):
        sig = np.hypot(D["err"], a.sys_err)
        if a.search_window:
            lo, hi = a.period * (1 - a.search_window), a.period * (1 + a.search_window)
        else:
            lo, hi = min(1.0, a.pmin), a.pmax
        P, chi2, nu = per.periodogram(D["t"], D["mag"], sig, D["band_idx"], D["alpha"],
                                      lo, hi, K=a.harmonics, oversample=5,
                                      exptime=D["exptime"])
        sel = (P >= a.pmin) if not a.search_window else np.ones(P.shape, bool)
        P_best = per.local_minima(P[sel], chi2[sel], n=6)[0][0]
        ref = per.refine_period_auto(D["t"], D["mag"], sig, D["band_idx"], D["alpha"], P_best,
                                     K=a.harmonics, window=a.search_window, exptime=D["exptime"])
        return sig, P, chi2, nu, ref

    sig, P, chi2, nu, ref = search(D)
    clipped = []
    if a.clip_sigma and a.clip_sigma > 0:
        keep = setup_data.clip_outliers(D, a.period or ref["P"], K=a.harmonics, nsig=a.clip_sigma,
                                        sys_err=a.sys_err)
        if not keep.all():
            bad = D["df"][~keep]
            clipped = [float(x) for x in bad["epoch"]]
            print(f"clipped {len(clipped)} outlier(s) (> {a.clip_sigma:g} robust sigma from the Fourier fit): "
                  + ", ".join(f"JD {e:.5f} {b}" for e, b in zip(bad["epoch"], bad["band"])))
            D = setup_data.subset(D, keep)
            sig, P, chi2, nu, ref = search(D)
    amp, amp_err = per.amplitude(D["t"], D["mag"], sig, D["band_idx"], D["alpha"], ref["P"], K=a.harmonics,
                                 exptime=D["exptime"])
    abp = per.ab_prior_from_amplitude(amp, amp_err, float(np.mean(D["alpha"])), m_slope=a.amp_phase_slope)
    plots.periodogram(os.path.join(a.outdir, "fig_periodogram.png"), P, chi2, nu, ref,
                      per.local_minima(P, chi2, n=6))
    range_warn = [] if a.search_window else per.range_warnings(P, chi2, nu, a.pmin, ref["red_chi2"])
    P0 = a.period if a.period else ref["P"]
    sP = a.period_sigma if a.period_sigma else 0.5 * (ref["P_hi"] - ref["P_lo"])
    print(f"period {ref['P']:.5f} h (Delta chi2=1: +-{ref['sigma']:.5f}; threshold half-width {sP:.5f}); "
          f"amplitude {amp:.3f}+-{amp_err:.3f}; a/b >= {abp['ab_min']:.3f}")
    for w in range_warn:
        print("WARNING (period search):", w)
    ab_spec = None
    if a.ab_prior != "none":
        val = a.ab if a.ab else abp["ab_min"]
        s = a.ab_sigma if a.ab_sigma else float(np.hypot(abp["ln_ab_min_err"], a.ab_model_err))
        ab_spec = (a.ab_prior, val, s)
    bc_spec = None if a.bc_prior == "none" else (a.bc_prior, a.bc, a.bc_sigma)

    # 3. model ----------------------------------------------------------------------------
    M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"],
                        D["e_obs"], lmax=a.lmax, nfacets=a.nfacets, n_inner=a.n_inner, n_starts=a.n_starts, n_stage1=a.n_stage1,
                        tau0=a.tau0, ab_prior=ab_spec, bc_prior=bc_spec,
                        exptime=D["exptime"], smear_nodes=a.smear_nodes, smear_P_h=P0,
                        c_lambert=a.c_lambert, fit_cl=a.fit_cl, cohesion=cohesion.spec_from_args(a),
                        fit_tau=a.fit_tau)
    if M.cohesion:
        print(f"cohesion limit: shapes needing more than {a.cohesion_max:g} Pa are penalised "
              f"(D = {a.diameter_km:g} km, rho = {a.density:g} kg/m^3, friction angle {a.friction_angle:g} deg)")
    if M.Q > 1 or np.any(D["exptime"] > 0):
        print(f"exposure smearing: T = {np.median(D['exptime']):.1f} s (median), rotation swept "
              f"{np.degrees(2 * np.pi * np.median(D['exptime']) / (P0 * 3600)):.1f} deg, {M.Q} node(s) per exposure")
    pspec = (("gaussian", P0, sP, a.period_nsig) if a.period_prior == "gaussian"
             else ("uniform", P0 - a.period_nsig * sP, P0 + a.period_nsig * sP))
    prior = sampler.PriorTransform(pspec, ln_cl_range=tuple(np.log(a.cl_range)) if a.fit_cl else None,
                                   ln_tau_range=tuple(np.log(a.tau_range)) if a.fit_tau else None)
    M.loglike(prior(np.full(prior.ndim, 0.5)))  # compile

    # 4. nested sampling ------------------------------------------------------------------
    ckpt = os.path.join(a.outdir, "dynesty.save")
    if a.resume and os.path.exists(ckpt):
        res = sampler.finalize_from_checkpoint(ckpt, model=M, npool=a.npool, extra_calls=a.extra_calls,
                                               dlogz=a.dlogz, worker_device=a.worker_device,
                                               worker_threads=a.worker_threads)
    else:
        res = sampler.run_nested(M, prior, nlive=a.nlive, dlogz=a.dlogz, sample=a.sample,
                                 maxcall=a.maxcall, npool=a.npool, checkpoint_file=ckpt,
                                 worker_device=a.worker_device, worker_threads=a.worker_threads,
                                 bootstrap=a.bootstrap, slices=a.slices, seed=a.seed)
    ncall_tot = int(np.sum(res.ncall))
    conv = diagnostics.nested(res.logwt, res.logz, res.logzerr, a.nlive, beta=np.asarray(res.samples)[:, 1],
                              resumed=a.resume, maxcall_hit=bool(a.maxcall and ncall_tot >= a.maxcall))
    print(diagnostics.format_report("nested sampling", conv))
    post = sampler.equal_weight_samples(res)
    i_ml = int(np.argmax(res.logl))
    theta_ml = np.array(res.samples[i_ml], float)
    theta_ml[0] %= 360.0

    # 5. shapes ----------------------------------------------------------------------------
    ens = sampler.shape_ensemble(M, post, n=a.n_post)
    o_ml = M.loglike_jit(np.asarray(theta_ml, float))
    A_ml = np.asarray(M.areas(o_ml["c"]))
    v_ml, f_ml = mk.reconstruct(M.normals_np, A_ml)
    mk.write_obj(os.path.join(a.outdir, "shapes", "shape_maxlike.obj"), v_ml, f_ml,
                 comment=f"max-likelihood sample: lambda={theta_ml[0]:.2f} beta={theta_ml[1]:.2f} "
                         f"P={theta_ml[2]:.6f} h, rotational phase 0 at t_ref JD {M.t_ref:.6f}\n"
                         "body frame: z = spin axis; volume normalised to 1")
    draw_verts = []
    for k in range(min(a.n_obj, len(ens["c_draw"]))):
        A = np.asarray(M.areas(np.asarray(ens["c_draw"][k])))
        v, f = mk.reconstruct(M.normals_np, A)
        draw_verts.append(v)
        th = ens["theta"][k]
        mk.write_obj(os.path.join(a.outdir, "shapes", f"shape_post_{k:03d}.obj"), v, f,
                     comment=f"posterior draw {k}: lambda={th[0]:.2f} beta={th[1]:.2f} "
                             f"P={th[2]:.6f} h, phase 0 at t_ref JD {M.t_ref:.6f}")
    dims = np.array([np.sort(mk.dimensions(v))[::-1] for v in draw_verts])

    # 6. figures ---------------------------------------------------------------------------
    los = -np.mean(D["e_obs"], axis=0)
    los_l = np.degrees(np.arctan2(los[1], los[0])) % 360
    los_b = np.degrees(np.arcsin(los[2] / np.linalg.norm(los)))
    plots.pole_map(os.path.join(a.outdir, "fig_pole_posterior.png"), post,
                   [(los_l, los_b), ((los_l + 180) % 360, -los_b)], map_theta=theta_ml)
    try:
        import corner
        fig = corner.corner(post, labels=sampler.param_names(post.shape[1], a.fit_cl, a.fit_tau),
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt=".4g", color=plots.INK)
        fig.savefig(os.path.join(a.outdir, "fig_corner.png"), dpi=130)
    except Exception as e:  # corner is optional
        print("corner plot skipped:", e)

    nb = M.nb
    X = np.c_[np.eye(nb)[D["band_idx"]], D["alpha"]]
    beta_ml = np.asarray(o_ml["beta"])
    r_idx = D["bands"].index("r") if "r" in D["bands"] else 0
    jd = D["df"]["epoch"].values
    curves = []
    rng = np.random.default_rng(7)
    pick = rng.choice(len(ens["theta"]), size=min(30, len(ens["theta"])), replace=False)
    for k in range(D["night"].max() + 1):
        m = D["night"] == k
        jd_d = setup_data.dense_epochs(jd[m].min() - 0.01, jd[m].max() + 0.01, P0)
        g = setup_data.plot_geometry(D, jd_d)
        bi = np.full(len(jd_d), r_idx)
        Xd = np.c_[np.eye(nb)[bi], g["alpha"]]
        cs = []
        for j in pick:
            mod = M.predict(ens["theta"][j], g["t_emit"], g["e_sun"], g["e_obs"], bi, g["alpha"],
                            ens["c_hat"][j], ens["beta"][j],
                            exptime=float(np.median(D["exptime"])) if np.any(D["exptime"] > 0) else None)
            cs.append(mod - Xd @ beta_ml)
        curves.append([(g["t_emit"], cs)])
    plots.lightcurves(os.path.join(a.outdir, "fig_lightcurves.png"), D, curves, beta_ml, X)
    try:
        folding.convex(os.path.join(a.outdir, "fig_folded_lightcurve.png"), D, M, theta_ml, o_ml, ens, pick,
                          f"{D.get('designation', a.designation) or ''} global convex model".strip())
    except Exception as e:  # plotting must never fail a finished run
        print("folded lightcurve plot skipped:", e)
    plots.shapes(os.path.join(a.outdir, "fig_shapes.png"), v_ml, draw_verts, ens["ab"], ens["bc"])

    # 7. summary ---------------------------------------------------------------------------
    q = lambda x: [float(v) for v in np.nanpercentile(np.asarray(x, float), [16, 50, 84])]
    fq = M.fit_quality(theta_ml)
    print(fitstats.format_line("best fit (convex)", fq))
    coh = None
    rspec = cohesion.report_spec(a)
    if rspec:
        ax = cohesion.semi_axes_from_ratios(rspec["diameter_km"], np.asarray(ens["ab"]),
                                            np.maximum(np.asarray(ens["bc"]), 1.0))
        sig = cohesion.required_cohesion(*ax, np.asarray(ens["theta"])[:, 2], rspec["density"], rspec["friction_deg"])
        coh = cohesion.summarise(sig, rspec)
    if coh and coh.get("required_cohesion_Pa"):
        print(f"required cohesion (Sanchez & Scheeres 2014): {coh['required_cohesion_Pa'][1]:.3g} Pa "
              f"[{coh['required_cohesion_Pa'][0]:.3g}, {coh['required_cohesion_Pa'][2]:.3g}]")
    summary = dict(
        designation=a.designation, horizons_target=ephem.unpack_designation(a.designation),
        n_points=int(len(D["t"])), bands=D["bands"], n_nights=int(D["night"].max() + 1),
        phase_angle_range=[float(D["alpha"].min()), float(D["alpha"].max())],
        geometry_check_vs_file=D["check"],
        period_search=dict(P_best_h=float(ref["P"]), sigma_dchi2_1=float(ref["sigma"]),
                           threshold_interval=[float(ref["P_lo"]), float(ref["P_hi"])],
                           red_chi2=float(ref["red_chi2"]), harmonics=a.harmonics,
                           other_minima=[(float(p), float(c / nu)) for p, c in per.local_minima(P, chi2, n=6)],
                           warnings=range_warn),
        amplitude=dict(A=amp, A_err=amp_err, **abp),
        priors=dict(period=pspec,
                    ab=ab_spec, bc=bc_spec, pole="uniform on sphere",
                    phi0="gauge-fixed to 0 at t_ref", ln_jitter="uniform in [ln 1e-3, ln 0.1]", shape_coeff_tau0=a.tau0, lmax=a.lmax),
        sampler=dict(nlive=a.nlive, sample=a.sample, niter=int(res.niter), ncall=int(np.sum(res.ncall)),
                     logZ=float(res.logz[-1]), logZ_err=float(res.logzerr[-1]),
                     finalised_from_checkpoint=bool(a.resume), seed=a.seed),
        convergence=conv,
        max_likelihood=dict(theta=[float(x) for x in theta_ml], logL=float(res.logl[i_ml]),
                            chi2=float(o_ml["chi2"]), t_ref_jd=M.t_ref),
        posterior=dict(P_h=q(post[:, 2]), jitter_mag=q(np.exp(post[:, 3])),
                       beta_deg=q(post[:, 1]), frac_prograde=float(np.mean(post[:, 1] > 0)),
                       pole_hpd_sky_fraction=sky_area_fraction(post),
                       ab=q(ens["ab"]), bc=q(ens["bc"]),
                       extent_ratios_a_b=q(dims[:, 0] / dims[:, 1]),
                       extent_ratios_b_c=q(dims[:, 1] / dims[:, 2]),
                       chi2_per_point=q(ens["chi2"] / len(D["t"])),
                       **({"c_lambert": q(np.exp(post[:, 4]))} if a.fit_cl else {}),
                       **({"tau0": q(np.exp(post[:, M.i_tau]))} if a.fit_tau else {})),
        fit_quality=fq, cohesion=coh, model="convex", data_fingerprint=fitstats.data_fingerprint(D),
        clipped_epochs=clipped,
        line_of_sight_lonlat=[float(los_l), float(los_b)],
        runtime_s=time.time() - T0)
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sampler.save(os.path.join(a.outdir, "nested_results.pkl"), samples=res.samples, logwt=res.logwt,
                 logl=res.logl, logz=res.logz, logzerr=res.logzerr, post=post, args=vars(a),
                 ensemble={k: v for k, v in ens.items()})
    print(json.dumps(summary["posterior"], indent=1))
    print(diagnostics.format_report("nested sampling", conv))
    print(f"done in {time.time() - T0:.0f} s -> {a.outdir}")


if __name__ == "__main__":
    main()
