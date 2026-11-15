#!/usr/bin/env python
"""Hybrid inversion: spin states (pole solutions) of the global nested-sampling run ->
radius-field NUTS per spin state.

  python run_hybrid.py data/K03Q21N.lc --designation K03Q21N \
      --global-dir results/K03Q21N --outdir results/K03Q21N_hybrid

Requires a finished global run (run_inversion.py) in --global-dir, or explicit --poles.
"""
import os
import sys
import argparse

# parallel NUTS chains on CPU need one XLA host device per chain; device, GPU memory and
# host-device settings must be applied before JAX is imported
from lcinv import jaxenv  # noqa: E402

jaxenv.early_setup(sys.argv, host_devices_flag="--chains", default_host_devices=2)

import json  # noqa: E402
import time  # noqa: E402
import numpy as np  # noqa: E402

from lcinv import sampler as gsampler, diagnostics  # noqa: E402
from lcinv.setup_data import prepare  # noqa: E402
from lcinv.hybrid.quadrature import make_grid, orthonormality_error  # noqa: E402
from lcinv.hybrid.shape import RadiusField  # noqa: E402
from lcinv.hybrid.model import LocalModel, lonlat_to_vec  # noqa: E402
from lcinv.hybrid import modes as md, inference as inf, checks, report, naming  # noqa: E402
from lcinv import cohesion, fitstats  # noqa: E402


def parse():
    """Parse the options of run_hybrid.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lc",
                   help="lightcurve file (plain or ADES), the same data as the global run")
    p.add_argument("--designation", default=None,
                   help="target designation (default: from an ADES file)")
    p.add_argument("--location", default="X05",
                   help="MPC observatory code for plain lightcurves")
    p.add_argument("--cache-dir", default="data",
                   help="folder of cached JPL Horizons tables")
    p.add_argument("--global-dir", default=None, help="output folder of run_inversion.py")
    p.add_argument("--poles", default=None, help="explicit spin-state pole centres 'lam,bet;lam,bet' (deg)")
    p.add_argument("--period", type=float, default=None,
                   help="centre of the Gaussian period prior [h] (default: the global run's "
                        "period-search result)")
    p.add_argument("--period-sigma", type=float, default=None,
                   help="width of the period prior [h] (default: half the global run's period "
                        "threshold interval)")
    p.add_argument("--outdir", default="results/hybrid",
                   help="results folder")
    # shape model
    p.add_argument("--lmax", type=int, default=5,
                   help="maximum spherical-harmonic degree of the log-radius field")
    p.add_argument("--nlat", type=int, default=16, help="mesh/quadrature latitude nodes")
    p.add_argument("--nu", type=float, default=2.0, help="Matern-like spectral slope (damping)")
    p.add_argument("--sigma-f", type=float, default=0.15, help="prior median of sd(log r)")
    p.add_argument("--ell", type=float, default=3.0, help="prior median of angular scale")
    p.add_argument("--inertia-offdiag", type=float, default=0.005,
                   help="scale of the soft constraint on the products of inertia I_xz, I_yz "
                        "(principal-axis rotation)")
    p.add_argument("--inertia-order", type=float, default=0.005,
                   help="scale of the soft constraint that z is the axis of maximum inertia")
    p.add_argument("--convexity-scale", type=float, default=0.02, help="<=0 disables")
    p.add_argument("--c-lambert", type=float, default=0.1, help="fixed Lambert weight c_L")
    p.add_argument("--fit-cl", action="store_true", help="sample ln c_L (lognormal prior, --cl-prior)")
    p.add_argument("--cl-prior", type=float, nargs=2, default=[0.3, 2.0], metavar=("MEDIAN", "LNSD"),
                   help="lognormal prior of c_L: median and sd of ln c_L (default 0.3, 2.0)")
    p.add_argument("--frame", choices=["parallel", "euler"], default="parallel",
                   help="body frame: quaternion parallel transport (default) or legacy Euler angles")
    # inference
    p.add_argument("--starts", type=int, default=4,
                   help="MAP starts per spin state (elongation at different angles)")
    p.add_argument("--map-iter", type=int, default=2000,
                   help="L-BFGS iterations of each MAP start")
    p.add_argument("--warmup", type=int, default=400,
                   help="NUTS warm-up iterations per chain")
    p.add_argument("--samples", type=int, default=400,
                   help="NUTS samples per chain")
    p.add_argument("--chains", type=int, default=2,
                   help="NUTS chains per spin state (run in parallel on CPU host devices)")
    p.add_argument("--tree-depth", type=int, default=7,
                   help="maximum NUTS tree depth")
    p.add_argument("--target-accept", type=float, default=0.9,
                   help="NUTS target acceptance probability (raise to 0.95 against divergences)")
    # checks
    p.add_argument("--check-draws", type=int, default=20, help="draws for the convexity check")
    p.add_argument("--shadow-draws", type=int, default=4,
                   help="posterior draws for the self-shadowing check")
    p.add_argument("--shadow-epochs", type=int, default=30,
                   help="epochs per draw for the self-shadowing check")
    p.add_argument("--sensitivity", action="store_true", help="MAP+Laplace for nu in {1.5, 3}")
    from lcinv import ades as _ades
    _ades.add_cli_arguments(p)
    cohesion.add_cli_arguments(p)
    jaxenv.add_cli_arguments(p)
    a = p.parse_args()
    if a.designation is None:
        a.designation = _ades.designation_from_file(a.lc)
        if a.designation is None:
            p.error("--designation is required for non-ADES lightcurves")
    return a


def main():
    """Run the hybrid inversion: spin states, MAP, Laplace evidence, NUTS, checks, figures and
    summary.
    """
    a = parse()
    print(jaxenv.describe())
    T0 = time.time()
    os.makedirs(a.outdir, exist_ok=True)
    from lcinv import ades
    D = prepare(a.lc, a.designation, location=a.location, cache_dir=a.cache_dir,
                ades_opts=ades.opts_from_args(a), observer=a.observer, **ades.timing_from_args(a))
    print(f"{len(D['t'])} points; geometry vs file {D['check']}")

    # ------------------------------------------------------------------ global layer
    gpost, gsumm = None, None
    if a.global_dir:
        G = gsampler.load(os.path.join(a.global_dir, "nested_results.pkl"))
        gsumm = json.load(open(os.path.join(a.global_dir, "summary.json")))
        gpost = G["post"]
        if gsumm.get("clipped_epochs"):
            from lcinv.setup_data import drop_epochs
            D = drop_epochs(D, gsumm["clipped_epochs"])
            print(f"dropped {len(gsumm['clipped_epochs'])} outlier(s) clipped in the global run -> {len(D['t'])} points")
    if a.poles:
        modes = []
        for k, s in enumerate(a.poles.split(";")):
            lam, bet = map(float, s.split(","))
            modes.append(dict(center=lonlat_to_vec(lam, bet), lam=lam, bet=bet, frac_global=np.nan,
                              weight_global=np.nan))
    else:
        modes = md.find_modes(gpost[:, 0], gpost[:, 1])
    for k, m in enumerate(modes):
        m["name"] = naming.name(k + 1)
    P0 = a.period or (gsumm["period_search"]["P_best_h"] if gsumm else None)
    sP = a.period_sigma or (0.5 * np.diff(gsumm["period_search"]["threshold_interval"])[0] if gsumm else None)
    print("spin states from the global layer:", [(m["name"], round(m["lam"]), round(m["bet"]), m["weight_global"]) for m in modes])

    # ------------------------------------------------------------------ local layer
    grid = make_grid(a.lmax, nlat=a.nlat, nlon=2 * a.nlat)
    print(f"grid {grid.nlat}x{grid.nlon}, k = {grid.k}, SH orthonormality error {orthonormality_error(grid):.1e}")

    def make_model(m, nu=a.nu, sigma_f=a.sigma_f, ell=a.ell):
        F = RadiusField(grid, nu=nu, ln_sigma_prior=(np.log(sigma_f), 0.5), ln_ell_prior=(np.log(ell), 0.5),
                        inertia_offdiag=a.inertia_offdiag, inertia_order=a.inertia_order,
                        convexity_scale=a.convexity_scale if a.convexity_scale > 0 else None)
        return LocalModel(D, F, m["center"], P0, sP, frame=a.frame, smear_nodes=a.smear_nodes,
                          c_lambert=a.c_lambert, fit_cl=a.fit_cl,
                          ln_cl_prior=(np.log(a.cl_prior[0]), a.cl_prior[1]), cohesion=coh_spec)

    coh_spec = cohesion.spec_from_args(a)
    rspec = cohesion.report_spec(a)
    if coh_spec:
        print(f"cohesion limit {a.cohesion_max:g} Pa (D = {a.diameter_km:g} km, rho = {a.density:g} kg/m^3)")

    runs = []
    for m in modes:
        t0 = time.time()
        M = make_model(m)
        if M.Q > 1:
            print(f"{m['name']}: exposure smearing with {M.Q} Gauss-Legendre nodes per exposure "
                  f"(median T = {np.median(M.exptime):.1f} s)")
        cl0 = (gsumm or {}).get("posterior", {}).get("c_lambert", [None, None])[1] if a.fit_cl else None
        x_map, lp_map, info = md.find_map(M, M.initial_points(a.starts, c_l0=cl0), n_iter=a.map_iter)
        lap = md.laplace_evidence(M, x_map)
        fq = M.fit_quality(x_map)
        print(fitstats.format_line(f"{m['name']} MAP", fq))
        print(f"{m['name']}: MAP logp {lp_map:.2f} (starts {np.round(info['start_logp'], 1)}), "
              f"|grad| {info['grad_norm']:.1e}, ln Z_Laplace {lap['lnZ']:.2f}, "
              f"non-positive curvatures {lap['n_nonpositive']}  [{time.time() - t0:.0f}s]")
        res = inf.run_nuts(M, x_map, lap["cov"], num_warmup=a.warmup, num_samples=a.samples,
                           num_chains=a.chains, max_tree_depth=a.tree_depth, target_accept=a.target_accept,
                           progress=False)
        xs = res["x"].reshape(-1, M.ndim)
        der = inf.derived(M, xs)
        conv = inf.convergence_summary(res, der)
        conv["diagnostics"] = diagnostics.nuts(res["x"], res["diverging"], der)
        print(diagnostics.format_report(m["name"], conv["diagnostics"]))
        print(f"{m['name']}: NUTS {res['seconds']:.0f}s ({res['chain_method']}), divergences {conv['divergences']}, "
              f"R-hat lam/bet/P {conv['lam']['rhat']:.3f}/{conv['bet']['rhat']:.3f}/{conv['P']['rhat']:.3f}, "
              f"ESS {conv['lam']['ess']:.0f}/{conv['bet']['ess']:.0f}/{conv['P']['ess']:.0f}")
        idata, loo = None, None
        try:
            idata = inf.to_inferencedata(res, der, m["name"])
            idata.to_netcdf(os.path.join(a.outdir, f"posterior_{naming.tag_of(m['name'])}.nc"))
        except Exception as e:
            print(f"[arviz] InferenceData export skipped ({e}); saving npz")
        np.savez_compressed(os.path.join(a.outdir, f"posterior_{naming.tag_of(m['name'])}.npz"),
                            x=res["x"], diverging=res["diverging"], x_map=np.asarray(x_map),
                            **{k: v for k, v in der.items() if k != "loglik"}, loglik=der["loglik"])
        try:
            import arviz as az
            lo = az.loo(idata)
            loo = dict(elpd=float(getattr(lo, "elpd", getattr(lo, "elpd_loo", np.nan))),
                       se=float(getattr(lo, "se", np.nan)), p_loo=float(getattr(lo, "p_loo", np.nan)),
                       pareto_k_max=float(np.max(np.asarray(getattr(lo, "pareto_k", np.nan)))))
        except Exception as e:
            print(f"[arviz] LOO skipped ({e})")
        # physical checks
        rng = np.random.default_rng(5)
        pick = rng.choice(len(xs), size=min(a.check_draws, len(xs)), replace=False)
        cv = [checks.convexity(M, xs[j]) for j in pick]
        sh = [checks.shadowing_error(M, xs[j], n_epochs=a.shadow_epochs) for j in pick[:a.shadow_draws]]
        sh = np.concatenate(sh) if sh else np.array([np.nan])
        chk = dict(max_protrusion_median=float(np.median([c["max_protrusion"] for c in cv])),
                   max_protrusion_max=float(np.max([c["max_protrusion"] for c in cv])),
                   area_frac_concave_median=float(np.median([c["area_frac_concave"] for c in cv])),
                   shadow_dmag_max=float(np.max(np.abs(sh))), shadow_dmag_rms=float(np.sqrt(np.mean(sh ** 2))))
        print(f"{m['name']}: convexity {chk['max_protrusion_median']:.4f} R_eq (median max protrusion), "
              f"shadowing error rms {1000 * chk['shadow_dmag_rms']:.2f} mmag, max {1000 * chk['shadow_dmag_max']:.2f} mmag")
        coh = None
        if rspec:
            sig = [cohesion.required_cohesion(*cohesion.semi_axes_from_moments(mo, rspec["diameter_km"]), P,
                                              rspec["density"], rspec["friction_deg"])
                   for mo, P in zip(der["moments"], der["P"])]
            coh = cohesion.summarise(sig, rspec)
        # ln Z in the convention of the convex / binary runs (band offsets and phase slope
        # profiled instead of marginalised under a flat prior): remove the marginalisation term
        corr = fq["offset_marginalisation_term"]
        runs.append(dict(name=m["name"], mode=m, model=M, x_map=x_map, lp_map=lp_map, map_info=info,
                         laplace={k: v for k, v in lap.items() if k != "cov"}, res=res, der=der,
                         conv=conv, loo=loo, checks=chk, fit_quality=fq, cohesion=coh,
                         lnZ_profiled=float(lap["lnZ"] - corr)))

    # ------------------------------------------------------------------ combination
    lz = np.array([r["laplace"]["lnZ"] for r in runs])
    w = np.exp(lz - lz.max())
    w /= w.sum()
    for r, wi in zip(runs, w):
        r["weight"] = float(wi)

    sens = None
    if a.sensitivity:
        settings = [("nu=1.5 (weaker damping)", dict(nu=1.5)), ("nu=3 (stronger damping)", dict(nu=3.0)),
                    ("sigma_f=0.08 (tighter)", dict(sigma_f=0.08))]
        sens = checks.prior_sensitivity(make_model, modes, settings,
                                        lambda M, s: md.find_map(M, s, n_iter=a.map_iter), md.laplace_evidence,
                                        starts=a.starts)
        for row in sens:
            print("sensitivity", row["setting"], [(r["mode"], round(r["lam"]), round(r["bet"]), round(r["weight"], 3)) for r in row["modes"]])

    # ------------------------------------------------------------------ report
    los = -np.mean(D["e_obs"], 0)
    los_l = float(np.degrees(np.arctan2(los[1], los[0])) % 360)
    los_b = float(np.degrees(np.arcsin(los[2] / np.linalg.norm(los))))
    report.pole_map(os.path.join(a.outdir, "fig_pole_hybrid.png"), runs, gpost, [(los_l, los_b), ((los_l + 180) % 360, -los_b)])
    report.traces(os.path.join(a.outdir, "fig_traces.png"), runs)
    report.diagnostics(os.path.join(a.outdir, "fig_diagnostics.png"), runs)
    report.radius_maps(os.path.join(a.outdir, "fig_radius_maps.png"), runs, grid)
    report.lightcurves(os.path.join(a.outdir, "fig_lightcurves_hybrid.png"), D, runs)
    try:
        from lcinv import folding
        folding.hybrid(os.path.join(a.outdir, "fig_folded_lightcurve_hybrid.png"), D, runs,
                       f"{gsumm['horizons_target'] if gsumm else a.designation} hybrid model")
    except Exception as e:  # plotting must never fail a finished run
        print("folded lightcurve plot skipped:", e)
    meta = dict(designation=gsumm["horizons_target"] if gsumm else a.designation, npts=int(len(D["t"])),
                nnights=int(D["night"].max() + 1), bands=D["bands"])
    for r in runs:
        report.html_viewer(os.path.join(a.outdir, f"viewer_{naming.tag_of(r['name'])}.html"), r, grid, meta)

    def q(v):
        return [float(x) for x in np.percentile(v, [16, 50, 84])]

    def pole_stats(d):
        l, b = np.radians(d["lam"]), np.radians(d["bet"])
        v = np.c_[np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)]
        mv = v.mean(0)
        mv /= np.linalg.norm(mv)
        ang = np.degrees(np.arccos(np.clip(v @ mv, -1, 1)))
        return dict(mean_pole=[float(np.degrees(np.arctan2(mv[1], mv[0])) % 360), float(np.degrees(np.arcsin(mv[2])))],
                    radius68=float(np.percentile(ang, 68)), radius95=float(np.percentile(ang, 95)))

    summary = dict(
        designation=meta["designation"], n_points=int(len(D["t"])), model="hybrid",
        data_fingerprint=fitstats.data_fingerprint(D),
        settings={k: v for k, v in vars(a).items()},
        grid=dict(nlat=grid.nlat, nlon=grid.nlon, lmax=grid.lmax, lmin=grid.lmin, k=grid.k,
                  orthonormality_error=orthonormality_error(grid)),
        period_prior=dict(P0=P0, sigma=sP),
        evidence=dict(lnZ_laplace_total=float(np.logaddexp.reduce(lz)),
                      lnZ_profiled_total=float(np.logaddexp.reduce([r["lnZ_profiled"] for r in runs])),
                      note="Laplace approximation summed over spin states; lnZ_profiled uses the "
                           "convex/binary convention (band offsets profiled), comparable with their ln Z"),
        spin_states=[dict(name=r["name"], global_center=[r["mode"]["lam"], r["mode"]["bet"]],
                    weight_global=r["mode"].get("weight_global"), weight_local_laplace=r["weight"],
                    lnZ_laplace=r["laplace"]["lnZ"], lnZ_profiled=r["lnZ_profiled"],
                    fit_quality=r["fit_quality"], cohesion=r["cohesion"],
                    laplace_nonpositive=r["laplace"]["n_nonpositive"],
                    map_logp=r["lp_map"], map_grad_norm=r["map_info"]["grad_norm"],
                    pole=pole_stats(r["der"]), P_h=q(r["der"]["P"]), jitter_mag=q(r["der"]["jitter"]),
                    sigma_f=q(r["der"]["sigma_f"]), ell=q(r["der"]["ell"]),
                    **({"c_lambert": q(r["der"]["c_L"])} if "c_L" in r["der"] else {}),
                    inertia_angle_deg=q(r["der"]["inertia_angle"]),
                    principal_moments_median=np.median(r["der"]["moments"], 0).tolist(),
                    moment_gap_top2=q(1 - r["der"]["moments"][:, 1]),
                    extents_median=np.median(r["der"]["extents"], 0).tolist(),
                    radius_sd_median_pct_Req=float(100 * np.median(r["der"]["r_mesh"].std(0))),
                    convergence=r["conv"], loo=r["loo"], checks=r["checks"],
                    nuts_seconds=r["res"]["seconds"]) for r in runs],
        sensitivity=sens, runtime_s=time.time() - T0)
    report.write_summary(os.path.join(a.outdir, "summary_hybrid.json"), summary)
    for r in runs:
        print(diagnostics.format_report(r["name"], r["conv"]["diagnostics"]))
    print(f"done in {time.time() - T0:.0f}s -> {a.outdir}")


if __name__ == "__main__":
    main()
