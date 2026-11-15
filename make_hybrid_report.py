#!/usr/bin/env python
"""Rebuild the hybrid report (figures + per-spin-state HTML viewers) from saved posteriors.

  python make_hybrid_report.py results/K03Q21N_hybrid [--global-dir results/K03Q21N]

Reads posterior_spin_state_*.npz (or legacy posterior_mode*.npz) and, if present,
summary_hybrid.json (for spin-state weights and
settings). Useful after a run was interrupted or to restyle figures without resampling.
"""
import argparse
import json
import os

import numpy as np

from lcinv import sampler as gsampler
from lcinv.setup_data import prepare
from lcinv.hybrid.quadrature import make_grid
from lcinv.hybrid.shape import RadiusField
from lcinv.hybrid.model import LocalModel, lonlat_to_vec
from lcinv.hybrid import report, naming


def main():
    """Command line of make_hybrid_report.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir",
                   help="hybrid results folder (run_hybrid.py)")
    p.add_argument("--global-dir", default=None,
                   help="global run, for the global pole posterior (default: from summary_hybrid.json)")
    p.add_argument("--lc", default=None,
                   help="lightcurve file (default: from summary_hybrid.json)")
    p.add_argument("--designation", default=None,
                   help="target designation (default: from summary_hybrid.json)")
    p.add_argument("--lmax", type=int, default=5,
                   help="radius-field degree (default: from summary_hybrid.json)")
    p.add_argument("--nlat", type=int, default=16,
                   help="grid latitude nodes (default: from summary_hybrid.json)")
    p.add_argument("--period", type=float, default=None,
                   help="period prior centre [h] (default: from summary_hybrid.json)")
    p.add_argument("--period-sigma", type=float, default=None,
                   help="period prior width [h] (default: from summary_hybrid.json)")
    a = p.parse_args()

    summ = None
    sp = os.path.join(a.outdir, "summary_hybrid.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))
        s = summ["settings"]
        a.lmax, a.nlat = s["lmax"], s["nlat"]
        a.lc, a.designation = s["lc"], s["designation"]
        a.global_dir = a.global_dir or s.get("global_dir")
    elif a.lc is None or a.designation is None:
        p.error("without summary_hybrid.json in OUTDIR, --lc and --designation are required")
    gsumm, gpost = None, None
    if a.global_dir:
        gsumm = json.load(open(os.path.join(a.global_dir, "summary.json")))
        gpost = gsampler.load(os.path.join(a.global_dir, "nested_results.pkl"))["post"]
    P0 = a.period or (summ["period_prior"]["P0"] if summ else gsumm["period_search"]["P_best_h"])
    sP = a.period_sigma or (summ["period_prior"]["sigma"] if summ else
                            0.5 * np.diff(gsumm["period_search"]["threshold_interval"])[0])

    from lcinv import ades
    st = summ["settings"] if summ else {}
    D = prepare(a.lc, a.designation, location=st.get("location", "X05"), cache_dir=st.get("cache_dir", "data"),
                ades_opts=ades.opts_from_args(st), observer=st.get("observer", "station"),
                **ades.timing_from_args(st))
    if gsumm and gsumm.get("clipped_epochs"):
        from lcinv.setup_data import drop_epochs
        D = drop_epochs(D, gsumm["clipped_epochs"])
    grid = make_grid(a.lmax, nlat=a.nlat, nlon=2 * a.nlat)
    files = naming.posterior_files(a.outdir)
    runs = []
    for idx, f in files:
        z = np.load(f)
        name = naming.name(idx)
        der = {key: z[key] for key in z.files if key not in ("x", "diverging")}
        res = dict(x=z["x"], diverging=z["diverging"])
        # the sampled pole coordinates are relative to the mode's chart centre, so the model
        # must be rebuilt with the centre used in the run (summary_hybrid.json)
        gc = naming.state_info(summ, idx).get("global_center")
        if gc is not None:
            lam, bet = gc
        else:
            print(f"warning: {name}: chart centre not stored; using the posterior mean pole (curves may be off)")
            lam = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(der["lam"])))))
            bet = float(np.median(der["bet"]))
        M = LocalModel(D, RadiusField(grid), lonlat_to_vec(lam, bet), P0, sP, frame=st.get("frame") or "euler",
                       smear_nodes=st.get("smear_nodes"), c_lambert=st.get("c_lambert", 0.1),
                       fit_cl=st.get("fit_cl", False),
                       ln_cl_prior=(np.log(st.get("cl_prior", [0.3, 2.0])[0]), st.get("cl_prior", [0.3, 2.0])[1]))
        w = np.nan
        if summ:
            w = naming.state_info(summ, idx).get("weight_local_laplace", np.nan)
        runs.append(dict(name=name, model=M, der=der, res=res, weight=float(w) if np.isfinite(w) else 1.0 / len(files)))
    los = -np.mean(D["e_obs"], 0)
    los_l = float(np.degrees(np.arctan2(los[1], los[0])) % 360)
    los_b = float(np.degrees(np.arcsin(los[2] / np.linalg.norm(los))))
    report.pole_map(os.path.join(a.outdir, "fig_pole_hybrid.png"), runs, gpost,
                    [(los_l, los_b), ((los_l + 180) % 360, -los_b)])
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
    print("report rebuilt for", [r["name"] for r in runs])


if __name__ == "__main__":
    main()
