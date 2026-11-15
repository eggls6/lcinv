#!/usr/bin/env python
"""Phase-folded (0-4 pi) lightcurve figures for existing results folders, without resampling.

  python make_folded_plots.py results/566130_X05_cl              -> fig_folded_lightcurve.png
  python make_folded_plots.py results/K03Q21N_hybrid_quat        -> fig_folded_lightcurve_hybrid.png
                                                                     (via make_hybrid_report.py)
New runs of run_inversion.py / run_hybrid.py make these figures themselves.
"""
import glob
import json
import os
import sys

import numpy as np


def global_folded(outdir, n_draws=30, seed=7):
    """Rebuild fig_folded_lightcurve.png of a global run from its results folder."""
    from lcinv import ades, folding, sampler
    from lcinv.convex import ConvexInversion
    from lcinv.setup_data import prepare, drop_epochs
    s = json.load(open(os.path.join(outdir, "summary.json")))
    R = sampler.load(os.path.join(outdir, "nested_results.pkl"))
    a = R["args"]
    D = prepare(a["lc"], a["designation"], location=a.get("location", "X05"), cache_dir=a.get("cache_dir", "data"),
                drop_bands=tuple(b for b in str(a.get("drop_bands", "u")).split(",") if b), ades_opts=ades.opts_from_args(a),
                observer=a.get("observer", "station"), **ades.timing_from_args(a))
    D = drop_epochs(D, s.get("clipped_epochs", []))
    P0 = s["priors"]["period"][1]
    M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"],
                        lmax=a["lmax"], nfacets=a["nfacets"], n_inner=a["n_inner"], n_starts=a["n_starts"],
                        n_stage1=a["n_stage1"], tau0=a["tau0"], ab_prior=tuple(s["priors"]["ab"]),
                        bc_prior=tuple(s["priors"]["bc"]), exptime=D["exptime"],
                        smear_nodes=a.get("smear_nodes"), smear_P_h=P0,
                        c_lambert=a.get("c_lambert", 0.1), fit_cl=a.get("fit_cl", False))
    th = np.array(s["max_likelihood"]["theta"], float)
    o = M.loglike_jit(th)
    ens = R["ensemble"]
    pick = np.random.default_rng(seed).choice(len(ens["theta"]), size=min(n_draws, len(ens["theta"])), replace=False)
    path = os.path.join(outdir, "fig_folded_lightcurve.png")
    folding.convex(path, D, M, th, o, ens, pick, f"{s['horizons_target']} global convex model")
    print("wrote", path)


def main():
    """Command line: python make_folded_plots.py OUTDIR [OUTDIR ...] (global, binary or hybrid
    folders).
    """
    for outdir in sys.argv[1:]:
        sp = os.path.join(outdir, "summary.json")
        if os.path.exists(sp) and json.load(open(sp)).get("model") == "contact binary":
            import subprocess
            subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_binary.py"),
                             "--replot", "--outdir", outdir])
        elif os.path.exists(os.path.join(outdir, "nested_results.pkl")):
            global_folded(outdir)
        elif glob.glob(os.path.join(outdir, "posterior_spin_state_*.npz")) or \
                glob.glob(os.path.join(outdir, "posterior_mode*.npz")):
            import make_hybrid_report
            sys.argv = ["make_hybrid_report.py", outdir]
            make_hybrid_report.main()
        else:
            print(f"{outdir}: no nested_results.pkl or posterior_spin_state_*.npz", file=sys.stderr)


if __name__ == "__main__":
    main()
