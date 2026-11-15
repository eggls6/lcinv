#!/usr/bin/env python
"""Importance-reweight posterior samples with a finer facet discretisation.

  python reweight_resolution.py results/K03Q21N --nfacets 800 --n 300

The production run uses 400 Gaussian-image facets. Here equal-weight posterior samples are
re-evaluated with a finer model, w_i ~ exp(lnL_fine - lnL_coarse), and the effective sample
size and the shift of pole/period summaries are reported. A high ESS with negligible shifts
means the discretisation does not affect the posterior.
"""
import argparse
import json
import os

import numpy as np
from lcinv import ades

from lcinv import sampler
from lcinv.convex import ConvexInversion
from lcinv.setup_data import prepare


def pole_summary(post, w):
    """Weighted prograde fraction, period mean and sd, and mean pole of each hemisphere."""
    w = w / w.sum()
    lam, bet = np.radians(post[:, 0]), np.radians(post[:, 1])
    v = np.c_[np.cos(bet) * np.cos(lam), np.cos(bet) * np.sin(lam), np.sin(bet)]
    pro = float(np.sum(w * (post[:, 1] > 0)))
    out = dict(frac_prograde=pro, P_mean=float(np.sum(w * post[:, 2])),
               P_std=float(np.sqrt(np.sum(w * (post[:, 2] - np.sum(w * post[:, 2])) ** 2))))
    for name, m in (("prograde", post[:, 1] > 0), ("retrograde", post[:, 1] <= 0)):
        if np.sum(w[m]) > 0.02:
            mv = np.sum(w[m, None] * v[m], 0)
            mv /= np.linalg.norm(mv)
            out[f"{name}_mean_pole"] = [float(np.degrees(np.arctan2(mv[1], mv[0])) % 360),
                                        float(np.degrees(np.arcsin(mv[2])))]
    return out


def main():
    """Command line of reweight_resolution.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir",
                   help="finished run_inversion.py folder")
    p.add_argument("--nfacets", type=int, default=800,
                   help="facets of the finer model")
    p.add_argument("--n", type=int, default=300,
                   help="posterior samples to reweight")
    a = p.parse_args()
    R = sampler.load(os.path.join(a.outdir, "nested_results.pkl"))
    args = R["args"]
    summ = json.load(open(os.path.join(a.outdir, "summary.json")))
    D = prepare(args["lc"], args["designation"], location=args["location"], cache_dir=args["cache_dir"],
                drop_bands=tuple(b for b in args["drop_bands"].split(",") if b),
                ades_opts=ades.opts_from_args(args), observer=args.get("observer", "station"),
                **ades.timing_from_args(args))
    pri = summ["priors"]
    kw = dict(lmax=args["lmax"], tau0=args["tau0"], n_inner=args["n_inner"], n_starts=args["n_starts"],
              n_stage1=args["n_stage1"], ab_prior=tuple(pri["ab"]) if pri["ab"] else None,
              bc_prior=tuple(pri["bc"]) if pri["bc"] else None,
              exptime=D["exptime"], smear_nodes=args.get("smear_nodes"),
              smear_P_h=summ["period_search"]["P_best_h"],
              c_lambert=args.get("c_lambert", 0.1), fit_cl=args.get("fit_cl", False))
    base = (D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"])
    Mc = ConvexInversion(*base, nfacets=args["nfacets"], **kw)
    Mf = ConvexInversion(*base, nfacets=a.nfacets, **kw)
    post = R["post"][np.random.default_rng(0).choice(len(R["post"]), size=min(a.n, len(R["post"])), replace=False)]
    lc = np.array([Mc.loglike(th) for th in post])
    lf = np.array([Mf.loglike(th) for th in post])
    lw = lf - lc
    w = np.exp(lw - lw.max())
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    res = dict(nfacets_coarse=args["nfacets"], nfacets_fine=a.nfacets, n=len(post),
               ess=ess, ess_fraction=ess / len(post), dlnL_std=float(np.std(lw)),
               coarse=pole_summary(post, np.ones(len(post))), fine=pole_summary(post, w))
    with open(os.path.join(a.outdir, "resolution_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
