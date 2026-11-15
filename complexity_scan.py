#!/usr/bin/env python
"""Data-driven model complexity: how much shape detail do the data support?

  python complexity_scan.py results/566130_X05_cl --lmax 2 3 4 5 6           # convex run
  python complexity_scan.py results/566130_X05_cl --lmax 3 4 6 --tau0 0.3 0.6 1.2
  python complexity_scan.py results/566130_X05_hybrid --lmax 3 4 5 6 7       # hybrid run
  python complexity_scan.py results/566130_X05_cl --lmax 6 --full            # nested runs

The evidence decides: each complexity (spherical-harmonic degree lmax, and for the convex model
optionally the prior scale tau0 of the shape coefficients) is a separate model, and ln Z
already contains the Occam penalty for the extra coefficients. More detail is supported only
if ln Z increases.

Convex runs (no resampling). The spin-state posterior of the finished run is re-used:
    Z' / Z = E_post[ L'(theta) / L(theta) ]
(importance sampling with n posterior draws; L is the Laplace-marginalised likelihood, so each
lmax is fully marginalised over its own shape coefficients). The effective sample size of the
weights says whether this is reliable; when the posteriors at the two complexities differ a
lot (ESS fraction < 0.1), the estimate is biased low and a full run is needed (--full prints
or runs the commands, results in <outdir>_lmax<L>/, which compare_models.py picks up).

Hybrid runs: per spin state MAP + Laplace evidence of the radius-field model at each lmax
(the Matern hyperparameters sigma_f and ell are already inferred from the data within each
run, which is the hybrid model's own data-driven smoothing).

Outputs in the folder: complexity_scan.json, fig_complexity.png, and a table on screen.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcinv import rebuild  # noqa: E402


def reading(dz):
    """Jeffreys reading of an evidence difference (as compare_models.jeffreys)."""
    a = abs(dz)
    s = "not worth mentioning" if a < 1 else "weak" if a < 2.3 else "moderate" if a < 5 else "strong"
    return s


def logmeanexp(x):
    """Numerically stable log of the mean of exp(x)."""
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m)))


# ----------------------------------------------------------------------------------- convex
def _evaluate(M, post):
    """Laplace-marginalised ln L at each draw and the share of draws where the Laplace step is
    not trustworthy (shape optimum not converged or non-positive curvature)."""
    import jax.numpy as jnp
    ll, bad = [], 0
    for th in post:
        o = M.loglike_jit(jnp.asarray(th, float))
        v = float(o["logL"])
        ll.append(v if np.isfinite(v) else -np.inf)
        bad += int(float(o["min_eig"]) <= 0 or float(o["grad_norm"]) > 0.05)
    return np.array(ll), bad / len(post)


def scan_convex(outdir, lmaxes, tau0s, n, seed, n_inner=400):
    """Importance-sampling evidence of a convex run at other lmax and tau0, re-using its posterior."""
    from lcinv.convex import ConvexInversion
    D, M0, s, R = rebuild.global_run(outdir)
    lnZ0 = float(s["convergence"]["lnZ"])
    lnZ0_err = float(s["convergence"]["lnZ_err"])
    L0, t00 = int(R["args"]["lmax"]), float(R["args"]["tau0"])
    rng = np.random.default_rng(seed)
    post = R["post"][rng.choice(len(R["post"]), size=min(n, len(R["post"])), replace=False)]
    print(f"{outdir}: base run lmax = {L0}, tau0 = {t00:g}, ln Z = {lnZ0:.2f} +- {lnZ0_err:.2f}; "
          f"re-evaluating {len(post)} posterior draws per setting")
    if getattr(M0, "fit_tau", False) and tau0s:
        print("  the base run infers tau0 (--fit-tau): scanning tau0 is not needed, only lmax is scanned")
        tau0s = None
    l0, bad0 = _evaluate(M0, post)
    rows = []
    for L in lmaxes:
        for t0 in tau0s or [t00]:
            base = (L == L0 and abs(t0 - t00) < 1e-12)
            if base:
                M, l1, bad = M0, l0, bad0
            else:
                kw = dict(M0.init_kwargs)
                kw.update(lmax=int(L), tau0=float(t0), n_inner=max(int(n_inner), int(kw.get("n_inner", 80))))
                M = ConvexInversion(**kw)
                l1, bad = _evaluate(M, post)
            lw = l1 - l0
            ok = np.isfinite(lw)
            lw = lw[ok]
            w = np.exp(lw - lw.max())
            ess = float(w.sum() ** 2 / np.sum(w ** 2))
            lnZ = lnZ0 + logmeanexp(lw)
            boot = [logmeanexp(lw[rng.integers(0, len(lw), len(lw))]) for _ in range(200)]
            err = float(np.hypot(lnZ0_err, np.std(boot)))
            j = int(np.argmax(np.where(np.isfinite(l1), l1, -np.inf)))
            fq = M.fit_quality(post[j])
            laplace_ok = bad <= 0.05
            reliable = bool(base or (ess / len(lw) >= 0.1 and laplace_ok))
            status = ("ok" if reliable else
                      "Laplace step unreliable (raise --n-inner)" if not laplace_ok else
                      "lower bound: low ESS, the posterior moves; run --full")
            rows.append(dict(model="convex", lmax=int(L), tau0=float(t0), k_shape=int(M.k), lnZ=float(lnZ),
                             lnZ_err=err, method="nested run" if base else "importance sampling",
                             ess=ess, ess_fraction=ess / len(lw), frac_laplace_bad=float(bad),
                             reliable=reliable, lower_bound=bool(not reliable and laplace_ok), status=status,
                             n_inner=int(M.n_inner), fit_quality=fq))
            print(f"  lmax {L:2d} tau0 {t0:4.2f}: ln Z {lnZ:8.2f} +- {err:.2f}  ESS {ess:6.1f}/{len(lw)}  "
                  f"p_eff {fq['p_eff']:5.1f}  chi2_red {fq['chi2_red_nominal']:.2f}  rms {1000 * fq['rms_mag']:.1f} mmag"
                  + ("" if reliable else f"  [{status}]"))
    return rows, dict(base_lmax=L0, base_tau0=t00, base_lnZ=lnZ0, n_draws=len(post))


def full_commands(outdir, lmaxes, tau0s):
    """run_inversion.py commands reproducing the base run at other complexities."""
    import run_inversion
    from lcinv import sampler
    R = sampler.load(os.path.join(outdir, "nested_results.pkl"))
    a = dict(R["args"])
    p_defaults = vars(run_inversion.parse([a["lc"]]))
    cmds = []
    for L in lmaxes:
        for t0 in tau0s or [a["tau0"]]:
            if L == a["lmax"] and abs(t0 - a["tau0"]) < 1e-12:
                continue
            suffix = f"_lmax{L}" + ("" if abs(t0 - a["tau0"]) < 1e-12 else f"_tau{t0:g}")
            args = dict(a, lmax=L, tau0=t0, outdir=outdir.rstrip("/") + suffix, resume=False, extra_calls=0)
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_inversion.py"),
                   str(args.pop("lc"))]
            for k, v in args.items():
                if k not in p_defaults or v == p_defaults[k] or v is None:
                    continue
                flag = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag)
                elif isinstance(v, (list, tuple)):
                    cmd += [flag] + [str(x) for x in v]
                else:
                    cmd += [flag, str(v)]
            run_inversion.parse(cmd[2:])          # validate
            cmds.append(cmd)
    return cmds


# ----------------------------------------------------------------------------------- hybrid
def scan_hybrid(outdir, lmaxes, starts, n_iter):
    """MAP and Laplace evidence of each spin state of a hybrid run at other lmax."""
    from lcinv.hybrid import modes as md
    rows = []
    s0 = json.load(open(os.path.join(outdir, "summary_hybrid.json")))
    L0 = s0["settings"]["lmax"]
    for L in lmaxes:
        D, grid, states, s = rebuild.hybrid_run(outdir, lmax=L)
        lz, lzp, fqs, nbad = [], [], [], 0
        for st in states:
            M = st["model"]
            x0 = M.initial_points(starts)
            if L == L0:
                x0 = np.vstack([x0, rebuild.best_hybrid_point(st, refine=False)[None]])
            x, lp, info = md.find_map(M, x0, n_iter=n_iter)
            lap = md.laplace_evidence(M, x)
            fq = M.fit_quality(x)
            lz.append(lap["lnZ"])
            lzp.append(lap["lnZ"] - fq["offset_marginalisation_term"])
            fqs.append(fq)
            nbad += int(lap["n_nonpositive"] > 0)
            print(f"  lmax {L:2d} {st['name']}: ln Z_Laplace {lap['lnZ']:8.2f} (non-positive curvatures "
                  f"{lap['n_nonpositive']})  p_eff {fq['p_eff']:5.1f}  chi2_red {fq['chi2_red_nominal']:.2f}  "
                  f"rms {1000 * fq['rms_mag']:.1f} mmag")
        best = int(np.argmax(lz))
        rows.append(dict(model="hybrid", lmax=int(L), k_shape=int(states[0]["model"].k),
                         lnZ=float(np.logaddexp.reduce(lz)), lnZ_profiled=float(np.logaddexp.reduce(lzp)),
                         lnZ_err=np.nan, method="Laplace (MAP per spin state)", reliable=nbad == 0,
                         lower_bound=False,
                         status="ok" if nbad == 0 else "MAP not converged (non-positive curvature): raise --map-iter",
                         per_state=[dict(name=st["name"], lnZ=float(z)) for st, z in zip(states, lz)],
                         fit_quality=fqs[best]))
    return rows, dict(base_lmax=L0)


# ----------------------------------------------------------------------------------- output
def merge_previous(outdir, rows):
    """Keep rows of earlier scans of the same folder for settings not recomputed now."""
    path = os.path.join(outdir, "complexity_scan.json")
    if not os.path.exists(path):
        return rows
    key = lambda r: (r["model"], r["lmax"], None if r.get("tau0") is None else round(r["tau0"], 6))
    new = {key(r) for r in rows}
    old = [r for r in json.load(open(path)).get("rows", []) if key(r) not in new]
    return sorted(rows + old, key=lambda r: (r.get("tau0") or 0, r["lmax"]))


def finish(outdir, rows, meta):
    """Merge with earlier scans, rank the settings, print the table and write complexity_scan.json
    and the figure.
    """
    rows = merge_previous(outdir, rows)
    ok = [r for r in rows if r["reliable"]]
    if not ok:
        print("warning: no setting has a reliable evidence; the ranking below is indicative only")
        ok = rows
    lb = [r for r in rows if r.get("lower_bound")]
    zmax = max(r["lnZ"] for r in ok)
    for r in rows:
        r["delta_lnZ"] = r["lnZ"] - zmax
        r["reading_vs_best"] = "best" if r["delta_lnZ"] == 0 else reading(r["delta_lnZ"])
    p = np.exp(np.array([r["lnZ"] - zmax if r["reliable"] else -np.inf for r in rows]))
    p /= p.sum()
    for r, pi in zip(rows, p):
        r["probability"] = float(pi)
    best = max(ok, key=lambda r: r["lnZ"])
    key = (lambda r: (r["k_shape"], r.get("tau0", 0)))
    simplest = min([r for r in ok if r["delta_lnZ"] > -1.0], key=key)
    text = (f"the evidence is highest for lmax = {best['lmax']}"
            + (f", tau0 = {best['tau0']:g}" if best.get("tau0") is not None else "")
            + (f"; the simplest setting within 1 of it is lmax = {simplest['lmax']}"
               + (f", tau0 = {simplest['tau0']:g}" if simplest.get("tau0") is not None else "")
               if simplest is not best else "") + ".")
    better = [r for r in lb if r["lnZ"] - zmax > 2.3]
    if better:
        b = max(better, key=lambda r: r["lnZ"])
        text += (f" But {len(better)} setting(s) with only a lower bound already beat it by more than 2.3, "
                 f"up to {b['lnZ'] - zmax:+.1f} at lmax = {b['lmax']}"
                 + (f", tau0 = {b['tau0']:g}" if b.get("tau0") is not None else "")
                 + ": the data support more shape freedom than the base run allows. Confirm with a full run "
                   "(--full), or let the data choose tau0 with run_inversion.py --fit-tau.")
    elif any(not r["reliable"] for r in rows):
        text += " Settings marked unreliable need a full run (--full)."
    rec = dict(best=dict(lmax=best["lmax"], tau0=best.get("tau0")),
               simplest_within_1=dict(lmax=simplest["lmax"], tau0=simplest.get("tau0")), text=text)
    out = dict(rows=rows, meta=meta, recommendation=rec,
               note="ln Z includes the Occam penalty of each setting's shape coefficients; differences "
                    "are meaningful, absolute values are not")
    with open(os.path.join(outdir, "complexity_scan.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plot(os.path.join(outdir, "fig_complexity.png"), rows)
    print(f"\nrecommendation: {rec['text']}")
    print(f"-> {os.path.join(outdir, 'complexity_scan.json')}, fig_complexity.png")
    return out


def plot(path, rows):
    """Evidence against lmax (and tau0), fig_complexity.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lcinv.plots import _style, INK, MUTE
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    taus = sorted({r.get("tau0") for r in rows}, key=lambda x: (x is None, x))
    cols = ["#2a6fb0", "#c4452f", "#2a9d8f", "#8a5a2b"]
    for i, t in enumerate(taus):
        rr = sorted([r for r in rows if r.get("tau0") == t], key=lambda r: r["lmax"])
        L = [r["lmax"] for r in rr]
        lab = f"τ0 = {t:g}" if t is not None else "hybrid"
        ax[0].errorbar(L, [r["delta_lnZ"] for r in rr], yerr=[0 if not np.isfinite(r["lnZ_err"]) else r["lnZ_err"]
                                                             for r in rr], color=cols[i % 4], marker="o", ms=4,
                       lw=1.2, capsize=2, label=lab)
        for r in rr:
            if r.get("lower_bound"):
                ax[0].plot(r["lmax"], r["delta_lnZ"], "^", mfc="none", color=INK, ms=9)
            elif not r["reliable"]:
                ax[0].plot(r["lmax"], r["delta_lnZ"], "x", color=INK, ms=9)
        ax[1].plot(L, [r["fit_quality"]["p_eff"] for r in rr], color=cols[i % 4], marker="o", ms=4, lw=1.2, label=lab)
    ax[0].axhspan(-1, 0.2, color=MUTE, alpha=0.15, lw=0)
    ax[0].set_xlabel("spherical-harmonic degree lmax")
    ax[0].set_ylabel("ln Z - ln Z_best")
    ax[0].set_title("evidence (△ lower bound: low ESS;  × Laplace step not converged)", fontsize=9)
    ax[1].set_xlabel("spherical-harmonic degree lmax")
    ax[1].set_ylabel("effective parameters p_eff")
    ax[1].set_title("parameters the data constrain", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Command line of complexity_scan.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir",
                   help="finished convex (run_inversion.py) or hybrid (run_hybrid.py) results folder")
    p.add_argument("--lmax", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                   help="spherical-harmonic degrees to compare")
    p.add_argument("--tau0", type=float, nargs="+", default=None, help="convex: shape prior scales to scan")
    p.add_argument("--n", type=int, default=200, help="convex: posterior draws for importance sampling")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed of the posterior subsample")
    p.add_argument("--starts", type=int, default=4, help="hybrid: MAP starts per spin state")
    p.add_argument("--map-iter", type=int, default=1500, help="hybrid: L-BFGS iterations")
    p.add_argument("--n-inner", type=int, default=400,
                   help="convex: shape-optimisation iterations for the re-evaluated settings")
    p.add_argument("--full", action="store_true", help="convex: run full nested sampling for each setting")
    p.add_argument("--print-commands", action="store_true", help="convex: only print the full-run commands")
    a = p.parse_args()
    kind = rebuild.run_kind(a.outdir)
    if kind == "convex":
        if a.full or a.print_commands:
            cmds = full_commands(a.outdir, a.lmax, a.tau0)
            for c in cmds:
                print(" ".join(shlex.quote(x) for x in c))
                if a.full:
                    subprocess.check_call(c)
            if a.print_commands:
                return
        rows, meta = scan_convex(a.outdir, a.lmax, a.tau0, a.n, a.seed, a.n_inner)
    elif kind == "hybrid":
        rows, meta = scan_hybrid(a.outdir, a.lmax, a.starts, a.map_iter)
    else:
        raise SystemExit(f"{a.outdir}: not a convex (run_inversion.py) or hybrid (run_hybrid.py) results folder")
    finish(a.outdir, rows, meta)


if __name__ == "__main__":
    main()
