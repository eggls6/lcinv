#!/usr/bin/env python
"""Honest comparison of all models fitted to one target, with a recommendation.

  python compare_models.py results/566130_X05_cl results/566130_X05_binary results/566130_X05_hybrid
  python compare_models.py results/566130_X05*  --out results/566130_comparison

Every folder may be a convex run (run_inversion.py), a contact-binary run (run_binary.py) or
a hybrid run (run_hybrid.py); a complexity_scan.json inside a convex folder adds its rows.

For each model the table gives
  - goodness of fit of the best fit: N, effective number of parameters p_eff, reduced chi^2
    with the quoted errors and with the fitted jitter, residual rms (lcinv.fitstats);
  - the evidence ln Z, its difference to the best model and the Jeffreys / Kass-Raftery
    reading, and the posterior probability of each model for equal prior odds;
  - convergence, the evidence method (nested sampling, importance sampling, Laplace) and
    the required cohesion (Sanchez & Scheeres 2014) where a diameter was given.

Only runs on identical data (same data fingerprint: epochs, magnitudes, errors, bands) are
compared; others are listed but excluded. Evidences from importance sampling with low ESS are
lower bounds and are not used for probabilities, but they enter the recommendation. Laplace
evidences (hybrid) are approximate: an uncertainty of 2.0 is assumed (two runs of the same hybrid
model in different pole parametrisations differ by ~3).

Outputs: <out>.md (table + recommendation), <out>.csv, <out>.json (default <first folder>/model_comparison).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcinv import rebuild, fitstats  # noqa: E402

LAPLACE_SYS = 2.0


def jeffreys(dz):
    """Jeffreys / Kass-Raftery reading of an evidence difference: 'not worth mentioning' (< 1),
    'weak' (< 2.3), 'moderate' (< 5), 'strong'.
    """
    a = abs(dz)
    return "not worth mentioning" if a < 1 else "weak" if a < 2.3 else "moderate" if a < 5 else "strong"


def _verdict(conv):
    return (conv or {}).get("verdict") or (conv or {}).get("diagnostics", {}).get("verdict")


# ------------------------------------------------------------------------------- collectors
def _store(path, _replace=None, **kw):
    """Add recomputed statistics to an older run's summary so they are computed only once."""
    try:
        s = _replace if _replace is not None else json.load(open(path))
        s.update(kw)
        with open(path + ".tmp", "w") as f:
            json.dump(s, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(path + ".tmp", path)
    except Exception as e:  # noqa: BLE001 - caching only
        print(f"note: could not update {path} ({e})", file=sys.stderr)
def convex_rows(d, recompute=True):
    """Table rows of a convex run, plus one per setting of a complexity_scan.json inside it."""
    s = json.load(open(os.path.join(d, "summary.json")))
    from lcinv import sampler
    R = sampler.load(os.path.join(d, "nested_results.pkl"))
    a = R["args"]
    fq, fp = s.get("fit_quality"), s.get("data_fingerprint")
    if recompute and (fq is None or fp is None):
        D, M, _, _ = rebuild.global_run(d)
        fq = fq or M.fit_quality(np.array(s["max_likelihood"]["theta"], float))
        fp = fp or fitstats.data_fingerprint(D)
        _store(os.path.join(d, "summary.json"), fit_quality=fq, data_fingerprint=fp)
    cl = "c_L fit" if a.get("fit_cl") else f"c_L {a.get('c_lambert', 0.1):g}"
    tau = "τ0 inferred" if a.get("fit_tau") else f"τ0 {a['tau0']:g}"
    smear = f", {a['exptime']:g} s exp." if a.get("exptime") else ""
    label = f"convex lmax {a['lmax']}, {tau}, {cl}{smear}"
    p = s["posterior"]
    conv = s.get("convergence")
    if conv is None:                                # runs from before the convergence report
        from lcinv import diagnostics
        sm = s["sampler"]
        conv = diagnostics.nested(R["logwt"], R["logz"], R["logzerr"], sm["nlive"],
                                  beta=np.asarray(R["samples"])[:, 1])
    row = dict(folder=d, model="convex", label=label, method="nested sampling",
               lnZ=conv["lnZ"], lnZ_err=conv["lnZ_err"],
               verdict=_verdict(conv), fit_quality=fq, data_fingerprint=fp,
               jitter_post=p.get("jitter_mag"), cohesion=s.get("cohesion"), reliable=True, lower_bound=False,
               extra=dict(tau0=p.get("tau0"), c_lambert=p.get("c_lambert"), pole_sky68=p.get("pole_hpd_sky_fraction", {}).get("68%")))
    rows = [row]
    cs = os.path.join(d, "complexity_scan.json")
    if os.path.exists(cs):
        for r in json.load(open(cs))["rows"]:
            if r["method"] == "nested run":
                continue
            rows.append(dict(folder=d, model="convex", label=f"convex lmax {r['lmax']}, τ0 {r['tau0']:g}, {cl}{smear} [scan]",
                             method=f"importance sampling (ESS {r['ess']:.0f})", lnZ=r["lnZ"], lnZ_err=r["lnZ_err"],
                             verdict=None, fit_quality=r["fit_quality"], data_fingerprint=fp, jitter_post=None,
                             cohesion=None, reliable=bool(r["reliable"]), lower_bound=bool(r.get("lower_bound")),
                             status=r.get("status"), extra={}))
    return rows


def binary_rows(d, recompute=True):
    """Table row of a contact-binary run."""
    s = json.load(open(os.path.join(d, "summary.json")))
    fq, fp = s.get("fit_quality"), s.get("data_fingerprint")
    if recompute and (fq is None or fp is None):
        D, M, _, _ = rebuild.binary_run(d)
        fq = fq or M.fit_quality(np.array(s["max_likelihood"]["theta"], float))
        fp = fp or fitstats.data_fingerprint(D)
        _store(os.path.join(d, "summary.json"), fit_quality=fq, data_fingerprint=fp)
    p = s["posterior"]
    note = ""
    pr = s.get("priors", {})
    if pr.get("q") and p["q"][0] - pr["q"][0] < 0.05 * (pr["q"][1] - pr["q"][0]):
        note = f"q at its prior bound ({pr['q'][0]:g})"
    return [dict(folder=d, model="contact binary", label=f"contact binary (q = {p['q'][1]:.2f}, s = {p['s'][1]:.2f})",
                 method="nested sampling", lnZ=s["convergence"]["lnZ"], lnZ_err=s["convergence"]["lnZ_err"],
                 verdict=_verdict(s.get("convergence")), fit_quality=fq, data_fingerprint=fp,
                 jitter_post=p.get("jitter_mag"), cohesion=s.get("cohesion"), reliable=True, lower_bound=False,
                 extra=dict(note=note) if note else {})]


def hybrid_rows(d, recompute=True):
    """Table rows of a hybrid run, one per spin state plus the combined model."""
    from lcinv.hybrid import naming
    s = json.load(open(os.path.join(d, "summary_hybrid.json")))
    states = naming.states(s)
    fp = s.get("data_fingerprint")
    need = recompute and (fp is None or any("fit_quality" not in st for st in states))
    if need:
        D, grid, built, _ = rebuild.hybrid_run(d)
        fp = fp or fitstats.data_fingerprint(D)
        for st, b in zip(states, built):
            if "fit_quality" not in st:
                x = rebuild.best_hybrid_point(b)
                st["fit_quality"] = b["model"].fit_quality(x)
                st["lnZ_profiled"] = st["lnZ_laplace"] - st["fit_quality"]["offset_marginalisation_term"]
        s["data_fingerprint"] = fp
        _store(os.path.join(d, "summary_hybrid.json"), _replace=s)
    lzp = [st.get("lnZ_profiled", np.nan) for st in states]
    for k, f in naming.posterior_files(d):          # verdicts of runs before the diagnostics report
        st = naming.state_info(s, k)
        if st and not _verdict(st.get("convergence")):
            from lcinv import diagnostics
            z = np.load(f)
            st.setdefault("convergence", {})["diagnostics"] = diagnostics.nuts(
                z["x"], z["diverging"], {"P": z["P"], "jitter": z["jitter"]})
    total = float(np.logaddexp.reduce(lzp)) if np.all(np.isfinite(lzp)) else np.nan
    verd = [_verdict(st.get("convergence")) for st in states]
    best = int(np.nanargmax(lzp)) if np.any(np.isfinite(lzp)) else 0
    lmax = s.get("grid", {}).get("lmax", s.get("settings", {}).get("lmax"))
    rows = [dict(folder=d, model="hybrid", label=f"hybrid star-shaped, lmax {lmax} ({len(states)} spin state(s))",
                 method="Laplace (per spin state, summed)", lnZ=total, lnZ_err=LAPLACE_SYS,
                 verdict="CONVERGED" if all(v == "CONVERGED" for v in verd) else (verd[best] if verd else None),
                 fit_quality=states[best].get("fit_quality") if states else None, data_fingerprint=fp,
                 jitter_post=states[best].get("jitter_mag") if states else None,
                 cohesion=states[best].get("cohesion") if states else None, reliable=bool(np.isfinite(total)),
                 lower_bound=False,
                 extra=dict(spin_states=[dict(name=st["name"], weight=st.get("weight_local_laplace"),
                                              pole=(st.get("pole") or {}).get("mean_pole"), verdict=v)
                                         for st, v in zip(states, verd)]))]
    return rows


def collect(folders, recompute=True):
    """Rows of all folders; unreadable folders are reported and skipped."""
    rows = []
    for d in folders:
        kind = rebuild.run_kind(d)
        try:
            if kind == "convex":
                rows += convex_rows(d, recompute)
            elif kind == "binary":
                rows += binary_rows(d, recompute)
            elif kind == "hybrid":
                rows += hybrid_rows(d, recompute)
            else:
                print(f"{d}: not a results folder, skipped", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - one bad folder must not stop the comparison
            print(f"{d}: skipped ({e})", file=sys.stderr)
    return rows


# ------------------------------------------------------------------------------- analysis
def _unique_labels(rows):
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    for r in rows:
        if c[r["label"]] > 1:
            r["label"] += f" [{os.path.basename(r['folder'].rstrip('/'))}]"


def analyse(rows, n_mc=4000, seed=0):
    """Mark comparable rows (same data fingerprint as the majority), then compute evidence
    differences and model probabilities with their Monte Carlo errors.
    """
    _unique_labels(rows)
    fps = [r["data_fingerprint"] for r in rows if r["data_fingerprint"]]
    ref = max(set(fps), key=fps.count) if fps else None
    for r in rows:
        r["comparable"] = (r["data_fingerprint"] == ref) if ref else True
    use = [r for r in rows if r["comparable"] and r["reliable"] and np.isfinite(r["lnZ"])]
    if not use:
        raise SystemExit("no comparable evidences")
    zmax = max(r["lnZ"] for r in use)
    for r in rows:
        r["delta_lnZ"] = r["lnZ"] - zmax if np.isfinite(r["lnZ"]) else np.nan
    z = np.array([r["lnZ"] for r in use])
    p = np.exp(z - z.max())
    p /= p.sum()
    sig = np.array([r["lnZ_err"] if np.isfinite(r["lnZ_err"]) else LAPLACE_SYS for r in use])
    rng = np.random.default_rng(seed)
    zs = z[None] + rng.standard_normal((n_mc, len(z))) * sig[None]
    pbest = np.bincount(np.argmax(zs, 1), minlength=len(z)) / n_mc
    for r, pi, pb in zip(use, p, pbest):
        r["probability"] = float(pi)
        r["p_best"] = float(pb)
    for r in rows:
        r.setdefault("probability", None)
        r.setdefault("p_best", None)
        dz = r.get("delta_lnZ", np.nan)
        if not np.isfinite(dz):
            r["reading"] = "n/a"
        elif not r["reliable"] and not r["lower_bound"]:
            r["reading"] = "unreliable (Laplace step)"
        elif r["lower_bound"]:
            r["reading"] = (f"≥ {jeffreys(dz)} (better)" if dz > 1 else "inconclusive (lower bound)")
        elif r in use and dz == 0:
            r["reading"] = "best"
        else:
            r["reading"] = jeffreys(dz) + (" (better)" if dz > 0 else " (worse)")
    return use, ref


def recommend(rows, use):
    """Most probable model and the recommendation text."""
    order = sorted(use, key=lambda r: -r["lnZ"])
    best = order[0]
    lines = []
    if len(order) > 1:
        d2 = best["lnZ"] - order[1]["lnZ"]
        s2 = np.hypot(best["lnZ_err"], order[1]["lnZ_err"])
        if d2 < 1:
            lines.append(f"**No decision** between *{best['label']}* and *{order[1]['label']}* "
                         f"(Δ ln Z = {d2:.1f} ± {s2:.1f}); report both.")
        elif d2 < 2.3:
            lines.append(f"*{best['label']}* is weakly preferred over *{order[1]['label']}* "
                         f"(Δ ln Z = {d2:.1f} ± {s2:.1f}, odds ≈ {np.exp(d2):.0f}:1): not decisive.")
        elif d2 < 5:
            lines.append(f"*{best['label']}* is moderately favoured over *{order[1]['label']}* "
                         f"(Δ ln Z = {d2:.1f} ± {s2:.1f}, odds ≈ {np.exp(d2):.0f}:1).")
        else:
            lines.append(f"*{best['label']}* is strongly favoured: Δ ln Z = {d2:.1f} ± {s2:.1f} over the next "
                         f"model (*{order[1]['label']}*), odds > {min(np.exp(d2), 1e6):.0f}:1.")
    else:
        lines.append(f"Only one comparable model: *{best['label']}*.")
    if best["verdict"] not in (None, "CONVERGED"):
        lines.append(f"Caution: the best model's run is **{best['verdict']}**; rerun it to convergence before "
                     "trusting the comparison.")
    if best["method"].startswith(("Laplace", "importance")):
        lines.append(f"Caution: the best model's evidence is from {best['method']}; confirm it with a full nested run.")
    lb = [r for r in rows if r["lower_bound"] and r["comparable"] and r["lnZ"] - best["lnZ"] > 2.3]
    if lb:
        top = max(lb, key=lambda r: r["lnZ"])
        lines.append(f"A setting with only a lower bound on ln Z already exceeds the best by "
                     f"{top['lnZ'] - best['lnZ']:+.1f}: *{top['label']}*. The data support more shape freedom; "
                     "run it fully (complexity_scan.py --full, or run_inversion.py --fit-tau) and compare again.")
    fq = best.get("fit_quality") or {}
    if fq and fq.get("chi2_red_nominal", 0) > 2:
        lines.append(f"Fit adequacy: χ²_red = {fq['chi2_red_nominal']:.1f} with the quoted errors "
                     f"(median {1000 * fq['median_err_mag']:.1f} mmag); the model needs {1000 * fq['jitter_mag']:.0f} mmag "
                     "extra scatter. Either the errors are underestimated or there is signal no model here "
                     "captures (shape detail, scattering, non-principal-axis rotation, calibration).")
    if fq:
        lines.append(f"The data constrain p_eff ≈ {fq['p_eff']:.0f} parameters of the best model "
                     f"(nominal {fq['p_nominal']}).")
    ch = best.get("cohesion") or {}
    if ch.get("required_cohesion_Pa"):
        c = ch["required_cohesion_Pa"]
        txt = (f"Required cohesion of the best model (Sánchez & Scheeres 2014, D = {ch['diameter_km']:g} km, "
               f"ρ = {ch['density']:g} kg/m³): {c[1]:.3g} Pa (68%: {c[0]:.3g}–{c[2]:.3g})")
        if ch.get("cohesion_max_Pa"):
            txt += f"; {100 * ch['frac_within_limit']:.0f}% of draws within the {ch['cohesion_max_Pa']:g} Pa limit"
        lines.append(txt + ".")
    nc = [r for r in rows if not r["comparable"]]
    if nc:
        lines.append(f"{len(nc)} run(s) used different data and were not compared: "
                     + ", ".join(os.path.basename(r["folder"].rstrip("/")) for r in nc) + ".")
    return best, lines


# ------------------------------------------------------------------------------- output
COLS = [("model", "Model"), ("method", "ln Z method"), ("N", "N"), ("p_eff", "p_eff"),
        ("chi2n", "χ²_red (errors)"), ("chi2j", "χ²_red (jitter)"), ("rms", "rms [mmag]"), ("jit", "jitter [mmag]"),
        ("lnZ", "ln Z"), ("dlnZ", "Δ ln Z"), ("reading", "reading"), ("prob", "P(model)"), ("pbest", "P(best)"),
        ("verdict", "converged")]


def flat(r):
    """Printable fields of one row."""
    fq = r.get("fit_quality") or {}
    f = lambda v, d=1: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
    jp = r.get("jitter_post")
    jit = 1000 * (jp[1] if isinstance(jp, (list, tuple)) else fq.get("jitter_mag", np.nan)) if (jp or fq) else None
    lnz = ("≥ " if r["lower_bound"] else "") + f(r["lnZ"], 2) + (f" ± {r['lnZ_err']:.2f}" if np.isfinite(r["lnZ_err"]) else "")
    return dict(model=r["label"], method=r["method"], N=fq.get("n_points", "–"), p_eff=f(fq.get("p_eff")),
                chi2n=f(fq.get("chi2_red_nominal"), 2), chi2j=f(fq.get("chi2_red_jitter"), 2),
                rms=f(1000 * fq["rms_mag"] if "rms_mag" in fq else None), jit=f(jit),
                lnZ=lnz if r["comparable"] else lnz + " (other data)",
                dlnZ=f(r.get("delta_lnZ"), 2) if r["comparable"] else "–", reading=r["reading"] if r["comparable"] else "not comparable",
                prob=f(r["probability"], 3) if r["probability"] is not None else "–",
                pbest=f(r["p_best"], 2) if r["p_best"] is not None else "–",
                verdict=r["verdict"] or ("–" if r["method"].startswith("import") else "?"))


def write(out, rows, best, lines, ref):
    """Write the report as <out>.md, <out>.csv and <out>.json."""
    import csv
    rows = sorted(rows, key=lambda r: (not r["comparable"], -(r["lnZ"] if np.isfinite(r["lnZ"]) else -1e99)))
    invalid = [r for r in rows if not r["reliable"] and not r["lower_bound"]]
    rows_all = rows
    rows = [r for r in rows if r not in invalid]
    fl = [flat(r) for r in rows]
    md = ["# Model comparison", "", f"Data fingerprint `{ref}` · equal prior odds · Jeffreys / Kass–Raftery scale "
          "(|Δ ln Z| < 1 not worth mentioning, 1–2.3 weak, 2.3–5 moderate, > 5 strong)", "",
          "| " + " | ".join(h for _, h in COLS) + " |", "|" + "---|" * len(COLS)]
    md += ["| " + " | ".join(str(x[k]) for k, _ in COLS) + " |" for x in fl]
    if invalid:
        md += ["", "Not shown (Laplace step of the importance-sampling scan not converged, rerun with a larger "
               "--n-inner or fully): " + "; ".join(r["label"] for r in invalid)]
    md += ["", "## Recommendation", "", f"**Most probable model: {best['label']}**", ""] + [f"- {l}" for l in lines]
    for r in rows:
        ss = r.get("extra", {}).get("spin_states")
        if ss:
            md += ["", f"Spin states of *{r['label']}*: " + "; ".join(
                f"{s['name']} (pole {s['pole'][0]:.0f}°, {s['pole'][1]:.0f}°, weight {s['weight']:.2f}, {s['verdict']})"
                if s.get("pole") else f"{s['name']} (weight {s['weight']})" for s in ss)]
    md += ["", "Notes: ln Z values are only meaningful as differences between runs on the same data. "
           "χ²_red uses dof = N − p_eff, where p_eff = tr(H_lik H_post⁻¹) counts the parameters the data "
           "actually constrain (shape coefficients shrunk by their prior count less than 1). Importance-sampled "
           "rows with low ESS are lower bounds (≥) and get no probability. Hybrid evidences are Laplace "
           "approximations in the profiled-offset convention of the convex/binary runs (±2 assumed)."]
    with open(out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[h for _, h in COLS] + ["folder"])
        w.writeheader()
        for x, r in zip(fl, rows):
            w.writerow({**{h: x[k] for k, h in COLS}, "folder": r["folder"]})
    clean = lambda o: o.item() if hasattr(o, "item") else (None if isinstance(o, float) and not np.isfinite(o) else str(o))
    with open(out + ".json", "w") as f:
        json.dump(dict(rows=rows_all, best=best["label"], recommendation=lines, data_fingerprint=ref), f, indent=1,
                  default=clean)
    return "\n".join(md)


def compare(folders, out=None, recompute=True, quiet=False):
    """Compare the models in `folders` and write the report; returns the rows and the
    recommendation.
    """
    rows = collect(folders, recompute)
    if not rows:
        raise SystemExit("nothing to compare")
    use, ref = analyse(rows)
    best, lines = recommend(rows, use)
    out = out or os.path.join(folders[0], "model_comparison")
    txt = write(out, rows, best, lines, ref)
    if not quiet:
        print(txt)
        print(f"\n-> {out}.md / .csv / .json")
    return dict(best=best, lines=lines, rows=rows, out=out)


def main():
    """Command line of compare_models.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+",
                   help="results folders (convex, binary or hybrid runs) of the same target")
    p.add_argument("--out", default=None, help="output path without extension")
    p.add_argument("--no-recompute", action="store_true",
                   help="do not rebuild models for old runs without stored fit statistics")
    a = p.parse_args()
    compare(a.folders, a.out, recompute=not a.no_recompute)


if __name__ == "__main__":
    main()
