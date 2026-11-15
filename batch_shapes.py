#!/usr/bin/env python
"""Batch shape modelling for a table of targets (pandas DataFrame).

For every row the pipeline runs, in its own folder <outroot>/<num>/:

    1. run_inversion.py     global convex inversion (period search in a window around the
                            table period, nested sampling)          -> global/
    2. run_binary.py        contact-binary model + evidence vs convex (optional, --binary) -> binary/
       run_hybrid.py        hybrid star-shaped model (optional, --hybrid)   -> hybrid/
    3. check_convergence.py convergence report                        -> convergence.txt
    4. export_obj.py        OBJ meshes, scaled to the table diameter      -> global/obj, hybrid/obj
    5. make_viewer.py       transparent-overlay 3D viewers (and hybrid radius viewers)
    6. complexity_scan.py   evidence vs shape complexity (optional, --complexity)
    7. compare_models.py    model comparison table + recommendation -> model_comparison.md

and finally writes <outroot>/batch_summary.csv with one row per target (input columns +
fitted period, pole, jitter, ln Z, convergence verdict, status, paths).

Command line
------------
    python batch_shapes.py targets.csv --data-pattern "data/{num}_ades.csv" \\
        --common-args "--stations X05 --observer local --exptime 30" \\
        --outroot results/batch --jobs 2 --npool 4 --hybrid

The table can be CSV, Parquet, Feather, pickle or Excel. From Python / a notebook:

    from batch_shapes import run_batch
    summary = run_batch(df, data_pattern="data/{num}_ades.csv", hybrid=True, jobs=2)

Columns used
------------
    num            target name, e.g. 2013_DV23 or 722574 (required; '_' -> ' ' for Horizons)
    period_h       period from the period search [h] (required)
    H / H_V        absolute magnitude for the size estimate (else looked up in JPL SBDB);
                   D = 10^(3.1236 - 0.5 log10 p - 0.2 H) km (CNEOS), p = --albedo (0.15)
    diameter_km    used instead with --size-source table, or when no H is available
                   The size scales the OBJ files and enables the cohesion report / --cohesion-max
    density        bulk density [kg/m^3] for the cohesion calculation (optional, else --density)
    a, b           optional: --ab-from-table uses a/b as the lower bound of the a/b prior
    file / path    optional per-row photometry path (--data-column), instead of --data-pattern
All other columns are carried through to batch_summary.csv unchanged.

Period convention
-----------------
The lightcurve of an elongated body has two maxima per rotation, so a single-peaked
Fourier period is half the rotation period. --period-factor (default 2) converts the table
period into the rotation period that is searched; use --period-factor 1 if period_h is
already the (double-peaked) rotation period. The search window is P_rot * (1 +- --window).

Photometry
----------
Local files are looked up with --data-pattern (or --data-column), then in --download-dir. With
--download, targets without a file get their ADES observations from the MPC Observations API
(lcinv/mpc.py), cached as <download-dir>/<num>_ades.csv; --refresh-data fetches them again,
e.g. after new Rubin data were released. The MPC file holds all stations; select with
--common-args "--stations X05" if only Rubin photometry should be used.

Re-running skips targets that already finished (status 'done'); --force reruns them.
Failures are recorded and do not stop the batch.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# ------------------------------------------------------------------------------ input table
def read_table(path):
    """Read a target table from CSV, Parquet, Feather, pickle or Excel (by extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if ext in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def horizons_name(num):
    """'2013_DV23' -> '2013 DV23'; '722574' / 722574.0 -> '722574'."""
    s = str(num).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s.replace("_", " ")


def safe_name(num):
    """Folder name of a target: its Horizons name with spaces and slashes replaced."""
    return str(horizons_name(num)).replace(" ", "_").replace("/", "_")


def find_data(row, patterns, column=None):
    """First existing data file of a target: the row's file column, else each --data-pattern in
    turn (None if none exists).
    """
    cands = []
    if column and column in row and isinstance(row[column], str) and row[column]:
        cands.append(row[column])
    d = horizons_name(row["num"])
    for p in patterns:
        cands.append(p.format(num=row["num"], desig=d, desig_nospace=d.replace(" ", ""),
                              name=safe_name(row["num"])))
    for c in cands:
        c = os.path.expanduser(c)
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


# ------------------------------------------------------------------------------ one target
class StepFailed(RuntimeError):
    """A pipeline step of one target failed; the batch continues with the next target."""
    pass


def _run(cmd, log, timeout_h=None, cwd=HERE):
    with open(log, "a") as fh:
        fh.write(f"\n$ {shlex.join(cmd)}\n")
        fh.flush()
        t0 = time.time()
        try:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
                               timeout=None if not timeout_h else 3600 * timeout_h)
        except subprocess.TimeoutExpired:
            fh.write(f"\n[batch] TIMEOUT after {timeout_h} h\n")
            raise StepFailed(f"timeout ({timeout_h} h): {os.path.basename(cmd[1])}")
        fh.write(f"\n[batch] exit {r.returncode} after {time.time() - t0:.0f} s\n")
    if r.returncode != 0:
        raise StepFailed(f"{os.path.basename(cmd[1])} exited with {r.returncode} (see {log})")


def _tail(path, n=8):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


def download_path(row, cfg):
    """Where the MPC download of a target is stored (--download-dir)."""
    return os.path.abspath(os.path.join(cfg["download_dir"], safe_name(row["num"]) + "_ades.csv"))


def download(row, cfg, log):
    """Fetch the ADES file of this target from the MPC (cached in --download-dir)."""
    from lcinv import mpc
    path = download_path(row, cfg)
    if cfg["dry_run"]:
        print(f"# {row['num']}: would download MPC ADES observations -> {path}")
        return path
    try:
        path, n, nph, new = mpc.fetch_ades(row["num"], path, refresh=cfg["refresh_data"],
                                           ades_version=cfg["ades_version"])
    except mpc.DownloadError as e:
        raise StepFailed(f"MPC download: {e}")
    msg = (f"[{row['num']}] MPC: {n} observations, {nph} with magnitudes -> {path}"
           + ("" if new else " (cached)"))
    print(msg, flush=True)
    with open(log, "a") as fh:
        fh.write(msg + "\n")
    return path


def row_factor(row, cfg):
    """P_rot / period_h: a per-row 'period_factor' column (if present and set) overrides
    the global setting."""
    try:
        f = float(row.get("period_factor"))
        if np.isfinite(f) and f > 0:
            return f
    except (TypeError, ValueError):
        pass
    return cfg["period_factor"]


def build_commands(row, data, tdir, cfg):
    """Shell commands of the pipeline for one target (list of (label, cmd, required))."""
    desig = horizons_name(row["num"])
    P = float(row["period_h"]) * row_factor(row, cfg)
    w = cfg["window"]
    gdir, hdir = os.path.join(tdir, "global"), os.path.join(tdir, "hybrid")
    common = shlex.split(cfg["common_args"] or "")
    dev = ["--device", cfg["device"]]

    inv = [PY, os.path.join(HERE, "run_inversion.py"), data, "--designation", desig,
           "--outdir", gdir, "--pmin", f"{P * (1 - w):.8g}", "--pmax", f"{P * (1 + w):.8g}",
           "--npool", str(cfg["npool"]), *dev, *common]
    if cfg["ab_from_table"] and {"a", "b"} <= set(row.index) and np.isfinite(row["a"] / row["b"]):
        inv += ["--ab", f"{max(1.0, row['a'] / row['b']):.4f}"]
    # diameter (required-cohesion report) and the optional cohesion limit
    dk = row.get("_D_km", np.nan)
    phys = []
    if pd.notna(dk) and float(dk) > 0:
        rho = row.get("density", np.nan)
        rho = float(rho) if pd.notna(rho) and float(rho) > 0 else cfg["density"]
        phys = ["--density", f"{rho:g}"]
        if cfg["cohesion_max"] is not None:
            phys += ["--cohesion-max", f"{float(cfg['cohesion_max']):g}"]
    elif cfg["cohesion_max"] is not None:
        print(f"[{row['num']}] no size estimate: cohesion limit not applied", flush=True)
    dk_args = ["--diameter-km", f"{float(dk):.6g}"] if phys else []
    inv += dk_args + phys
    inv += shlex.split(cfg["inversion_args"] or "")
    steps = [("global", inv, True)]
    if cfg["complexity"]:
        steps.append(("complexity:global", [PY, os.path.join(HERE, "complexity_scan.py"), gdir,
                                            *shlex.split(cfg["complexity_args"] or "")], False))

    dkm0 = row.get("_D_km", np.nan) if cfg["scale_obj"] else np.nan
    if cfg["binary"]:
        bdir = os.path.join(tdir, "binary")
        binc = [PY, os.path.join(HERE, "run_binary.py"), "--global-dir", gdir, "--outdir", bdir,
                "--npool", str(cfg["npool"]), *dev, *shlex.split(cfg["binary_args"] or "")]
        if pd.notna(dkm0) and float(dkm0) > 0:
            binc += ["--diameter-km", f"{float(dkm0):.6g}"]
        elif dk_args:
            binc += dk_args
        binc += phys
        steps.append(("binary", binc, False))
    if cfg["hybrid"]:
        hyb = [PY, os.path.join(HERE, "run_hybrid.py"), data, "--designation", desig,
               "--global-dir", gdir, "--outdir", hdir, *dev, *common, *dk_args, *phys,
               *shlex.split(cfg["hybrid_args"] or "")]
        steps.append(("hybrid", hyb, True))

    dkm = row.get("_D_km", np.nan) if cfg["scale_obj"] else np.nan
    scale = ["--diameter-km", f"{float(dkm):.6g}"] if pd.notna(dkm) and float(dkm) > 0 else []
    if cfg["binary"]:
        steps.append(("convergence:binary", [PY, os.path.join(HERE, "check_convergence.py"),
                                             os.path.join(tdir, "binary")], False))
    for name, d in [("global", gdir)] + ([("hybrid", hdir)] if cfg["hybrid"] else []):
        steps.append((f"convergence:{name}", [PY, os.path.join(HERE, "check_convergence.py"), d], False))
        steps.append((f"obj:{name}", [PY, os.path.join(HERE, "export_obj.py"), d, *scale], False))
        steps.append((f"viewer:{name}", [PY, os.path.join(HERE, "make_viewer.py"), d], False))
    if cfg["compare"]:
        dirs = [gdir] + ([os.path.join(tdir, "binary")] if cfg["binary"] else []) + ([hdir] if cfg["hybrid"] else [])
        steps.append(("compare", [PY, os.path.join(HERE, "compare_models.py"), *dirs, "--out",
                                  os.path.join(tdir, "model_comparison")], False))
    return steps


def collect(row, tdir, cfg):
    """Key results of a finished target as a flat dict."""
    out = {}
    gs = os.path.join(tdir, "global", "summary.json")
    if os.path.exists(gs):
        s = json.load(open(gs))
        p, ml, cv = s["posterior"], s["max_likelihood"], s.get("convergence", {})
        th = ml.get("theta", [np.nan] * 4)
        out.update(
            n_points=s.get("n_points"), n_nights=s.get("n_nights"),
            P_rot_h=p["P_h"][1], P_rot_lo_h=p["P_h"][0], P_rot_hi_h=p["P_h"][2],
            P_search_h=s["period_search"]["P_best_h"],
            lam_maxlike_deg=float(th[0]) % 360, bet_maxlike_deg=float(th[1]),
            bet_med_deg=p["beta_deg"][1], frac_prograde=p["frac_prograde"],
            pole_hpd68_sky_frac=p.get("pole_hpd_sky_fraction", {}).get("68%"),
            jitter_mmag=1000 * p["jitter_mag"][1], ab_extent=p["extent_ratios_a_b"][1],
            bc_extent=p["extent_ratios_b_c"][1], amplitude_mag=s["amplitude"]["A"],
            c_lambert=(p["c_lambert"][1] if p.get("c_lambert") else None),
            lnZ=cv.get("lnZ"), global_verdict=cv.get("verdict"),
            n_clipped=len(s.get("clipped_epochs", [])), global_runtime_s=s.get("runtime_s"))
    hs = os.path.join(tdir, "hybrid", "summary_hybrid.json")
    if os.path.exists(hs):
        h = json.load(open(hs))
        modes = h.get("spin_states") or h.get("modes", [])
        if modes:
            best = max(modes, key=lambda m: m.get("weight_local_laplace") or 0)
            verd = [m.get("convergence", {}).get("verdict") or
                    m.get("convergence", {}).get("diagnostics", {}).get("verdict") for m in modes]
            pole = best.get("pole", {}) or {}
            mp = pole.get("mean_pole", [None, None])
            jm = best.get("jitter_mag")
            out.update(
                hybrid_spin_states=len(modes), hybrid_best_spin_state=best.get("name"),
                hybrid_best_weight=best.get("weight_local_laplace"),
                hybrid_lam_deg=mp[0], hybrid_bet_deg=mp[1], hybrid_pole_r68_deg=pole.get("radius68"),
                hybrid_jitter_mmag=(1000 * (jm[1] if isinstance(jm, (list, tuple)) else jm)) if jm is not None else None,
                hybrid_verdicts=";".join(str(v) for v in verd),
                hybrid_runtime_s=h.get("runtime_s"))
    bs = os.path.join(tdir, "binary", "summary.json")
    if os.path.exists(bs):
        b = json.load(open(bs))
        bp, bc = b["posterior"], b.get("evidence_comparison") or {}
        out.update(binary_lnZ=b["sampler"]["logZ"], binary_verdict=b.get("convergence", {}).get("verdict"),
                   binary_delta_lnZ=bc.get("delta_lnZ_binary_minus_convex"), binary_reading=bc.get("reading"),
                   binary_jitter_mmag=1000 * bp["jitter_mag"][1], binary_q=bp["q"][1], binary_s=bp["s"][1],
                   binary_bet_med_deg=bp["beta_deg"][1])
    from lcinv.hybrid import naming
    cj = os.path.join(tdir, "model_comparison.json")
    if os.path.exists(cj):
        c = json.load(open(cj))
        out.update(best_model=c.get("best"), model_recommendation=" ".join(c.get("recommendation", []))[:2000])
    if os.path.exists(gs):
        s = json.load(open(gs))
        fq = s.get("fit_quality") or {}
        out.update(global_chi2_red=fq.get("chi2_red_nominal"), global_p_eff=fq.get("p_eff"),
                   global_rms_mmag=1000 * fq["rms_mag"] if "rms_mag" in fq else None)
        ch = s.get("cohesion") or {}
        if ch.get("required_cohesion_Pa"):
            out.update(required_cohesion_Pa_med=ch["required_cohesion_Pa"][1],
                       required_cohesion_Pa_lo=ch["required_cohesion_Pa"][0],
                       required_cohesion_Pa_hi=ch["required_cohesion_Pa"][2])
    cs = os.path.join(tdir, "global", "complexity_scan.json")
    if os.path.exists(cs):
        out["complexity_recommendation"] = json.load(open(cs)).get("recommendation", {}).get("text")
    best_mode = naming.tag_of(out.get("hybrid_best_spin_state", "spin state 1"))
    if not os.path.exists(os.path.join(tdir, "hybrid", f"shape_viewer_{best_mode}.html")):
        best_mode = best_mode.replace("spin_state_", "mode")       # results from before the rename
    for k, rel in (("viewer_global", "global/shape_viewer.html"), ("obj_dir_global", "global/obj"),
                   ("viewer_hybrid", f"hybrid/shape_viewer_{best_mode}.html"), ("obj_dir_hybrid", "hybrid/obj"),
                   ("folded_global", "global/fig_folded_lightcurve.png"),
                   ("folded_hybrid", "hybrid/fig_folded_lightcurve_hybrid.png"),
                   ("viewer_binary", "binary/shape_viewer.html"),
                   ("folded_binary", "binary/fig_folded_lightcurve.png")):
        if os.path.exists(os.path.join(tdir, rel)):
            out[k] = os.path.join(tdir, rel)
    return out


def size_for(row, cfg):
    """Diameter used for scaling (OBJ files, cohesion): CNEOS estimate from H (default) or the
    table's diameter_km (--size-source table). H: table column (H, H_V, Hmag) or JPL SBDB."""
    from lcinv import size
    out = {}
    if cfg["size_source"] == "H":
        try:
            e = size.estimate_for(row["num"], row=row, H=cfg.get("H"), albedo=cfg["albedo"],
                                  albedo_column="albedo_assumed" if cfg["albedo_from_table"] else None,
                                  lookup=cfg["sbdb"], cache_dir=cfg["sbdb_dir"])
            out.update(H_used=e["H"], H_source=e["H_source"], albedo_used=e["albedo"],
                       D_used_km=e["diameter_km"], D_min_km=e["diameter_min_km"], D_max_km=e["diameter_max_km"],
                       size_source="CNEOS estimate from H")
            return out
        except Exception as ex:  # noqa: BLE001 - fall back to the table diameter
            out["size_note"] = f"no H-based size ({ex})"
    dk = row.get("diameter_km", np.nan)
    try:
        dk = float(dk)
    except (TypeError, ValueError):
        dk = np.nan
    if np.isfinite(dk) and dk > 0:
        out.update(D_used_km=dk, size_source="table diameter_km")
    return out


def process(row, cfg):
    """Run the whole pipeline for one table row and return its batch_summary.csv record."""
    name = safe_name(row["num"])
    tdir = os.path.abspath(os.path.join(cfg["outroot"], name))
    os.makedirs(tdir, exist_ok=True)
    status_path = os.path.join(tdir, "status.json")
    log = os.path.join(tdir, "pipeline.log")
    if not cfg["force"] and os.path.exists(status_path):
        st = json.load(open(status_path))
        if st.get("status") == "done":
            return dict(num=row["num"], status="done (skipped)", outdir=tdir, **size_for(row, cfg),
                        **collect(row, tdir, cfg))

    rec = dict(num=row["num"], outdir=tdir, log=log, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        if not np.isfinite(float(row["period_h"])) or float(row["period_h"]) <= 0:
            raise StepFailed("no valid period_h")
        patterns = list(cfg["data_pattern"]) + [os.path.join(cfg["download_dir"], "{name}_ades.csv")]
        data = None if (cfg["download"] and cfg["refresh_data"]) else \
            find_data(row, patterns, cfg["data_column"])
        if data is None and cfg["download"]:
            data = download(row, cfg, log)
        if data is None:
            raise StepFailed("no photometry file found (check --data-pattern / --data-column, "
                             "or use --download)")
        rec["data"] = data
        row = row.copy()
        rec.update(size_for(row, cfg))
        row["_D_km"] = rec.get("D_used_km", np.nan)
        steps = build_commands(row, data, tdir, cfg)
        if cfg["dry_run"]:
            print(f"\n# {row['num']}  (P_rot = {row_factor(row, cfg):g} x {row['period_h']:.6g} h)")
            for label, cmd, _ in steps:
                print(shlex.join(cmd))
            return dict(rec, status="dry-run")
        if cfg["force"] and os.path.exists(log):
            os.remove(log)
        warnings = []
        for label, cmd, required in steps:
            done_marker = {"global": os.path.join(tdir, "global", "summary.json"),
                           "binary": os.path.join(tdir, "binary", "summary.json"),
                           "hybrid": os.path.join(tdir, "hybrid", "summary_hybrid.json")}.get(label)
            if done_marker and os.path.exists(done_marker) and not cfg["force"]:
                continue                      # resume: this stage finished in an earlier batch
            print(f"[{row['num']}] {label} ...", flush=True)
            start = os.path.getsize(log) if os.path.exists(log) else 0
            try:
                _run(cmd, log, cfg["timeout_h"] if label in ("global", "hybrid") else None)
            except StepFailed as e:
                if required:
                    raise
                warnings.append(f"{label}: {e}")
            if label.startswith("convergence:"):
                with open(log, errors="replace") as f:
                    f.seek(start)
                    txt = f.read()
                i = txt.find("== convergence")
                rep = txt[i:].split("\n[batch]")[0].strip() if i >= 0 else "(no report)"
                with open(os.path.join(tdir, "convergence.txt"), "w" if label.endswith("global") else "a") as f:
                    f.write(f"[{label.split(':')[1]}]\n{rep}\n\n")
        rec.update(status="done" if not warnings else "done with warnings", warnings="; ".join(warnings))
    except Exception as e:  # noqa: BLE001 - record and continue with the next target
        rec.update(status="failed", error=str(e) if isinstance(e, StepFailed) else traceback.format_exc(),
                   log_tail=_tail(log))
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not cfg["dry_run"]:
        with open(status_path, "w") as f:
            json.dump({k: v for k, v in rec.items() if k != "log_tail"}, f, indent=2, default=str)
    print(f"[{row['num']}] {rec['status']}" + (f": {rec.get('error', '')[:200]}" if rec["status"] == "failed" else ""),
          flush=True)
    rec.update(collect(row, tdir, cfg))
    return rec


# ------------------------------------------------------------------------------ batch
DEFAULTS = dict(outroot="results/batch", data_pattern=("data/{num}_ades.csv", "data/{num}.csv", "data/{num}.lc"),
                data_column=None, period_factor=2.0, window=0.05, hybrid=False, jobs=1, npool=2,
                device="cpu", common_args="", inversion_args="", hybrid_args="", ab_from_table=False,
                scale_obj=True, timeout_h=None, download=False, download_dir="data/mpc",
                size_source="H", albedo=0.15, albedo_from_table=False, sbdb=True, sbdb_dir="data/sbdb",
                binary=False, binary_args="", cohesion_max=None, density=1700.0, complexity=False,
                complexity_args="--lmax 3 4 6 8 --tau0 0.6 1.2", compare=True,
                refresh_data=False, ades_version="2022", force=False, dry_run=False, query=None, limit=None, only=None)


def run_batch(df, **kw):
    """Run the pipeline for every row of df; returns the summary DataFrame (also written to
    <outroot>/batch_summary.csv). Keyword arguments: see DEFAULTS / the command-line help."""
    cfg = {**DEFAULTS, **kw}
    if isinstance(cfg["data_pattern"], str):
        cfg["data_pattern"] = (cfg["data_pattern"],)
    for col in ("num", "period_h"):
        if col not in df.columns:
            raise ValueError(f"input table needs a '{col}' column")
    sel = df.copy()
    if cfg["query"]:
        sel = sel.query(cfg["query"])
    if cfg["only"]:
        want = {safe_name(x) for x in cfg["only"]}
        sel = sel[sel["num"].map(safe_name).isin(want)]
    if cfg["limit"]:
        sel = sel.head(int(cfg["limit"]))
    os.makedirs(cfg["outroot"], exist_ok=True)
    ncpu = os.cpu_count() or 1
    if cfg["jobs"] * cfg["npool"] > ncpu and cfg["device"] == "cpu":
        print(f"warning: --jobs {cfg['jobs']} x --npool {cfg['npool']} = {cfg['jobs'] * cfg['npool']} "
              f"workers on {ncpu} CPUs; runs will compete for cores", file=sys.stderr)
    print(f"{len(sel)} target(s) -> {os.path.abspath(cfg['outroot'])} "
          f"(P_rot = {cfg['period_factor']:g} x period_h, window +-{100 * cfg['window']:g}%)")

    rows = [r for _, r in sel.iterrows()]
    results = []
    if cfg["jobs"] <= 1 or cfg["dry_run"]:
        for r in rows:
            results.append(process(r, cfg))
    else:
        with ThreadPoolExecutor(max_workers=cfg["jobs"]) as ex:
            futs = {ex.submit(process, r, cfg): r["num"] for r in rows}
            for f in as_completed(futs):
                results.append(f.result())
    if cfg["dry_run"]:
        return pd.DataFrame(results)

    res = pd.DataFrame(results)
    res["_key"] = res["num"].map(safe_name)
    base = sel.assign(_key=sel["num"].map(safe_name)).drop(columns=["num"])
    out = res.merge(base, on="_key", how="left").drop(columns="_key")
    lead = ["num", "status", "H_used", "D_used_km", "size_source", "P_rot_h", "lam_maxlike_deg", "bet_maxlike_deg", "frac_prograde", "jitter_mmag",
            "lnZ", "global_verdict", "binary_delta_lnZ", "binary_reading", "hybrid_verdicts"]
    out = out[[c for c in lead if c in out] + [c for c in out if c not in lead]]
    path = os.path.join(cfg["outroot"], "batch_summary.csv")
    # merge with an earlier summary so a partial rerun (--only/--limit) does not lose rows
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["num"].map(safe_name).isin(out["num"].map(safe_name))]
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(path, index=False)
    n = out["status"].value_counts().to_dict()
    print(f"\nsummary -> {path}: " + ", ".join(f"{v} {k}" for k, v in n.items()))
    return out


def main(argv=None):
    """Command line of batch_shapes.py (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Arguments after the options are not allowed; pass script options via "
                                       "--common-args / --inversion-args / --hybrid-args (quoted).")
    p.add_argument("table", help="CSV/Parquet/Feather/pickle/Excel with columns num, period_h, ...")
    p.add_argument("--outroot", default=DEFAULTS["outroot"],
                   help="folder for the per-target results and batch_summary.csv")
    p.add_argument("--data-pattern", action="append", default=None,
                   help="photometry path template with {num}, {desig}, {desig_nospace} or {name}; "
                        "repeatable, first existing file wins (default data/{num}_ades.csv, data/{num}.csv, data/{num}.lc)")
    p.add_argument("--data-column", default=None, help="table column holding the photometry path per row")
    p.add_argument("--period-factor", type=float, default=DEFAULTS["period_factor"],
                   help="rotation period = factor x period_h (2: table periods are single-peaked)")
    p.add_argument("--window", type=float, default=DEFAULTS["window"],
                   help="fractional half-width of the period search around the rotation period")
    g = p.add_argument_group("MPC download")
    g.add_argument("--download", action="store_true",
                   help="download ADES observations from the MPC for targets without a local file")
    g.add_argument("--download-dir", default=DEFAULTS["download_dir"],
                   help="where downloaded files are stored (<num>_ades.csv); also searched for data")
    g.add_argument("--refresh-data", action="store_true",
                   help="download again even if a downloaded file exists (new observations)")
    g.add_argument("--ades-version", default="2022", choices=["2017", "2022"],
                   help="ADES version requested from the MPC")
    p.add_argument("--hybrid", action="store_true", help="also run the hybrid model")
    p.add_argument("--binary", action="store_true",
                   help="also fit the contact-binary model (run_binary.py) and compare its evidence")
    p.add_argument("--binary-args", default="", help="extra options for run_binary.py, e.g. \"--nlive 400\"")
    g = p.add_argument_group("size (scales OBJ files, used for cohesion)")
    g.add_argument("--size-source", choices=["H", "table"], default="H",
                   help="H: CNEOS size estimate D = 10^(3.1236 - 0.5 log10 p - 0.2 H) km (default); "
                        "table: the diameter_km column")
    g.add_argument("--albedo", type=float, default=0.15, help="geometric albedo for the H-based size")
    g.add_argument("--albedo-from-table", action="store_true",
                   help="use the table's albedo_assumed column where present")
    g.add_argument("--no-sbdb", dest="sbdb", action="store_false",
                   help="do not look up H in the JPL Small-Body Database when the table has no H column")
    g.add_argument("--sbdb-dir", default="data/sbdb", help="cache of SBDB responses")
    g = p.add_argument_group("physics, complexity and model comparison")
    g.add_argument("--cohesion-max", type=float, default=None, metavar="PA",
                   help="penalise shapes/spins needing more cohesion than this (Sanchez & Scheeres 2014), "
                        "using the size estimate (and the table's density column if present)")
    g.add_argument("--density", type=float, default=DEFAULTS["density"],
                   help="bulk density [kg/m^3] when the table has no density column")
    g.add_argument("--complexity", action="store_true",
                   help="run complexity_scan.py on every global run (evidence vs lmax / tau0)")
    g.add_argument("--complexity-args", default=DEFAULTS["complexity_args"],
                   help="options passed to complexity_scan.py (quoted)")
    g.add_argument("--no-compare", dest="compare", action="store_false",
                   help="skip compare_models.py (model_comparison.md per target)")
    p.add_argument("--jobs", type=int, default=1, help="targets processed in parallel")
    p.add_argument("--npool", type=int, default=2, help="likelihood workers per global run")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                   help="JAX device of the model scripts")
    p.add_argument("--common-args", default="",
                   help="options for both run_inversion.py and run_hybrid.py, e.g. "
                        "\"--stations X05 --observer local --exptime 30 --fit-cl\"")
    p.add_argument("--inversion-args", default="", help="extra options for run_inversion.py, e.g. \"--nlive 300\"")
    p.add_argument("--hybrid-args", default="", help="extra options for run_hybrid.py, e.g. \"--chains 4\"")
    p.add_argument("--ab-from-table", action="store_true", help="use a/b from the table as the a/b prior bound")
    p.add_argument("--no-scale-obj", dest="scale_obj", action="store_false",
                   help="do not scale OBJ files to the size estimate")
    p.add_argument("--timeout-h", type=float, default=None, help="per-step time limit for global/hybrid runs")
    p.add_argument("--query", default=None, help="pandas query to select rows, e.g. \"chi2_red < 2\"")
    p.add_argument("--only", nargs="+", default=None, help="process only these targets (num values)")
    p.add_argument("--limit", type=int, default=None, help="process only the first N selected rows")
    p.add_argument("--force", action="store_true", help="rerun targets that already finished")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = p.parse_args(argv)
    cfg = vars(a)
    df = read_table(cfg.pop("table"))
    if cfg["data_pattern"] is None:
        cfg["data_pattern"] = DEFAULTS["data_pattern"]
    res = run_batch(df, **cfg)
    return 0 if a.dry_run or not (res["status"] == "failed").any() else 1


if __name__ == "__main__":
    sys.exit(main())
