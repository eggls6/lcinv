"""Collect period-search results (*.par, with *.obsfull) into a targets table laid out like
targets.csv, so batch_shapes.py can use it.

    python par_to_targets.py par_files/ -o targets_from_par.csv [--like targets.csv]

Every column that can be read or recomputed from the .par / .obsfull files is filled in.
Columns produced by later steps of the pipeline (period-rival analysis, spectral
classification, H_V / albedo / diameter, strength) are left empty. If you supply albedos
or diameters, a and b follow from the elongation (see axes_from_diameter).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
COLUMNS = [
    "num", "period_h", "period_left", "period_right", "period_day", "freq", "other_periods", "sigma",
    "chi2_red", "chi2_floor", "min_sig", "chi2", "thres_sig", "num_solut", "n_out", "n_obs", "n_cos",
    "n_val", "period_candidates", "multi_factor", "cov_len", "cov_ceiling", "cov_grp", "cov_gap",
    "coef_cos", "mumag_i", "mumag_u", "mumag_g", "mumag_r", "mumag_z", "mumag_y", "mumag_err_i",
    "mumag_err_u", "mumag_err_g", "mumag_err_r", "mumag_err_z", "mumag_err_y", "mumag", "mumag_err",
    "bands", "phase_model", "coef_phase", "coef_phase_err", "x0", "phase", "jd_ref", "f_test_val",
    "order_table", "n_peak", "1cos_per", "2cos_per", "f_test_matr", "vuong_matr", "significance_matr",
    "cov_H", "p2p", "period", "elongation_m02", "cov_floor", "cov_below_floor", "n_rivals", "n_harm",
    "n_indep", "p_correct", "p_correct_min", "p_harm", "p_harm_min", "rel_unc", "period_class",
    "coverage_class", "spec_class", "spec_class_prob", "spec_n_bands", "spec_probs", "spec_bands_used",
    "H_V", "albedo_assumed", "diameter_km", "a", "b", "required_strength_Pa", "rivals_compact",
    "cohesive_strenght_accurate", "diameter_min", "diameter_max"]
NOT_FROM_PAR = COLUMNS[COLUMNS.index("cov_floor"):]
NOT_FROM_PAR.remove("rel_unc")


def _f(s):
    return float(s.replace("D", "E").replace("d", "e"))


def _nums(s):
    return [_f(x) for x in re.findall(FLOAT, s)]


def _num(x):
    """'2018_PD28' stays a string; '860' -> '860' (as in targets.csv)."""
    return str(x)


def _block(lines, start, stop_pred):
    out = []
    for ln in lines[start:]:
        if stop_pred(ln):
            break
        out.append(ln)
    return out


def read_par(path):
    """Parse a period-search .par file: summary values, Fourier coefficients, period candidates."""
    L = open(path).read().splitlines()
    txt = "\n".join(L)
    r = {}

    def grab(pattern, cast=_f, default=np.nan):
        m = re.search(pattern, txt)
        return cast(m.group(1)) if m else default

    r["num"] = _num(grab(r"Asteroid's number\s*=\s*(\S+)", str))
    m = re.search(rf"Period\s*=\s*({FLOAT})\s*hours\s*\[\s*({FLOAT})\s*,\s*({FLOAT})\s*\]", txt)
    r["period_h"], r["period_left"], r["period_right"] = (_f(m.group(i)) for i in (1, 2, 3))
    other = re.search(r"Other possible periods[^=]*=(.*)", txt).group(1)
    cand = [(float(p), int(n), float(s)) for p, n, s in
            re.findall(rf"({FLOAT})\s*\(\s*(\d+)\s*,\s*({FLOAT})\s*\)", other)]
    m = re.search(rf"sigma of observations\s*=\s*({FLOAT})\s*reduced chi\^2\s*=\s*({FLOAT})\s*\(floor\s*({FLOAT})\)", txt)
    r["sigma"], r["chi2_red"], r["chi2_floor"] = (_f(m.group(i)) for i in (1, 2, 3))
    r["thres_sig"] = grab(rf"significance level of sigma[^=]*=\s*({FLOAT})")
    r["num_solut"] = int(grab(r"number of solutions[^=]*=\s*(\d+)", int))
    r["n_out"] = int(grab(r"number of excluded obs\s*=\s*(\d+)", int))
    r["n_obs"] = int(grab(r"number of observations\s*=\s*(\d+)", int))
    r["n_cos"] = int(grab(r"Number of cosinus fit\s*=\s*(\d+)", int))
    r["multi_factor"] = int(grab(r"Multiplication factor of period\s*=\s*(\d+)", int))
    m = re.search(rf"maximal uncovered interval\s*=\s*({FLOAT})\s*\(ceiling\s*({FLOAT})\)\s*({FLOAT})\s*({FLOAT})", txt)
    r["cov_len"], r["cov_ceiling"], r["cov_grp"], r["cov_gap"] = (_f(m.group(i)) for i in (1, 2, 3, 4))

    i = next(k for k, ln in enumerate(L) if "Amplitude cos" in ln) + 1
    coef = np.array([_nums(ln) for ln in L[i:i + r["n_cos"]]])

    m = re.search(r"Magnitude offset\s*=(.*?)uncertainties\s*=(.*?)bands:(.*)", txt)
    off, unc, bands = _nums(m.group(1)), _nums(m.group(2)), m.group(3).split()
    mumag = dict(zip(bands, off))
    mumag_err = dict(zip(bands, unc))

    i = next(k for k, ln in enumerate(L) if "Parameters of the phase function" in ln)
    phase_model = L[i].split(":")[-1].strip()
    pl = L[i + 1].split("uncertainties")
    coef_phase, coef_phase_err = _nums(pl[0]), _nums(pl[1])
    m = re.search(rf"Value of x0\s*=\s*({FLOAT})\s*weighted average angle phase\s*=\s*({FLOAT})", txt)
    x0, phase = _f(m.group(1)), _f(m.group(2))
    jd_ref = grab(rf"First observation JD\s*=\s*({FLOAT})")

    i = next(k for k, ln in enumerate(L) if "Results of F-test" in ln) + 1
    ftest = sum((_nums(ln) for ln in _block(L, i, lambda s: s.startswith("num of cos"))), [])
    i = next(k for k, ln in enumerate(L) if ln.startswith("num of cos")) + 1
    order = []
    for ln in _block(L, i, lambda s: "matrix" in s):
        v = _nums(ln)
        if len(v) >= 6:
            order.append({"n_cos": int(v[0]), "period_h": v[1], "sigma": v[2], "n_peak": int(v[3]),
                          "bic": v[4], "n_excl": int(v[5])})

    def matrix(title):
        i = next(k for k, ln in enumerate(L) if title in ln) + 1
        return [_nums(ln) for ln in _block(L, i, lambda s: "matrix" in s or not s.strip())]

    ftm, vm, sm, cov = (matrix(t) for t in ("F-test matrix", "Vuong matrix",
                                             "Significance of the parameters matrix",
                                             "Covariance matrix for H mags"))
    return r, coef, mumag, mumag_err, bands, phase_model, coef_phase, coef_phase_err, x0, phase, \
        jd_ref, ftest, order, ftm, vm, sm, cov, cand


def fourier(coef, t):
    """Fourier series sum a cos(2 pi f t) + b sin(2 pi f t) over (a, b, f) rows."""
    return sum(a * np.cos(2 * np.pi * f * t) + b * np.sin(2 * np.pi * f * t) for a, b, f in coef)


def row(par_path, obs_path=None):
    """One targets-table row from a .par file and, if given, its .obsfull file."""
    (r, coef, mumag, mumag_err, bands, pm, cp, cpe, x0, phase, jd_ref, ftest, order, ftm, vm, sm,
     cov, cand) = read_par(par_path)
    P = r["period_h"]
    r.update(period_day=P / 24.0, freq=float(coef[0, 2]) if len(coef) else 24.0 / P,
             other_periods="[]", min_sig=r["sigma"], chi2=r["chi2_red"],
             n_val=r["n_obs"] - r["n_out"], period_candidates=str(cand), coef_cos=str(coef))
    for b in "iugrzy":
        r[f"mumag_{b}"] = mumag.get(b, np.nan)
        r[f"mumag_err_{b}"] = mumag_err.get(b, np.nan)
    r.update(mumag=str(mumag), mumag_err=str(mumag_err), bands=str(bands), phase_model=pm,
             coef_phase=str(cp), coef_phase_err=str(cpe), x0=x0, phase=phase, jd_ref=jd_ref,
             f_test_val=str(ftest), order_table=str(order),
             f_test_matr=str(np.array(ftm)), vuong_matr=str(np.array(vm)),
             significance_matr=str(sm), cov_H=str(np.array(cov)), period=P,
             rel_unc=(r["period_right"] - r["period_left"]) / P)
    sel = [o for o in order if o["n_cos"] == r["n_cos"]]
    r["n_peak"] = sel[0]["n_peak"] if sel else np.nan
    for j, c in ((0, "1cos_per"), (1, "2cos_per")):
        if j < len(order):
            o = order[j]                                    # single-peaked fits -> double period
            r[c] = o["period_h"] * 2 if o["n_peak"] == 1 else o["period_h"]
    # peak-to-peak amplitude of the Fourier model at the observed epochs (days since jd_ref)
    if obs_path and os.path.exists(obs_path):
        t = np.genfromtxt(obs_path, usecols=(0,))
        m = fourier(coef, t)
        r["p2p"] = float(m.max() - m.min())
    else:
        tt = np.linspace(0, 2 * P / 24.0, 20001)
        m = fourier(coef, tt)
        r["p2p"] = float(m.max() - m.min())
    # P_rot / period_h for batch_shapes.py: a fit with a single maximum per cycle is the
    # half-period alias, otherwise the period is taken as the rotation period
    r["period_factor"] = 2.0 if r["n_peak"] == 1 else 1.0
    r["elongation_m02"] = 10 ** (0.4 * r["p2p"] / (1 + 0.02 * phase))   # amplitude at 0 deg phase
    return r


def axes_from_diameter(D, elongation):
    """Full axes a >= b with a/b = elongation and b(a+b)/2 = D^2 (as in targets.csv)."""
    b = D * np.sqrt(2.0 / (1.0 + elongation))
    return elongation * b, b


def main(argv=None):
    """Command line of par_to_targets.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder",
                   help="folder with *.par files (and *.obsfull)")
    p.add_argument("-o", "--output", default="targets_from_par.csv",
                   help="output table (CSV)")
    p.add_argument("--like", help="existing targets.csv: compare objects present in both")
    a = p.parse_args(argv)
    rows = []
    for par in sorted(glob.glob(os.path.join(a.folder, "*.par"))):
        stem = par[:-4]
        try:
            rows.append(row(par, stem + ".obsfull"))
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(par)}: could not parse ({e})")
    df = pd.DataFrame(rows).reindex(columns=COLUMNS + ["period_factor"])
    df.to_csv(a.output)
    print(f"{len(df)} objects -> {a.output}")
    if a.like:
        ref = pd.read_csv(a.like, index_col=0)
        ref["num"] = ref["num"].astype(str)
        both = df.merge(ref, on="num", suffixes=("", "_ref"))
        for _, r in both.iterrows():
            diff = []
            for c in COLUMNS[1:]:
                if c in NOT_FROM_PAR:
                    continue
                x, y = r[c], r[c + "_ref"]
                try:
                    ok = np.isclose(float(x), float(y), rtol=2e-3, equal_nan=True)
                except (TypeError, ValueError):
                    ok = str(x) == str(y)
                if not ok:
                    diff.append(c)
            print(f"check against {a.like}, object {r['num']}: differing columns {diff or 'none'}")
    return df


if __name__ == "__main__":
    main()
