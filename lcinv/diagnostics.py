"""Convergence diagnostics and verdicts for the global (nested sampling) and local (NUTS) layers.

Global layer (dynesty)
----------------------
* dlogz_live: ln Z - ln(Z - Z_live), the evidence contributed by the final live points. It is
  computed from the result itself, so it is meaningful however the run stopped (stopping rule,
  --maxcall, or finalising a stalled run with --resume).
* ESS: Kish effective sample size of the importance weights, ESS = (sum w)^2 / sum w^2.
* ln Z error, and the Monte Carlo error of the prograde fraction, sqrt(f (1 - f) / ESS).

Local layer (NUTS)
------------------
Rank-normalised split R-hat and bulk ESS on the sampled pole-chart coordinates (d1, d2), the
period and the jitter, plus the divergence count. The chart coordinates are used rather than
the ecliptic longitude, whose R-hat is inflated near the ecliptic poles even for well-mixed chains.

Each check is graded 'ok', 'warn' or 'fail'; the verdict is the worst grade.
"""
import numpy as np
from scipy.special import logsumexp

# thresholds: (ok if <= / >= first, warn if within second, else fail)
GLOBAL = dict(dlogz_live=(0.5, 1.0), ess=(400, 150), logz_err=(0.5, 1.0), frac_err=(0.05, 0.10))
LOCAL = dict(rhat=(1.01, 1.05), ess=(400, 100), divergences=(0, 10))

_RANK = {"ok": 0, "warn": 1, "fail": 2}
VERDICT = {"ok": "CONVERGED", "warn": "CONVERGED WITH WARNINGS", "fail": "NOT CONVERGED"}


def _grade_upper(x, lim):
    return "ok" if x <= lim[0] else ("warn" if x <= lim[1] else "fail")


def _grade_lower(x, lim):
    return "ok" if x >= lim[0] else ("warn" if x >= lim[1] else "fail")


def _worst(grades):
    return max(grades, key=_RANK.get) if grades else "ok"


def nested(logwt, logz, logzerr, nlive, beta=None, resumed=False, maxcall_hit=False):
    """Convergence diagnostics of a dynesty run.

    Parameters
    ----------
    logwt, logz, logzerr : array_like
        From the dynesty results (final live points included).
    nlive : int
        Number of live points.
    beta : array_like, optional
        Pole latitude of each sample, for the prograde fraction and its error.
    resumed, maxcall_hit : bool, optional
        How the run ended; used in the advice.

    Returns
    -------
    dict
        ``lnZ``, ``lnZ_err``, ``ess``, ``evidence_fraction_in_live``, ``dlogz_live``,
        ``frac_prograde``, ``frac_prograde_err``, ``resumed``, ``maxcall_hit``, ``checks`` (grade
        per check), ``verdict`` and ``advice``. Thresholds are in `GLOBAL`.
    """
    logwt = np.asarray(logwt)
    lnZ = float(logz[-1])
    w = np.exp(logwt - lnZ)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    frac_live = float(np.exp(logsumexp(logwt[-nlive:]) - lnZ))
    dlogz_live = float(-np.log1p(-min(frac_live, 1 - 1e-12)))
    out = dict(lnZ=lnZ, lnZ_err=float(logzerr[-1]), ess=ess, evidence_fraction_in_live=frac_live,
               dlogz_live=dlogz_live, resumed=bool(resumed), maxcall_hit=bool(maxcall_hit))
    checks = {"dlogz_live": _grade_upper(dlogz_live, GLOBAL["dlogz_live"]),
              "ess": _grade_lower(ess, GLOBAL["ess"]),
              "lnZ_err": _grade_upper(out["lnZ_err"], GLOBAL["logz_err"])}
    if beta is not None:
        # weighted prograde fraction and its Monte Carlo error
        f = float(np.sum(w * (np.asarray(beta) > 0)) / w.sum())
        out["frac_prograde"] = f
        out["frac_prograde_err"] = float(np.sqrt(max(f * (1 - f), 1e-12) / ess))
        checks["frac_prograde_err"] = _grade_upper(out["frac_prograde_err"], GLOBAL["frac_err"])
    out["checks"] = checks
    out["verdict"] = VERDICT[_worst(list(checks.values()))]
    out["advice"] = _nested_advice(out)
    return out


def _nested_advice(d):
    a = []
    c = d["checks"]
    if c["dlogz_live"] != "ok":
        a.append(f"{100 * d['evidence_fraction_in_live']:.0f}% of the evidence is still in the final live "
                 "points: the run stopped early" + (" (--maxcall reached)" if d["maxcall_hit"] else
                                                     " (finalised from a checkpoint)" if d["resumed"] else "")
                 + "; rerun to completion or continue with --resume --extra-calls N")
    if c["ess"] != "ok" or c.get("frac_prograde_err", "ok") != "ok":
        a.append("posterior ESS is low: increase --nlive (e.g. x2) or lower --dlogz to 0.1")
    if c["lnZ_err"] != "ok":
        a.append("ln Z is noisy: increase --nlive")
    if not a:
        a.append("for publication: --dlogz 0.1 and one repeat with another --seed "
                 "(compare with check_convergence.py)")
    return a


def nuts(x, diverging, der):
    """Convergence diagnostics of the NUTS chains of one spin state.

    Parameters
    ----------
    x : ndarray
        Sampled coordinates, (chain, draw, ndim); the first two are the pole-chart coordinates.
    diverging : array_like of bool
        Divergent transitions.
    der : dict
        Derived quantities flattened over (chain, draw): ``P``, ``jitter`` and optionally ``c_L``.

    Returns
    -------
    dict
        ``params`` (R-hat and ESS per parameter), ``rhat_max``, ``ess_min``, ``divergences``,
        ``chains``, ``draws``, ``checks``, ``verdict``, ``advice``. Thresholds are in `LOCAL`.
    """
    from lcinv.hybrid.inference import _rhat, _ess
    nc, nd = x.shape[:2]
    params = {"d1": x[:, :, 0], "d2": x[:, :, 1], "P": der["P"].reshape(nc, nd),
              "jitter": der["jitter"].reshape(nc, nd)}
    if "c_L" in der:
        params["c_L"] = np.log(np.asarray(der["c_L"]).reshape(nc, nd))
    stats = {k: dict(rhat=_rhat(v), ess=_ess(v)) for k, v in params.items()}
    ndiv = int(np.sum(diverging))
    rmax = max(s["rhat"] for s in stats.values())
    emin = min(s["ess"] for s in stats.values())
    checks = {"rhat": _grade_upper(rmax, LOCAL["rhat"]), "ess": _grade_lower(emin, LOCAL["ess"]),
              "divergences": _grade_upper(ndiv, LOCAL["divergences"])}
    if nc < 2:
        checks["chains"] = "warn"
    out = dict(params=stats, rhat_max=rmax, ess_min=emin, divergences=ndiv, chains=nc, draws=nd,
               checks=checks, verdict=VERDICT[_worst(list(checks.values()))])
    adv = []
    if checks["rhat"] != "ok" or checks["ess"] != "ok":
        adv.append("chains have not mixed: increase --warmup/--samples (e.g. 1000/1500) and --chains (4)")
    if checks["divergences"] != "ok":
        adv.append("divergences: raise --target-accept (0.95) or --tree-depth")
    if nc < 2:
        adv.append("R-hat needs at least 2 chains")
    out["advice"] = adv
    return out


def format_report(title, d):
    """Text report of `nested` or `nuts` diagnostics, as printed by the scripts and
    ``check_convergence.py``.
    """
    lines = [f"== convergence: {title}: {d['verdict']} =="]
    if "dlogz_live" in d:
        lines.append(f"   ln Z = {d['lnZ']:.2f} +- {d['lnZ_err']:.2f} [{d['checks']['lnZ_err']}]; "
                     f"evidence left in live points {100 * d['evidence_fraction_in_live']:.0f}% "
                     f"(dlogz {d['dlogz_live']:.2f}) [{d['checks']['dlogz_live']}]; "
                     f"ESS {d['ess']:.0f} [{d['checks']['ess']}]")
        if "frac_prograde" in d:
            lines.append(f"   prograde fraction {d['frac_prograde']:.3f} +- {d['frac_prograde_err']:.3f} "
                         f"[{d['checks']['frac_prograde_err']}]")
    else:
        ps = ", ".join(f"{k} {v['rhat']:.3f}/{v['ess']:.0f}" for k, v in d["params"].items())
        lines.append(f"   R-hat/ESS: {ps}; divergences {d['divergences']} "
                     f"[rhat {d['checks']['rhat']}, ess {d['checks']['ess']}, div {d['checks']['divergences']}]")
    lines += [f"   -> {s}" for s in d["advice"]]
    return "\n".join(lines)
