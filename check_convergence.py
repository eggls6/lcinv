#!/usr/bin/env python
"""Convergence report for finished runs, and agreement between repeat runs.

    python check_convergence.py results/K03Q21N                    # one global run
    python check_convergence.py results/X_seed1 results/X_seed2    # + reproducibility
    python check_convergence.py results/X_hybrid                   # hybrid (NUTS) run

Global runs are read from nested_results.pkl + summary.json; hybrid runs from
posterior_spin_state_*.npz. Repeat runs agree when ln Z differs by less than ~2 combined sigma and the
period, jitter and prograde fraction overlap within their errors.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcinv import diagnostics  # noqa: E402


def global_run(out):
    """Convergence report of a run_inversion.py or run_binary.py folder; returns its numbers for
    compare().
    """
    from lcinv import sampler
    d = sampler.load(os.path.join(out, "nested_results.pkl"))
    s = json.load(open(os.path.join(out, "summary.json")))
    sm = s["sampler"]
    conv = diagnostics.nested(d["logwt"], d["logz"], d["logzerr"], sm["nlive"],
                              beta=np.asarray(d["samples"])[:, 1], resumed=sm.get("finalised_from_checkpoint", False),
                              maxcall_hit=bool(d["args"].get("maxcall") and sm["ncall"] >= d["args"]["maxcall"]))
    print(diagnostics.format_report(f"{out} (nlive {sm['nlive']}, {sm.get('sample', 'unif')}, "
                                    f"seed {sm.get('seed', 42)})", conv))
    return dict(out=out, conv=conv, post=s["posterior"])


def hybrid_run(out):
    """Convergence report of every spin state of a run_hybrid.py folder."""
    from lcinv.hybrid import naming
    for _, f in naming.posterior_files(out):
        z = np.load(f)
        der = {"P": z["P"], "jitter": z["jitter"]}
        print(diagnostics.format_report(f"{out}/{os.path.basename(f)}",
                                        diagnostics.nuts(z["x"], z["diverging"], der)))


def compare(runs):
    """Reproducibility of repeated global runs of one model: ln Z, period and prograde fraction
    within their errors.
    """
    print("== reproducibility ==")
    ok = True
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            dz = a["conv"]["lnZ"] - b["conv"]["lnZ"]
            sz = np.hypot(a["conv"]["lnZ_err"], b["conv"]["lnZ_err"])
            df = a["conv"]["frac_prograde"] - b["conv"]["frac_prograde"]
            sf = np.hypot(a["conv"]["frac_prograde_err"], b["conv"]["frac_prograde_err"])
            pa, pb = a["post"]["P_h"], b["post"]["P_h"]
            p_ok = pa[0] <= pb[2] and pb[0] <= pa[2]
            good = abs(dz) < 2 * sz and abs(df) < 2 * sf + 0.02 and p_ok
            ok &= good
            print(f"   {a['out']} vs {b['out']}: d lnZ {dz:+.2f} ({abs(dz) / sz:.1f} sigma), "
                  f"d prograde {df:+.3f} ({abs(df) / sf:.1f} sigma), period intervals "
                  f"{'overlap' if p_ok else 'DISJOINT'} -> {'agree' if good else 'DISAGREE'}")
    print("   runs agree" if ok else "   runs disagree: increase --nlive and/or lower --dlogz")


if __name__ == "__main__":
    runs = []
    for out in sys.argv[1:]:
        if os.path.exists(os.path.join(out, "nested_results.pkl")):
            runs.append(global_run(out))
        elif glob.glob(os.path.join(out, "posterior_spin_state_*.npz")) or glob.glob(os.path.join(out, "posterior_mode*.npz")):
            hybrid_run(out)
        else:
            print(f"{out}: no nested_results.pkl or posterior_spin_state_*.npz found")
    if len(runs) > 1:
        compare(runs)
