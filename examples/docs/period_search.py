"""Period search of K03Q21N with lcinv.period (no JAX involved)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lcinv import period as per
from lcinv.setup_data import prepare

D = prepare("data/K03Q21N.lc", "K03Q21N", location="X05", cache_dir="data")
sig = np.hypot(D["err"], 0.015)                  # quoted errors + 15 mmag floor, as run_inversion
args = (D["t"], D["mag"], sig, D["band_idx"], D["alpha"])

# 1. coarse periodogram from 1 to 30 h, uniform in frequency
P, chi2, nu = per.periodogram(*args, P_min_h=1.0, P_max_h=30.0, K=6, oversample=5)
print(f"{len(P)} trial periods, nu = {nu}")

# 2. the separated minima: the competing aliases
print("  P [h]    chi2/nu")
for p, c in per.local_minima(P, chi2, n=5):
    print(f"  {p:8.5f}  {c / nu:6.3f}")

# 3. refine the best minimum above 2 h
best = per.local_minima(P[P >= 2], chi2[P >= 2], n=1)[0][0]
ref = per.refine_period_auto(*args, best, K=6)
print(f"P = {ref['P']:.5f} h  +- {ref['sigma']:.5f} (Delta chi2 = 1)  "
      f"threshold [{ref['P_lo']:.5f}, {ref['P_hi']:.5f}]  red. chi2 {ref['red_chi2']:.3f}")

# 4. amplitude and the lower bound on a/b
amp, amp_err = per.amplitude(*args, ref["P"], K=6)
ab = per.ab_prior_from_amplitude(amp, amp_err, float(np.mean(D["alpha"])))
print(f"amplitude {amp:.3f} +- {amp_err:.3f} mag at mean phase angle {np.mean(D['alpha']):.1f} deg "
      f"-> a/b >= {ab['ab_min']:.3f}")

# 5. figure: periodogram with the half-period alias marked
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(P, chi2 / nu, lw=0.7, color="#1b2433")
for p, lab in [(ref["P"], "rotation period"), (ref["P"] / 2, "half-period alias")]:
    ax.axvline(p, color="#3b4f8f" if lab == "rotation period" else "#c4322a", lw=1, ls="--")
    left = lab != "rotation period"              # alias label to the left of its line
    ax.text(p / 1.04 if left else p * 1.04, 43, f"{lab} {p:.4f} h", fontsize=8, va="bottom",
            ha="right" if left else "left", bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.set_xscale("log"); ax.set_xlabel("period [h]"); ax.set_ylabel(r"$\chi^2/\nu$")
ax.set_ylim(0, 48); fig.tight_layout(); fig.savefig("examples/docs/period_search.png", dpi=150)
