"""Inject a triaxial ellipsoid with a known spin state (phi0 != 0) into the real K03Q21N
geometry and scan the gauge-fixed likelihood over the pole: does it recover the truth?"""
import time
import numpy as np
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion

D = prepare("data/K03Q21N.lc", "K03Q21N")
true = dict(lam=60.0, bet=50.0, P=3.24826, phi0=0.7, a=1.6, b=1.1, c=0.9)
args = (D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"])
M = ConvexInversion(*args)
n = M.normals_np
h = np.sqrt((true["a"] * n[:, 0]) ** 2 + (true["b"] * n[:, 1]) ** 2 + (true["c"] * n[:, 2]) ** 2)
A = M.w0 * (true["a"] * true["b"] * true["c"]) ** 2 / h ** 4
S = np.asarray(M.scattering_matrix(true["lam"], true["bet"], true["P"], true["phi0"]))
rng = np.random.default_rng(0)
off = np.array([20.4, 19.7, 19.9])[D["band_idx"]]
syn = -2.5 * np.log10(S @ A) + off + 0.03 * D["alpha"] + rng.normal(0, np.hypot(D["err"], 0.012))
M2 = ConvexInversion(D["t"], syn, *args[2:], bc_prior=("lower", 1.0, 0.05))
lnj = np.log(0.012)
print("at truth: logL %.1f chi2 %.1f (N=%d)" % (M2.loglike([60, 50, true["P"], lnj]),
      float(M2.loglike_jit(np.array([60, 50, true["P"], lnj]))["chi2"]), len(syn)))
t0 = time.time()
grid = {}
for lam in range(0, 360, 20):
    for bet in (-70, -45, -20, 0, 20, 45, 70):
        grid[(lam, bet)] = M2.loglike([lam, bet, true["P"], lnj])
print("scan %.0fs" % (time.time() - t0))
best = max(grid.values())
for k, v in sorted(grid.items(), key=lambda kv: -kv[1])[:10]:
    print(k, "%.1f" % (v - best))
