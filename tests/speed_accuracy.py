"""Speed vs accuracy of the marginal likelihood for cheaper optimiser settings."""
import time
import numpy as np
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion

D = prepare("data/K03Q21N.lc", "K03Q21N")
args = (D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"])
pri = dict(ab_prior=("lower", 1.337, 0.055), bc_prior=("lower", 1.0, 0.05))
rng = np.random.default_rng(0)
thetas = [[rng.uniform(0, 360), np.degrees(np.arcsin(rng.uniform(-1, 1))),
           3.24826 + rng.normal(0, 0.0003), np.log(0.012)] for _ in range(25)]
ref = None
for cfg in [dict(nfacets=400, n_starts=4, n_stage1=50, n_inner=120),
            dict(nfacets=800, n_starts=4, n_stage1=50, n_inner=120),
            dict(nfacets=400, n_starts=3, n_stage1=40, n_inner=80),
            dict(nfacets=400, n_starts=2, n_stage1=30, n_inner=60)]:
    M = ConvexInversion(*args, **pri, **cfg)
    M.loglike(thetas[0])
    t = time.time()
    ll = np.array([M.loglike(th) for th in thetas])
    dt = (time.time() - t) / len(thetas) * 1000
    if ref is None:
        ref = ll
    d = (ll - ll.max()) - (ref - ref.max())
    print(cfg, "%.0f ms/call" % dt, "relative dlnL vs reference: rms %.2f max %.2f" % (np.std(d), np.abs(d).max()))
