"""Is the Laplace-marginalised likelihood a smooth function of the pole and period?"""
import time
import numpy as np
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion

D = prepare("data/K03Q21N.lc", "K03Q21N")
M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"],
                    ab_prior=("lower", 1.337, 0.055), bc_prior=("lower", 1.0, 0.05))
lnj = np.log(0.01)
M.loglike([0, 0, 3.24826, lnj])
t = time.time()
lams = np.arange(0, 360, 6.0)
ll = np.array([M.loglike([l, 40.0, 3.24826, lnj]) for l in lams])
print("per call %.1f ms" % ((time.time() - t) / len(lams) * 1000))
print("lambda scan (beta=40) dlnL:", (ll - ll.max()).round(1))
bets = np.arange(-84, 90, 6.0)
lb = np.array([M.loglike([120.0, b, 3.24826, lnj]) for b in bets])
print("beta scan (lambda=120) dlnL:", (lb - lb.max()).round(1))
Ps = 3.24826 + np.linspace(-0.002, 0.002, 21)
lp = np.array([M.loglike([lams[ll.argmax()], 40.0, P, lnj]) for P in Ps])
print("P scan dlnL:", (lp - lp.max()).round(1))
o = M.loglike_jit(np.array([lams[ll.argmax()], 40.0, 3.24826, lnj]))
print("best: chi2 %.1f grad %.2e" % (o["chi2"], o["grad_norm"]))
