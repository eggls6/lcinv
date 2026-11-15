"""How narrow is the likelihood in rotational phase and period at a fixed pole?"""
import time
import numpy as np
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion
from lcinv.sampler import los_azimuth

D = prepare("data/K03Q21N.lc", "K03Q21N")
M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"],
                    ab_prior=("lower", 1.337, 0.055), bc_prior=("lower", 1.0, 0.05))
e = np.mean(D["e_obs"], 0)
e /= np.linalg.norm(e)
lam, bet, lnj = 120.0, 40.0, np.log(0.01)
az = los_azimuth(lam, bet, e)


def L(P, psi):
    return M.loglike([lam, bet, P, (az - psi) % (2 * np.pi), lnj])


t = time.time()
psis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
ll = np.array([L(3.24826, p) for p in psis])
print("psi scan %.0fs" % (time.time() - t))
print("coarse psi dlnL:", (ll - ll.max()).round(0))
pb = psis[ll.argmax()]
fine = pb + np.linspace(-0.3, 0.3, 31)
lf = np.array([L(3.24826, p) for p in fine])
print("psi fine dlnL (step 0.02 rad):", (lf - lf.max()).round(1))
Ps = 3.24826 + np.linspace(-0.004, 0.004, 21)
lp = np.array([L(P, fine[lf.argmax()]) for P in Ps])
print("P dlnL (step 0.0004 h):", (lp - lp.max()).round(1))
