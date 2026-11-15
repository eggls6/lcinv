"""Compare inner-optimiser variants at fixed spin states (phi0 gauge-fixed to 0)."""
import numpy as np
import jax
import jax.numpy as jnp
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion

D = prepare("data/K03Q21N.lc", "K03Q21N")
args = (D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"])
lnj = np.log(0.01)
thetas = [[48, 40, 3.24826], [72, 40, 3.24826], [180, 40, 3.24826], [120, -60, 3.24826], [120, 30, 3.24826]]
for bc in (None, ("lower", 1.0, 0.05)):
    M = ConvexInversion(*args, ab_prior=("lower", 1.337, 0.055), bc_prior=bc)
    for th in thetas:
        sig2 = M.err2 + np.exp(2 * lnj)
        S = M.scattering_matrix(th[0], th[1], th[2], 0.0)
        f = jax.jit(lambda c: M.objective(c, S, sig2))
        c_new = M.fit_shape(S, sig2)
        # long multi-start reference: 12 random starts x 600 iterations
        best = np.inf
        for s in range(12):
            c0 = jnp.asarray(np.random.default_rng(s).normal(0, 0.2, M.k))
            c, v = M._run_lbfgs(lambda c: M.objective(c, S, sig2), c0, 600)
            best = min(best, float(v))
        print("bc" if bc else "--", th[:2], "f(new) %.1f  f(ref) %.1f" % (float(f(c_new)), best))
