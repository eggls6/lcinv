"""The convex likelihood from Python: a pole map at fixed period, then the shape at its peak."""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lcinv                                     # before jax: GPU preallocation off
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion
from lcinv import minkowski as mk

D = prepare("data/K03Q21N.lc", "K03Q21N", location="X05", cache_dir="data")
M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"],
                    D["e_sun"], D["e_obs"], lmax=4, nfacets=400)

# theta = (lambda [deg], beta [deg], P [h], ln jitter [mag]); the phase is gauge-fixed
P, lnsj = 3.24836, np.log(0.017)
t0 = time.time()
print(f"ln L at (60, +70): {M.loglike([60.0, 70.0, P, lnsj]):.2f}  (first call compiles: {time.time() - t0:.0f} s)")

# ln L on a 15-degree pole grid (shape optimised and marginalised at every point)
lams = np.arange(0.0, 360.0, 15.0)
bets = np.arange(-82.5, 90.0, 15.0)
t0 = time.time()
L = np.array([[M.loglike([lam, bet, P, lnsj]) for lam in lams] for bet in bets])
print(f"{L.size} poles in {time.time() - t0:.0f} s")
for hemi, m in (("north", bets > 0), ("south", bets < 0)):
    sub = np.where(m[:, None], L, -np.inf)
    i, j = np.unravel_index(np.argmax(sub), L.shape)
    print(f"best {hemi:5s} pole: lambda {lams[j]:5.1f}  beta {bets[i]:+5.1f}  ln L {L[i, j]:.2f}")

# the shape at the overall best pole: optimal coefficients -> convex polyhedron -> OBJ
i, j = np.unravel_index(np.argmax(L), L.shape)
theta = np.array([lams[j], bets[i], P, lnsj])
out = M.loglike_jit(theta)
areas = np.asarray(M.areas(out["c"]))
verts, faces = mk.reconstruct(M.normals_np, areas)
mk.write_obj("examples/docs/best_pole_shape.obj", verts, faces, comment="lcinv docs example")
x, y, z = np.sort(mk.dimensions(verts))[::-1]
print(f"shape: {len(verts)} vertices, extents a/b = {x / y:.2f}, b/c = {y / z:.2f}, "
      f"area mismatch {mk.area_mismatch(M.normals_np, areas, verts):.3f}")
fq = M.fit_quality(theta)
print(f"fit: chi2_red {fq['chi2_red_nominal']:.2f} (quoted errors) / {fq['chi2_red_jitter']:.2f} (with jitter), "
      f"rms {1e3 * fq['rms_mag']:.1f} mmag, p_eff {fq['p_eff']:.1f}")

fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.pcolormesh(np.r_[lams, 360] - 7.5, np.r_[bets - 7.5, 90], L - L.max(), cmap="magma",
                   vmin=-60, vmax=0, shading="flat")
ax.set_xlabel(r"pole longitude $\lambda$ [deg]"); ax.set_ylabel(r"pole latitude $\beta$ [deg]")
fig.colorbar(im, label=r"$\ln L - \ln L_{\max}$"); fig.tight_layout()
fig.savefig("examples/docs/pole_map.png", dpi=150)
