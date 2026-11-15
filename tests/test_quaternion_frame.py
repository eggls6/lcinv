"""Checks of the quaternion parallel-transport body frame.

1. R0(p) is a proper rotation and maps the pole onto +z.
2. At the mode centre the quaternion and Euler frames differ only by a rotation about z
   (a constant twist = phase gauge), so the two models are physically equivalent.
3. Twist coupling: moving the pole by a small angle eps twists the Euler frame about the spin
   axis by ~ eps * tan(beta) (divergent at the ecliptic poles); the parallel-transport frame
   twists by at most eps * tan(d/2), d = distance from the mode centre (0 at the centre).
4. The two frames give the same MAP log density; the Laplace curvature of the quaternion
   frame shows whether the pole-shape ridge is gone (smallest eigenvalues, pole-shape
   correlations).
"""
import sys
import numpy as np
import jax.numpy as jnp

from lcinv.convex import _rz, _ry
from lcinv.hybrid.rotations import ParallelFrame
from lcinv.hybrid.model import tangent_basis, lonlat_to_vec


def euler_R0(lam, bet):
    return np.asarray(_ry(np.radians(90 - bet)) @ _rz(np.radians(lam)))


def rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-12:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def lonlat(p):
    return np.degrees(np.arctan2(p[1], p[0])) % 360, np.degrees(np.arcsin(p[2]))


def geometry_tests():
    rng = np.random.default_rng(0)
    for lam_k, bet_k in [(66, 71), (244, -73), (120, 89.5), (10, 5)]:
        pk = lonlat_to_vec(lam_k, bet_k)
        fr = ParallelFrame(pk)
        p0, e1, e2 = tangent_basis(pk)
        for _ in range(5):
            d = rng.normal(0, 0.4, 2)
            p = p0 + d[0] * e1 + d[1] * e2
            p /= np.linalg.norm(p)
            R = np.asarray(fr.R0(jnp.asarray(p)))
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12
            assert np.allclose(R @ p, [0, 0, 1], atol=1e-12)
        # gauge equivalence at the centre: R_q R_e^T is a rotation about z
        D = np.asarray(fr.R0(jnp.asarray(pk))) @ euler_R0(lam_k, bet_k).T
        assert abs(D[2, 2] - 1) < 1e-10, D
    print("geometry: proper rotations, pole -> +z, and Euler/quaternion frames differ by a z-rotation  OK")


def twist_tests():
    print("twist about the spin axis per unit pole displacement (eps = 0.5 deg):")
    eps = np.radians(0.5)
    for bet in (0, 45, 70, 80, 85, 89):
        pk = lonlat_to_vec(60, bet)
        fr = ParallelFrame(pk)
        p0, e1, e2 = tangent_basis(pk)
        worst_e, worst_q = 0, 0
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            p = p0 + np.tan(eps) * (np.cos(ang) * e1 + np.sin(ang) * e2)
            p /= np.linalg.norm(p)
            l2, b2 = lonlat(p)
            Re = euler_R0(l2, b2) @ euler_R0(60, bet).T
            Rq = np.asarray(fr.R0(jnp.asarray(p))) @ np.asarray(fr.R0(jnp.asarray(pk))).T
            # twist = component of the relative rotation about the new spin axis (body z)
            worst_e = max(worst_e, abs(rotvec(Re)[2]) / eps)
            worst_q = max(worst_q, abs(rotvec(Rq)[2]) / eps)
        print(f"  beta = {bet:2d} deg:  Euler {worst_e:8.3f}   quaternion {worst_q:8.5f}")
    print("  (Euler grows like tan(beta); the quaternion frame has no twist at the mode centre)")
    print("off-centre twist of the quaternion frame vs the bound tan(d/2):")
    pk = lonlat_to_vec(66, 85)
    fr = ParallelFrame(pk)
    p0, e1, e2 = tangent_basis(pk)
    h = 1e-4
    for dd in (0.1, 0.25, 0.5):
        worst = 0.0
        for psi in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            p1 = p0 + np.tan(dd) * (np.cos(psi) * e1 + np.sin(psi) * e2)
            p1 /= np.linalg.norm(p1)
            u = np.cross(p1, np.cross(p0, p1))
            u /= np.linalg.norm(u)
            for dv in (u, np.cross(p1, u)):
                p2 = p1 + h * dv
                p2 /= np.linalg.norm(p2)
                R2 = np.asarray(fr.R0(jnp.asarray(p2)))
                R = R2 @ np.asarray(fr.R0(jnp.asarray(p1))).T
                worst = max(worst, abs(rotvec(R) @ (R2 @ p2)) / h)
        print(f"  d = {dd:.2f} rad: max twist rate {worst:.4f}, tan(d/2) = {np.tan(dd / 2):.4f}")
        assert abs(worst - np.tan(dd / 2)) < 1e-3


def laplace_tests():
    from lcinv.setup_data import prepare
    from lcinv.hybrid.quadrature import make_grid
    from lcinv.hybrid.shape import RadiusField
    from lcinv.hybrid.model import LocalModel
    from lcinv.hybrid import modes as md
    D = prepare("data/K03Q21N.lc", "K03Q21N")
    g = make_grid(5, nlat=16, nlon=32)
    for name, pole in [("mode 1", (244, -73)), ("mode 2", (66, 71))]:
        for frame in ("euler", "parallel"):
            M = LocalModel(D, RadiusField(g, inertia_offdiag=0.005, inertia_order=0.005), lonlat_to_vec(*pole),
                           3.24826, 0.00142, frame=frame)
            x, lp, info = md.find_map(M, M.initial_points(4))
            L = md.laplace_evidence(M, x)
            cov = L["cov"]
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            pole_shape = np.abs(corr[:2, 6:]).max()
            H = np.linalg.inv(cov)
            ev = np.linalg.eigvalsh(H)
            lam, bet, P = [float(q) for q in M.spin(jnp.asarray(x))]
            print(f"{name} {frame:8s}: MAP logp {lp:8.2f} |grad| {info['grad_norm']:.1e}  pole ({lam:6.1f},{bet:6.1f})  "
                  f"ln Z {L['lnZ']:8.2f}  pole sd ({sd[0]:.3f},{sd[1]:.3f}) rad  max|corr(pole,shape)| {pole_shape:.3f}  "
                  f"min eig {ev[0]:.2f}")


def test_geometry():
    geometry_tests()


def test_twist():
    twist_tests()


if __name__ == "__main__":
    geometry_tests()
    twist_tests()
    if "--laplace" in sys.argv:
        laplace_tests()
