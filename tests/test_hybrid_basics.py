"""Unit checks for the hybrid layer: SH orthonormality, quadrature of volume/inertia,
mesh orientation, and the prior spectrum normalisation."""
import numpy as np
import jax.numpy as jnp

from lcinv.hybrid.quadrature import make_grid, orthonormality_error
from lcinv.hybrid.shape import RadiusField


def ellipsoid_radius(u, a, b, c):
    return 1.0 / np.sqrt((u[:, 0] / a) ** 2 + (u[:, 1] / b) ** 2 + (u[:, 2] / c) ** 2)


def main():
    g = make_grid(lmax=6)
    print("grid", g.nlat, "x", g.nlon, "k =", g.k, "orthonormality error %.2e" % orthonormality_error(g))
    assert orthonormality_error(g) < 1e-12

    F = RadiusField(g)
    a_, b_, c_ = 1.6, 1.1, 0.9
    r = ellipsoid_radius(g.u_q, a_, b_, c_)
    F.radius_quad = lambda a: jnp.asarray(r)  # inject an analytic ellipsoid
    V, cen, I = F.mass_properties(None)
    V_true = 4 / 3 * np.pi * a_ * b_ * c_
    I_true = V_true / 5 * np.array([b_ ** 2 + c_ ** 2, a_ ** 2 + c_ ** 2, a_ ** 2 + b_ ** 2])
    print("volume rel err %.2e, inertia rel err %s, centroid %s" % (
        abs(float(V) / V_true - 1), np.round(np.diag(np.asarray(I)) / I_true - 1, 6),
        np.round(np.asarray(cen), 8)))
    assert abs(float(V) / V_true - 1) < 1e-4
    assert np.allclose(np.diag(np.asarray(I)) / I_true, 1, atol=1e-3)

    # mesh orientation and closure on the unit sphere
    F2 = RadiusField(g)
    v, area, nrm = F2.mesh(jnp.zeros(g.k))
    cen_f = np.asarray(v)[g.tri].mean(1)
    assert np.all(np.sum(np.asarray(nrm) * cen_f, 1) > 0)
    closure = np.linalg.norm(np.sum(np.asarray(area)[:, None] * np.asarray(nrm), 0))
    print("mesh area %.4f (4pi=%.4f), closure %.1e" % (float(area.sum()), 4 * np.pi, closure))

    # prior spectrum: pointwise sd of log r equals sigma_f
    tau = np.asarray(F2.tau(np.log(0.15), np.log(3.0)))
    sd = np.sqrt(np.sum(tau ** 2) / (4 * np.pi))
    print("pointwise prior sd of log r: %.4f (target 0.15)" % sd)
    assert abs(sd - 0.15) < 1e-6
    print("all basic checks passed")


def test_basics():
    main()


if __name__ == "__main__":
    main()
