"""Quaternion construction of the body frame (JAX), free of Euler-angle singularities.

Convention: unit quaternions q = (w, x, y, z), Hamilton product, R(q) acting on column
vectors (active rotation). The body frame at the reference epoch is the rotation R0 that
maps ecliptic vectors to body coordinates, with the spin axis as body z:

    R0(p) = R(q_k) R(q_t(p)),
    q_t(p) = minimal rotation taking the current pole p onto the mode's reference pole p_k
           = normalize( 1 + p.p_k ,  p x p_k ),
    q_k    = fixed rotation taking p_k onto +z (any fixed choice; it only sets a constant
             orientation of the body frame about its spin axis).

Hence R0(p) p = R(q_k) p_k = z. The tilt quaternion q_t transports the frame along the great
circle between p and p_k. Moving the pole by dp twists the body frame about its spin axis by at
most tan(d/2) |dp|, where d is the angular distance from the mode centre p_k (zero for radial
moves and at the centre; the connection form of geodesic polar coordinates centred on p_k).
The Euler construction Rz(phi) Ry(90-beta) Rz(lambda) is the same construction centred on the
ecliptic pole instead, with twist rate tan(beta), which diverges near the ecliptic poles.
q_t is smooth everywhere except at p = -p_k, which lies outside the hemisphere covered by the
mode's gnomonic chart.

The remaining rotational degree of freedom, the twist about the spin axis, is the
rotational-phase gauge: it is absorbed by the m != 0 shape harmonics and is fixed (phase 0
at t_ref). Time evolution: R(t) = Rz(phi(t)) R0, phi(t) = 2 pi (t - t_ref) / P.
"""
import numpy as np
import jax.numpy as jnp


def qmul(a, b):
    """Hamilton product of quaternions ``(w, x, y, z)``."""
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return jnp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qnormalize(q):
    """Quaternion scaled to unit norm."""
    return q / jnp.linalg.norm(q)


def qmatrix(q):
    """Rotation matrix of a unit quaternion (active rotation of column vectors)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def q_between(a, b):
    """Minimal rotation taking unit vector a onto unit vector b (half-angle formula, no trig)."""
    return qnormalize(jnp.concatenate([jnp.atleast_1d(1.0 + jnp.dot(a, b)), jnp.cross(a, b)]))


def q_between_np(a, b):
    """NumPy version of `q_between`, robust for antiparallel vectors."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    d = float(a @ b)
    if d < -1 + 1e-12:  # antiparallel: rotate by pi about any perpendicular axis
        ax = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(ax) < 1e-8:
            ax = np.cross(a, [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        return np.r_[0.0, ax]
    q = np.r_[1.0 + d, np.cross(a, b)]
    return q / np.linalg.norm(q)


class ParallelFrame:
    """Body frame of a spin state, built by parallel transport from its reference pole.

    The rotation taking a pole ``p`` onto z is the reference rotation composed with the minimal
    rotation from ``p_k`` to ``p``, so moving the pole never twists the frame about the spin
    axis at the reference pole (see the validation of ``test_quaternion_frame.py``).

    Parameters
    ----------
    p_k : array_like
        Reference pole (ecliptic unit vector).
    """

    def __init__(self, p_k):
        self.p_k = jnp.asarray(np.asarray(p_k, float) / np.linalg.norm(p_k))
        self.q_k = jnp.asarray(q_between_np(np.asarray(self.p_k), [0.0, 0.0, 1.0]))

    def R0(self, p):
        """Rotation matrix taking pole `p` onto the body z axis."""
        return qmatrix(qmul(self.q_k, q_between(p, self.p_k)))

    def to_body(self, e, p, P_h, t, t_ref):
        """Ecliptic unit vectors e (N,3) -> body frame at times t (days)."""
        v = e @ self.R0(p).T
        ph = 2.0 * jnp.pi * (t - t_ref) * 24.0 / P_h
        c, s = jnp.cos(ph), jnp.sin(ph)
        # passive rotation by phi about z (the body rotates prograde about p)
        return jnp.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)
