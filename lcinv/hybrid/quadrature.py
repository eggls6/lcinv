"""Gauss-Legendre x uniform-longitude grid on the sphere, real spherical harmonics that are
exactly orthonormal under the grid quadrature, and a closed triangle mesh of the grid.

For nlat Gauss-Legendre nodes in cos(theta) and nlon equispaced longitudes, the quadrature
integrates exactly all products Y_lm Y_l'm' with l, l' <= lmax provided
nlat >= lmax + 1 and nlon >= 2 lmax + 1. The discrete basis is therefore orthonormal by
construction (no Gram-Schmidt across degrees, so degree damping and the rotation
equivariance about the z axis are preserved).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import sph_harm_y


def real_sph_harm(lmax, theta, phi, lmin=0):
    """Real orthonormal SH for lmin <= l <= lmax. Returns (Y[npts, nbasis], l[nbasis], m[nbasis])."""
    cols, ls, ms = [], [], []
    for l in range(lmin, lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m > 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.real)
            elif m < 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.imag)
            else:
                cols.append(Y.real)
            ls.append(l)
            ms.append(m)
    return np.stack(cols, axis=1), np.array(ls), np.array(ms)


@dataclass
class SphereGrid:
    """Gauss-Legendre x uniform grid on the sphere with its real spherical-harmonic basis and surface mesh.

    Built by `make_grid`; the attributes are listed in the class body.
    """
    lmax: int
    lmin: int
    nlat: int
    nlon: int
    u_q: np.ndarray       # (nq, 3) quadrature directions
    w_q: np.ndarray       # (nq,) quadrature weights (sum = 4 pi)
    Y_q: np.ndarray       # (nq, k) basis on quadrature points
    u_m: np.ndarray       # (nm, 3) mesh vertex directions (quadrature points + 2 poles)
    Y_m: np.ndarray       # (nm, k) basis on mesh vertices
    tri: np.ndarray       # (nt, 3) outward-oriented triangles
    l: np.ndarray
    m: np.ndarray
    nbr: np.ndarray       # (nq, 4) ring neighbours of each quadrature vertex (for convexity)

    @property
    def k(self):
        """Number of basis functions (shape coefficients)."""
        return self.Y_q.shape[1]


def make_grid(lmax=6, lmin=2, nlat=None, nlon=None):
    """Quadrature grid and basis for degrees ``lmin..lmax``.

    The default resolution (``nlat = max(24, 4 (lmax + 1))``, ``nlon = 2 nlat``) is four times the
    exactness limit, enough for an accurate facet sum.

    Parameters
    ----------
    lmax, lmin : int, optional
    nlat, nlon : int, optional

    Returns
    -------
    SphereGrid
    """
    nlat = nlat or max(24, 4 * (lmax + 1))
    nlon = nlon or 2 * nlat
    if nlat < lmax + 1 or nlon < 2 * lmax + 1:
        raise ValueError("grid too coarse for exact orthonormality")
    x, wx = np.polynomial.legendre.leggauss(nlat)     # x = cos(theta), ascending
    theta = np.arccos(x)
    phi = 2 * np.pi * np.arange(nlon) / nlon
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    u_q = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    w_q = (wx[:, None] * np.full(nlon, 2 * np.pi / nlon)[None, :]).reshape(-1)
    Y_q, l, m = real_sph_harm(lmax, TH.reshape(-1), PH.reshape(-1), lmin=lmin)

    # mesh: rings (south -> north since x ascending) plus two pole vertices
    nq = nlat * nlon
    south, north = nq, nq + 1
    u_m = np.vstack([u_q, [0, 0, -1], [0, 0, 1]])
    th_m = np.r_[TH.reshape(-1), np.pi, 0.0]
    ph_m = np.r_[PH.reshape(-1), 0.0, 0.0]
    Y_m, _, _ = real_sph_harm(lmax, th_m, ph_m, lmin=lmin)
    idx = lambda i, j: i * nlon + (j % nlon)
    tris = []
    for i in range(nlat - 1):
        for j in range(nlon):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            tris += [(a, b, d), (a, d, c)]
    for j in range(nlon):
        tris.append((south, idx(0, j + 1), idx(0, j)))
        tris.append((north, idx(nlat - 1, j), idx(nlat - 1, j + 1)))
    tri = np.array(tris)
    # orient outward on the unit sphere
    p = u_m[tri]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = np.sum(nrm * p.mean(1), axis=1) < 0
    tri[flip] = tri[flip][:, ::-1]
    # ring neighbours (east, west, south, north) for a vertex-based convexity measure
    nbr = np.zeros((nq, 4), int)
    for i in range(nlat):
        for j in range(nlon):
            nbr[idx(i, j)] = [idx(i, j + 1), idx(i, j - 1),
                              idx(i - 1, j) if i > 0 else south,
                              idx(i + 1, j) if i < nlat - 1 else north]
    return SphereGrid(lmax, lmin, nlat, nlon, u_q, w_q, Y_q, u_m, Y_m, tri, l, m, nbr)


def orthonormality_error(g):
    """Largest deviation of the basis Gram matrix under the grid quadrature from the identity."""
    G = g.Y_q.T @ (g.w_q[:, None] * g.Y_q)
    return float(np.max(np.abs(G - np.eye(G.shape[0]))))
