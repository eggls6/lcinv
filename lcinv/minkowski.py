"""Gaussian image (facet normals + areas) -> convex polyhedron.

Minkowski problem solved as a convex minimisation over support distances h_i:
    E(h) = sum_i A_i h_i - ln V(h),      dV/dh_i = F_i(h) (actual facet area),
E is convex (Brunn-Minkowski: ln V concave) and its minimum satisfies F_i / V = A_i,
i.e. the facet areas match the Gaussian image up to scale (Little 1983; Lamberg 1993).
The areas are first projected to exact closure (sum A_i n_i = 0), otherwise E is
unbounded under translations.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import HalfspaceIntersection, ConvexHull


def close_areas(normals, areas):
    """Smallest multiplicative correction ``A_i (1 + n_i . v)`` that closes the Gaussian image (``sum A_i n_i = 0``).

    Parameters
    ----------
    normals : ndarray
        Facet normals, (N, 3).
    areas : ndarray
        Facet areas, (N,).

    Returns
    -------
    ndarray
        Corrected areas.
    """
    M = (normals * areas[:, None]).T @ normals
    v = np.linalg.solve(M, -(areas @ normals))
    return np.clip(areas * (1.0 + normals @ v), 1e-12, None)


def _polytope(normals, h):
    hs = HalfspaceIntersection(np.c_[normals, -h], np.zeros(3))
    return ConvexHull(hs.intersections)


def facet_areas(normals, hull):
    """Area of each target facet in a realised polytope (zero where the facet vanished)."""
    tri = hull.points[hull.simplices]
    ta = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    owner = np.argmax(hull.equations[:, :3] @ normals.T, axis=1)
    F = np.zeros(len(normals))
    np.add.at(F, owner, ta)
    return F


def reconstruct(normals, areas, maxiter=2000):
    """Convex polyhedron with given facet normals and (relative) areas.

    Parameters
    ----------
    normals : ndarray
        Facet normals, (N, 3).
    areas : ndarray
        Facet areas, e.g. ``ConvexInversion.areas(c)``.
    maxiter : int, optional
        Iterations of the minimisation.

    Returns
    -------
    vertices : ndarray
        (V, 3), volume 1, centred on the vertex centroid; z is the spin axis.
    faces : ndarray of int
        (F, 3) triangles, zero-based.

    Examples
    --------
    >>> v, f = reconstruct(M.normals_np, np.asarray(M.areas(c)))
    >>> area_mismatch(M.normals_np, np.asarray(M.areas(c)), v)  # 0 = exact
    """
    A = close_areas(normals, areas)
    A = A / A.sum() * 3.0  # scale so the solution has V^(1/3) ~ O(1)

    def fun(h):
        try:
            hull = _polytope(normals, h)
        except Exception:
            return 1e10, np.zeros_like(h)
        F = facet_areas(normals, hull)
        V = hull.volume
        return float(A @ h - np.log(V)), A - F / V

    h0 = np.ones(len(normals))
    r = minimize(fun, h0, jac=True, method="L-BFGS-B", bounds=[(1e-3, None)] * len(h0),
                 options=dict(maxiter=maxiter, gtol=1e-9, ftol=1e-13))
    hull = _polytope(normals, r.x)
    verts = hull.points - hull.points[hull.vertices].mean(axis=0)
    verts = verts / hull.volume ** (1 / 3)
    hull = ConvexHull(verts)
    faces = hull.simplices.copy()
    # orient triangles outward using the hull plane normals
    for k, f in enumerate(faces):
        p = verts[f]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), hull.equations[k, :3]) < 0:
            faces[k] = f[::-1]
    return verts, faces


def area_mismatch(normals, areas, verts):
    """Half the L1 distance between the normalised target and realised facet areas (0 = exact, 1 =
    disjoint).
    """
    F = facet_areas(normals, ConvexHull(verts))
    A = close_areas(normals, areas)
    return float(np.sum(np.abs(F / F.sum() - A / A.sum())) / 2)


def write_obj(path, verts, faces, comment=""):
    """Write a triangle mesh as a Wavefront OBJ file.

    Parameters
    ----------
    path : str
    verts : ndarray
        (V, 3) vertices.
    faces : ndarray of int
        (F, 3) zero-based triangles (written one-based).
    comment : str, optional
        Header lines, each written after ``#``.
    """
    with open(path, "w") as f:
        for line in comment.splitlines():
            f.write(f"# {line}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def dimensions(verts):
    """Extents of a vertex set along body x, y and z (z = spin axis)."""
    return verts.max(axis=0) - verts.min(axis=0)
