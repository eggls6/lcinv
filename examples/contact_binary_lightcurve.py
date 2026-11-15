"""Synthetic lightcurves: contact binary vs ellipsoid (Lommel-Seeliger + Lambert, with self-shadowing)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def dirs(n):
    i = np.arange(n) + 0.5; z = 1 - 2 * i / n; ph = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z); return np.c_[r * np.cos(ph), r * np.sin(ph), z]

def ell_ray(u, c, ax):
    """distance along rays u (from origin) to the far intersection with ellipsoid centre c, semi-axes ax"""
    A = np.sum((u / ax) ** 2, 1); B = -2 * np.sum(u * c / ax ** 2, 1); C = np.sum((c / ax) ** 2) - 1
    D = B * B - 4 * A * C
    t = np.where(D >= 0, (-B + np.sqrt(np.clip(D, 0, None))) / (2 * A), 0.0)
    return t

def mesh(kind, n=6000):
    u = dirs(n); tri = ConvexHull(u).simplices
    if kind == "contact binary":   # two overlapping ellipsoidal lobes, 1 : 0.8 size ratio
        r = np.maximum(ell_ray(u, np.array([-0.95, 0, 0]), np.array([1.0, 0.85, 0.8])),
                       ell_ray(u, np.array([0.85, 0, 0]), 0.8 * np.array([1.0, 0.85, 0.8])))
    else:                           # single ellipsoid with the same overall length and width
        r = 1.0 / np.sqrt(np.sum((u / np.array([1.85, 0.85, 0.8])) ** 2, 1))
    # orient triangles outward on the unit sphere (same orientation holds for the radial map)
    ua, ub, uc = u[tri[:, 0]], u[tri[:, 1]], u[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(ub - ua, uc - ua), ua + ub + uc) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    v = u * r[:, None]
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    nrm = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= 2 * area[:, None]
    cen = (a + b + c) / 3
    vol = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6
    s = (3 * abs(vol) / (4 * np.pi)) ** (1 / 3)          # scale to unit volume-equivalent radius
    return v / s, tri, cen / s, nrm, area / s ** 2

_rng = np.random.default_rng(0)
_SAMPLES = {}

def samples(m, k=40):
    """k random points per facet (barycentric), cached per mesh"""
    key = id(m)
    if key not in _SAMPLES:
        v, tri = m[0], m[1]
        r1, r2 = _rng.random((2, len(tri), k)); s1 = np.sqrt(r1)
        w = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], -1)               # (nf, k, 3)
        _SAMPLES[key] = np.einsum("fkj,fjd->fkd", w, v[tri])
    return _SAMPLES[key]

def visible(m, d, pix=0.02, tol=0.03):
    """fraction of each facet that faces d and is not hidden (point z-buffer along d)"""
    nrm = m[3]; pts = samples(m); nf, k, _ = pts.shape
    face = nrm @ d > 0
    e1 = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    P = pts[face].reshape(-1, 3)
    depth = P @ d
    key = np.floor(P @ e1 / pix).astype(np.int64) * 1000003 + np.floor(P @ e2 / pix).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    top = np.full(len(uk), -np.inf); np.maximum.at(top, inv, depth)
    mu = np.repeat(np.clip(nrm[face] @ d, 1e-3, 1), k)
    slack = tol + 2.0 * pix * np.sqrt(1 - mu ** 2) / mu          # surface slope within a pixel
    vis = (depth > top[inv] - np.minimum(slack, 0.5)).reshape(-1, k).mean(1)
    out = np.zeros(nf); out[face] = vis
    return out

def lightcurve(m, alpha_deg, aspect_deg=90.0, nphi=180, cl=0.1):
    v, tri, cen, nrm, area = m
    a = np.radians(alpha_deg); asp = np.radians(aspect_deg)
    # observer in the x-z plane at aspect angle from the spin axis (z), Sun rotated by alpha about z-ish
    eo = np.array([np.sin(asp), 0, np.cos(asp)])
    es = np.array([np.sin(asp) * np.cos(a), np.sin(asp) * np.sin(a), np.cos(asp)])
    es /= np.linalg.norm(es)
    out = []
    for ph in np.linspace(0, 2 * np.pi, nphi, endpoint=False):
        R = np.array([[np.cos(ph), np.sin(ph), 0], [-np.sin(ph), np.cos(ph), 0], [0, 0, 1]])   # body frame
        so, oo = R @ es, R @ eo
        mu0, mu = nrm @ so, nrm @ oo
        f = visible(m, so) * visible(m, oo)
        S = f * np.clip(mu, 0, None) * np.clip(mu0, 0, None) * (1 / (np.abs(mu) + np.abs(mu0) + 1e-12) + cl)
        out.append(np.sum(S * area))
    L = np.array(out); return -2.5 * np.log10(L / L.max())

M = {k: mesh(k) for k in ("contact binary", "ellipsoid")}
fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2.3], hspace=0.35, wspace=0.3)
col = {"contact binary": "#c4452f", "ellipsoid": "#2a6fb0"}
for i, (k, m) in enumerate(M.items()):
    v, tri, cen, nrm, area = m
    for j, (view, lab) in enumerate([(np.array([0, 0, 1.0]), "pole-on"), (np.array([0, -1.0, 0]), "equator-on")]):
        ax = fig.add_subplot(gs[i, j])
        e1 = np.array([1.0, 0, 0]); e2 = np.cross(view, e1)
        front = nrm @ view > 0
        order = np.argsort(cen[front] @ view)
        t = tri[front][order]
        P = np.c_[v @ e1, v @ e2]
        light = np.array([0.5, 0.3, 0.8]) ; light = light / np.linalg.norm(light)
        shade = np.clip(nrm[front][order] @ (light if j == 0 else np.array([0.4, -0.8, 0.45])), 0.05, 1)
        from matplotlib.collections import PolyCollection
        ax.add_collection(PolyCollection(P[t], facecolors=plt.cm.Greys_r(0.15 + 0.8 * shade), edgecolors="none"))
        ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.3, 1.3); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{k}\n{lab}" if j == 0 else lab, fontsize=8.5, color=col[k])
axl = fig.add_subplot(gs[:, 2])
ph = np.linspace(0, 2, 180, endpoint=False)
for k, m in M.items():
    for alpha, ls in ((5, "-"), (40, "--")):
        L = lightcurve(m, alpha)
        L2 = np.r_[L, L, L[0]]; x = np.r_[ph, ph + 2, 4]
        axl.plot(x, L2, ls, color=col[k], lw=1.6 if ls == "-" else 1.2,
                 label=f"{k}, phase angle {alpha}°  (amplitude {L.max() - L.min():.2f} mag)")
axl.invert_yaxis(); axl.set_xlim(0, 4)
axl.set_xticks(np.arange(0, 4.01, 0.5)); axl.set_xticklabels(["0" if t == 0 else ("π" if t == 1 else f"{t:g}π") for t in np.arange(0, 4.01, 0.5)])
axl.set_xlabel("rotational phase"); axl.set_ylabel("relative magnitude")
axl.set_title("equator-on view, Lommel–Seeliger + Lambert (c_L = 0.1), with self-shadowing", fontsize=9)
axl.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.33), ncol=1)
fig.savefig("contact_binary_lightcurve.png", dpi=150, bbox_inches="tight")
for k, m in M.items():
    for alpha in (5, 40):
        for asp in (60,):
            L = lightcurve(m, alpha, asp, nphi=90); print(k, alpha, asp, round(L.max() - L.min(), 3))
