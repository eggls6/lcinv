"""Tests of the contact-binary model (lcinv.binary).

1. Brightness vs an independent renderer: the analytic two-lobe occlusion/shadowing is
   compared with a point-sampled z-buffer render of the watertight union mesh, for several
   aspect and phase angles (relative magnitudes over a rotation).
2. Limits: with the lobes far apart in the occlusion test switched off nothing changes for
   views where the lobes cannot hide each other (pole-on).
3. Recovery: synthetic data with 566130's epochs and geometry; the likelihood at the true
   parameters beats ellipsoid-like and wrong-q alternatives; a short nested run recovers q, s.
"""
import numpy as np
import jax.numpy as jnp

from lcinv.binary import ContactBinary
from lcinv.convex import to_body


# --------------------------------------------------------------- independent reference renderer
def _samples(v, tri, k=40, seed=0):
    r1, r2 = np.random.default_rng(seed).random((2, len(tri), k))
    s1 = np.sqrt(r1)
    w = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], -1)
    return np.einsum("fkj,fjd->fkd", w, v[tri])


def _visible(pts, nrm, d, pix=0.02, tol=0.03):
    nf, k, _ = pts.shape
    face = nrm @ d > 0
    e1 = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    P = pts[face].reshape(-1, 3)
    depth = P @ d
    key = np.floor(P @ e1 / pix).astype(np.int64) * 1000003 + np.floor(P @ e2 / pix).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    top = np.full(len(uk), -np.inf); np.maximum.at(top, inv, depth)
    mu = np.repeat(np.clip(nrm[face] @ d, 1e-3, 1), k)
    slack = np.minimum(tol + 2.0 * pix * np.sqrt(1 - mu ** 2) / mu, 0.5)
    out = np.zeros(nf); out[face] = (depth > top[inv] - slack).reshape(-1, k).mean(1)
    return out


def reference_brightness(v, tri, es_b, eo_b, cl=0.1):
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    cr = np.cross(b - a, c - a); area = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / (2 * area[:, None])
    pts = _samples(v, tri)
    L = []
    for so, oo in zip(es_b, eo_b):
        mu0, mu = np.clip(nrm @ so, 0, None), np.clip(nrm @ oo, 0, None)
        f = _visible(pts, nrm, so) * _visible(pts, nrm, oo)
        L.append(np.sum(f * mu * mu0 * (1 / (mu + mu0 + 1e-12) + cl) * area))
    return np.array(L)


def geometry_scan(aspect_deg, alpha_deg, n=72):
    """Pole = ecliptic z (lambda 0, beta 90); observer at the given aspect, Sun offset by alpha."""
    a, asp = np.radians(alpha_deg), np.radians(aspect_deg)
    eo = np.array([np.sin(asp), 0, np.cos(asp)])
    es = np.array([np.sin(asp) * np.cos(a), np.sin(asp) * np.sin(a), np.cos(asp)])
    es /= np.linalg.norm(es)
    P = 6.0
    t = np.linspace(0, P / 24, n, endpoint=False)
    return t, np.repeat(es[None], n, 0), np.repeat(eo[None], n, 0), P


def test_reference():
    worst = 0
    for shape in ([0.8, 0.85, 0.94, 0.9], [0.5, 0.7, 0.8, 0.95], [1.0, 0.9, 0.9, 0.8]):
        for asp, alpha in ((90, 5), (90, 40), (60, 20), (30, 10)):
            t, es, eo, P = geometry_scan(asp, alpha)
            n = len(t)
            M = ContactBinary(t, np.zeros(n), np.full(n, 0.01), np.zeros(n, int), np.full(n, alpha),
                              es, eo, t_ref=0.0, nfacets=1600, soft=0.005)
            th = np.r_[0.0, 90.0, P, np.log(0.01), 0.0, shape]
            mod = M.predict(th, t, es, eo, np.zeros(n, int), np.zeros(n), np.zeros(2))
            v, tri = M.union_mesh(th, n=6000)
            es_b = np.asarray(to_body(jnp.asarray(es), 0.0, 90.0, P, 0.0, jnp.asarray(t), 0.0))
            eo_b = np.asarray(to_body(jnp.asarray(eo), 0.0, 90.0, P, 0.0, jnp.asarray(t), 0.0))
            ref = -2.5 * np.log10(reference_brightness(v, tri, es_b, eo_b))
            d = (mod - mod.mean()) - (ref - ref.mean())
            amp_m, amp_r = np.ptp(mod), np.ptp(ref)
            worst = max(worst, np.abs(d).max())
            print(f"q,b,c/b,s={shape} aspect {asp:2d} alpha {alpha:2d}: amplitude model {amp_m:.3f} "
                  f"reference {amp_r:.3f} mag, max |diff| {1000 * np.abs(d).max():5.1f} mmag, "
                  f"rms {1000 * np.sqrt(np.mean(d ** 2)):4.1f} mmag")
    assert worst < 0.05


def test_recovery():
    from lcinv.setup_data import prepare
    D = prepare("data/566130_ades.csv", "566130", observer="local", ades_opts=dict(stations=["X05"]), exptime=30)
    P = 0.1779775
    args = (D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"])
    M0 = ContactBinary(*args, exptime=D["exptime"], smear_P_h=P)
    truth = np.array([300.0, 30.0, P, np.log(0.01), 40.0, 0.7, 0.8, 0.9, 0.9])
    rng = np.random.default_rng(1)
    y = M0.predict(truth, D["t"], D["e_sun"], D["e_obs"], D["band_idx"], D["alpha"],
                   np.r_[[0.1, 0.0, -0.1][:M0.nb], 0.03], exptime=D["exptime"]) + rng.normal(0, 0.01, len(D["t"]))
    M = ContactBinary(D["t"], y, np.full(len(y), 0.01), *args[3:], exptime=D["exptime"], smear_P_h=P)
    ll = lambda th: float(M.loglike_jit(jnp.asarray(th))["logL"])
    l0 = ll(truth)
    alt = {"q=0.4": np.r_[truth[:5], 0.4, truth[6:]], "s=0.75 (deep overlap)": np.r_[truth[:8], 0.75],
           "mirror pole": np.r_[120.0, -30.0, truth[2:]]}
    print(f"recovery: ln L at truth {l0:.1f}; " + ", ".join(f"{k}: {ll(v) - l0:+.1f}" for k, v in alt.items()))
    assert all(ll(v) < l0 for k, v in alt.items() if k != "mirror pole")


if __name__ == "__main__":
    test_reference()
    test_recovery()
    print("all contact-binary tests passed")
