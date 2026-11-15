"""Tests of exposure smearing (lcinv.smear and its use in the convex and hybrid models).

1. Quadrature: Gauss-Legendre averages of cos(k phi) over the exposure match the exact sinc
   attenuation within the error that choose_nodes promises.
2. No change without exposure times: Q = 1 reproduces the instantaneous models exactly.
3. Convex model, synthetic data for 566130's geometry (Rubin X05, P = 640.7 s, T = 30 s):
   data generated with Q = 64 nodes (reference integral); the model with the automatically
   chosen Q reproduces them to < 1 mmag.
   Fitting the shape at the true spin (informational): for a smooth lmax = 4 convex shape the
   smearing effect is a few mmag, so both fits reach the noise level.
4. Hybrid model: same reference comparison for the star-shaped facet sum.
5. Fourier period search: the sinc-attenuated design recovers the un-smeared amplitude.
"""
import numpy as np
import jax.numpy as jnp

from lcinv import smear, period as per
from lcinv.setup_data import prepare
from lcinv.convex import ConvexInversion

P_H = 0.1779775          # 566130 sidereal period (h)
T_S = 30.0               # exposure time (s)


def test_quadrature():
    dphi = smear.phase_sweep(T_S, P_H)
    print(f"rotation swept in {T_S:.0f} s: {np.degrees(dphi):.1f} deg")
    for k in (2, 6, 10, 14):
        Q = smear.choose_nodes([T_S], P_H, k, tol=1e-4)
        x, w = smear.gauss_legendre(Q)
        approx = np.sum(w * np.cos(k * dphi / 2 * x))
        exact = smear.sinc_attenuation(k, T_S, P_H)
        print(f"  k={k:2d}: attenuation {exact:.4f}, Q={Q}, GL error {abs(approx - exact):.1e}")
        assert abs(approx - exact) < 1e-4
    assert smear.choose_nodes([0.0], P_H, 10) == 1
    assert smear.choose_nodes([30.0], 12.0, 10, tol=1e-3) == 1   # slow rotator: midpoint suffices


def data_566130():
    D = prepare("data/566130_ades.csv", "566130", observer="local",
                ades_opts=dict(stations=["X05"]), exptime=T_S)
    assert np.allclose(D["exptime"], T_S)
    return D


try:  # fixture for pytest; the script entry point below does not need pytest
    import pytest

    @pytest.fixture(scope="module")
    def D():
        return data_566130()
except ImportError:
    pass


def convex_model(D, **kw):
    return ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"],
                           lmax=4, nfacets=400, n_inner=60, n_starts=3, n_stage1=30, tau0=0.6,
                           t_ref=float(np.mean(D["t"])), **kw)


def sharp_shape(M, seed=2):
    """Elongated, angular convex shape (large l = 2..4 coefficients)."""
    rng = np.random.default_rng(seed)
    c = np.zeros(M.k)
    i22 = int(np.where(M.ls == 2)[0][-1])
    c[i22] = -0.9
    c += rng.normal(0, 0.25, M.k) * (M.ls >= 3)
    return jnp.asarray(c)


def test_convex(D):
    theta = np.array([300.0, 30.0, P_H, np.log(0.012)])
    M0 = convex_model(D)                                     # instantaneous
    Mref = convex_model(D, exptime=D["exptime"], smear_nodes=64)
    Mq = convex_model(D, exptime=D["exptime"], smear_P_h=P_H)
    M1 = convex_model(D, exptime=np.zeros(len(D["t"])), smear_P_h=P_H)
    assert M1.Q == 1
    c = sharp_shape(M0)
    lam, bet, P = theta[:3]
    L = {name: np.asarray(m.scattering_matrix(lam, bet, P, 0.0) @ m.areas(c))
         for name, m in (("inst", M0), ("ref", Mref), ("auto", Mq), ("zero", M1))}
    dm = lambda a, b: 2.5 * np.log10(L[a] / L[b]) * 1000
    amp = 2.5 * np.log10(L["ref"].max() / L["ref"].min())
    print(f"convex: synthetic amplitude {amp:.2f} mag; auto Q = {Mq.Q}")
    print(f"  max |model - reference| [mmag]: instantaneous {np.abs(dm('inst', 'ref')).max():.1f}, "
          f"auto-Q {np.abs(dm('auto', 'ref')).max():.3f}, exptime=0 vs instantaneous {np.abs(dm('zero', 'inst')).max():.1e}")
    assert np.abs(dm("auto", "ref")).max() < 1.0
    assert np.abs(dm("zero", "inst")).max() < 1e-9

    # synthetic data (reference integral + 12 mmag noise), shape fitted at the true spin
    rng = np.random.default_rng(5)
    y = -2.5 * np.log10(L["ref"]) + 0.01 * D["band_idx"] + rng.normal(0, 0.012, len(L["ref"]))
    out = {}
    for name, kw in (("instantaneous", {}), ("smeared", dict(exptime=D["exptime"], smear_P_h=P_H))):
        m = ConvexInversion(D["t"], y, np.full(len(y), 0.012), D["band_idx"], D["alpha"], D["e_sun"],
                            D["e_obs"], lmax=4, nfacets=400, n_inner=150, n_starts=3, n_stage1=40,
                            tau0=0.6, t_ref=float(np.mean(D["t"])), **kw)
        best = max(((j, m.loglike_jit(jnp.asarray([lam, bet, P, j]))) for j in np.log([0.002, 0.006, 0.012, 0.02, 0.03])),
                   key=lambda z: float(z[1]["logL"]))
        o = best[1]
        rms = float(np.sqrt(np.mean((y - np.asarray(o["model"])) ** 2)))
        out[name] = (float(o["logL"]), rms, float(np.exp(best[0])))
        print(f"  fit with {name:13s} model: ln L {out[name][0]:8.2f}, residual rms {1000 * rms:5.1f} mmag, "
              f"best jitter {1000 * out[name][2]:.0f} mmag")
    # informational: a smooth lmax = 4 convex lightcurve has little power above k ~ 4, so
    # smearing changes it by only a few mmag and both fits reach the noise level


def test_hybrid(D):
    from lcinv.hybrid.quadrature import make_grid
    from lcinv.hybrid.shape import RadiusField
    from lcinv.hybrid.model import LocalModel, lonlat_to_vec
    grid = make_grid(5, nlat=16, nlon=32)
    F = RadiusField(grid)
    pole = lonlat_to_vec(300.0, 30.0)
    Mi = LocalModel({**D, "exptime": np.zeros(len(D["t"]))}, F, pole, P_H, 1e-6)
    Mr = LocalModel(D, F, pole, P_H, 1e-6, smear_nodes=64)
    Mq = LocalModel(D, F, pole, P_H, 1e-6)
    x = Mq.initial_points(1, amp=0.4)[0]
    x[Mq.NSPIN:] += np.random.default_rng(1).normal(0, 1.0, Mq.k)
    Li, Lr, Lq = (np.asarray(m.brightness(jnp.asarray(x))) for m in (Mi, Mr, Mq))
    d = lambda a, b: np.abs(2.5 * np.log10(a / b)).max() * 1000
    print(f"hybrid: auto Q = {Mq.Q}; max |model - reference| [mmag]: instantaneous {d(Li, Lr):.1f}, auto-Q {d(Lq, Lr):.3f}")
    assert d(Lq, Lr) < 1.0 and Mi.Q == 1


def test_fourier(D):
    t = D["t"]
    rng = np.random.default_rng(0)
    w = 2 * np.pi * 24 / P_H
    # sharp two-peaked lightcurve: harmonics 2, 4, 6, 8
    amps = {2: 0.06, 4: 0.04, 6: 0.03, 8: 0.02}
    sum(a * np.cos(k * w * (t - t.mean())) for k, a in amps.items())
    sm = sum(a * smear.sinc_attenuation(k, D["exptime"], P_H) * np.cos(k * w * (t - t.mean())) for k, a in amps.items())
    y = sm + rng.normal(0, 0.005, len(t))
    bi = np.zeros(len(t), int)
    alpha = np.zeros(len(t))
    for name, ex in (("without", None), ("with", D["exptime"])):
        x = per.fit_fourier(t, y, np.full(len(t), 0.005), bi, alpha, P_H, K=8, exptime=ex)[2]
        rec = {k: np.hypot(x[2 + 2 * (k - 1)], x[3 + 2 * (k - 1)]) for k in amps}
        print(f"Fourier {name:7s} sinc correction: " + ", ".join(f"k={k}: {rec[k]:.4f} (true {a})" for k, a in amps.items()))
    assert all(abs(rec[k] - a) < 0.004 for k, a in amps.items())


def test_real_amplitude(D):
    """566130 X05 data: Fourier amplitude and harmonic content with and without smearing."""
    from lcinv.setup_data import subset
    keep = np.abs(D["mag"] - np.median(D["mag"])) < 0.6           # drop the 1.2 mag outlier
    S = subset(D, keep)
    for name, ex in (("instantaneous", None), ("smeared (30 s)", S["exptime"])):
        A, dA = per.amplitude(S["t"], S["mag"], S["err"], S["band_idx"], S["alpha"], P_H, K=10, n_boot=100, exptime=ex)
        print(f"566130 Fourier (K=10) amplitude, {name:15s}: {A:.3f} +- {dA:.3f} mag")


if __name__ == "__main__":
    test_quadrature()
    D = data_566130()
    test_real_amplitude(D)
    test_fourier(D)
    test_convex(D)
    test_hybrid(D)
    print("all smearing tests passed")
