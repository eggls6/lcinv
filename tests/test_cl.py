"""Tests of the fitted Lambert weight c_L of the scattering law S = mu mu0 [1/(mu+mu0) + c_L].

1. Regression: without --fit-cl the convex likelihood is unchanged (566130 stored max-likelihood).
2. Consistency: fit_cl with ln c_L = ln 0.1 equals the fixed-c_L model (convex and hybrid).
3. Recovery: synthetic convex data (566130 geometry) generated with c_L = 0.1 and c_L = 3; the
   profile likelihood over a c_L grid (shape re-fitted at the true spin) contains the truth
   within Delta ln L < 2. At 4-7 deg phase angle values c_L <~ 0.1 are nearly degenerate
   (Lommel-Seeliger-dominated), while Lambert-dominated values are well separated.
4. Hybrid gradients with respect to ln c_L are finite.
"""
import json

import numpy as np
import jax.numpy as jnp

from lcinv import sampler
from lcinv.setup_data import prepare, drop_epochs
from lcinv.convex import ConvexInversion

P_H = 0.1779775


def load():
    s = json.load(open("results/566130_X05/summary.json"))
    R = sampler.load("results/566130_X05/nested_results.pkl")
    D = drop_epochs(prepare("data/566130_ades.csv", "566130", observer="local",
                            ades_opts=dict(stations=["X05"])), s["clipped_epochs"])
    return s, R["args"], D


try:  # fixtures for pytest; the script entry point below does not need pytest
    import pytest

    @pytest.fixture(scope="module")
    def _loaded():
        return load()

    @pytest.fixture(scope="module")
    def s(_loaded):
        return _loaded[0]

    @pytest.fixture(scope="module")
    def a(_loaded):
        return _loaded[1]

    @pytest.fixture(scope="module")
    def D(_loaded):
        return _loaded[2]
except ImportError:
    pass


def convex(D, a, s, mag=None, **kw):
    return ConvexInversion(D["t"], D["mag"] if mag is None else mag, D["err"], D["band_idx"], D["alpha"],
                           D["e_sun"], D["e_obs"], lmax=a["lmax"], nfacets=a["nfacets"], n_inner=a["n_inner"],
                           n_starts=a["n_starts"], n_stage1=a["n_stage1"], tau0=a["tau0"],
                           ab_prior=tuple(s["priors"]["ab"]), bc_prior=tuple(s["priors"]["bc"]), **kw)


def test_regression_and_consistency(s, a, D):
    th = np.array(s["max_likelihood"]["theta"])
    M = convex(D, a, s)
    Mf = convex(D, a, s, fit_cl=True)
    l0 = float(M.loglike_jit(th)["logL"])
    l1 = float(Mf.loglike_jit(np.r_[th, np.log(0.1)])["logL"])
    print(f"convex: stored ln L {s['max_likelihood']['logL']:.6f}, fixed c_L {l0:.6f}, fitted c_L=0.1 {l1:.6f}")
    assert abs(l0 - s["max_likelihood"]["logL"]) < 1e-6 and abs(l1 - l0) < 1e-6


def test_recovery(s, a, D):
    rng = np.random.default_rng(3)
    lam, bet = 300.0, 30.0
    Mt = convex(D, a, s, fit_cl=True)
    c = np.zeros(Mt.k)
    c[int(np.where(Mt.ls == 2)[0][-1])] = -0.8
    c += rng.normal(0, 0.3, Mt.k) * (Mt.ls >= 3)
    grid = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    for c_true in (0.1, 3.0):
        S = Mt.scattering_matrix(lam, bet, P_H, 0.0, c_true)
        y = -2.5 * np.log10(np.asarray(S @ Mt.areas(jnp.asarray(c)))) + rng.normal(0, 0.012, len(D["t"]))
        Mf = ConvexInversion(D["t"], y, np.full(len(y), 0.012), D["band_idx"], D["alpha"], D["e_sun"], D["e_obs"],
                            lmax=a["lmax"], nfacets=a["nfacets"], n_inner=150, n_starts=3, n_stage1=40,
                            tau0=a["tau0"], fit_cl=True)
        ll = np.array([float(Mf.loglike_jit(np.array([lam, bet, P_H, np.log(0.003), np.log(g)]))["logL"]) for g in grid])
        best = grid[np.argmax(ll)]
        print(f"recovery, true c_L = {c_true}: profile ln L - max = "
              + ", ".join(f"{g:g}:{v - ll.max():.1f}" for g, v in zip(grid, ll)) + f"  -> best {best:g}")
        i = int(np.argmin(np.abs(np.log(grid / c_true))))
        inside = grid[ll - ll.max() > -2.0]
        print(f"   c_L values within 2 of max ln L: {inside.min():g} - {inside.max():g}")
        assert ll[i] - ll.max() > -2.0                      # truth inside the ~2-sigma profile region
        assert inside.max() / inside.min() < 100            # and the data do constrain c_L


def test_hybrid(D):
    from lcinv.hybrid.quadrature import make_grid
    from lcinv.hybrid.shape import RadiusField
    from lcinv.hybrid.model import LocalModel, lonlat_to_vec
    F = RadiusField(make_grid(5, nlat=16, nlon=32))
    pole = lonlat_to_vec(300.0, 30.0)
    M0 = LocalModel(D, F, pole, P_H, 1e-6)
    M1 = LocalModel(D, F, pole, P_H, 1e-6, fit_cl=True, ln_cl_prior=(np.log(0.3), 2.0))
    x0 = M0.initial_points(1)[0]
    x1 = np.r_[x0[:6], np.log(0.1), x0[6:]]
    p0 = float(jnp.sum(M0.loglike_terms(jnp.asarray(x0))[0]))
    p1 = float(jnp.sum(M1.loglike_terms(jnp.asarray(x1))[0]))
    g = np.asarray(M1.grad(jnp.asarray(x1)))
    print(f"hybrid: ndim {M0.ndim} -> {M1.ndim}; log-likelihood fixed {p0:.6f}, fitted c_L=0.1 {p1:.6f}; "
          f"d logp / d ln c_L = {g[6]:.3f}")
    assert abs(p0 - p1) < 1e-8 and np.all(np.isfinite(g))


if __name__ == "__main__":
    s, a, D = load()
    test_regression_and_consistency(s, a, D)
    test_hybrid(D)
    test_recovery(s, a, D)
    print("all c_L tests passed")
