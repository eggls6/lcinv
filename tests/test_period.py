"""Tests: multiband period search, period refinement and the search-range warnings.

    PYTHONPATH=. python tests/test_period.py

A synthetic double-peaked lightcurve of a fast rotator (P = 0.5 h) in two bands over five
nights. With the right range the period search recovers the period and gives no warning; with
the default range (P >= 2 h) it returns a wrong period, and `range_warnings` flags it.
"""
import numpy as np

from lcinv import period as per


def synthetic(P_h=0.5, n_nights=5, n_per_night=40, sig=0.01, seed=1):
    rng = np.random.default_rng(seed)
    t = np.concatenate([2460000.0 + k + np.sort(rng.uniform(0.0, 0.25, n_per_night))
                        for k in range(n_nights)])
    band = rng.integers(0, 2, len(t))
    alpha = np.linspace(8.0, 6.0, len(t))
    ph = 2 * np.pi * (t - t.mean()) * 24.0 / P_h
    mag = (0.15 * np.cos(2 * ph) + 0.03 * np.sin(ph) + 0.02 * np.cos(4 * ph + 0.4)
           + np.where(band == 1, 0.35, 0.0) + 0.03 * alpha + rng.normal(0.0, sig, len(t)))
    return t, mag, np.full(len(t), sig), band, alpha


def search(t, mag, err, band, alpha, pmin, pmax, K=6):
    P, chi2, nu = per.periodogram(t, mag, err, band, alpha, min(1.0, pmin), pmax, K=K, oversample=5)
    sel = P >= pmin
    P_best = per.local_minima(P[sel], chi2[sel], n=6)[0][0]
    ref = per.refine_period_auto(t, mag, err, band, alpha, P_best, K=K)
    return ref, per.range_warnings(P, chi2, nu, pmin, ref["red_chi2"])


def test_right_range():
    t, mag, err, band, alpha = synthetic()
    ref, warn = search(t, mag, err, band, alpha, 0.1, 1.0)
    assert abs(ref["P"] - 0.5) < 5 * max(ref["sigma"], 1e-6), ref["P"]
    assert ref["P_lo"] <= 0.5 <= ref["P_hi"]
    assert 0.6 < ref["red_chi2"] < 1.4, ref["red_chi2"]
    assert warn == [], warn
    amp, amp_err = per.amplitude(t, mag, err, band, alpha, ref["P"], n_boot=50)
    assert abs(amp - 0.33) < 0.03 and amp_err < 0.02, (amp, amp_err)
    print(f"right range: P = {ref['P']:.6f} h (+-{ref['sigma']:.1e}), chi2/nu {ref['red_chi2']:.2f}, "
          f"amplitude {amp:.3f}, no warning")


def test_default_range_is_flagged():
    t, mag, err, band, alpha = synthetic()
    ref, warn = search(t, mag, err, band, alpha, 2.0, 30.0)
    assert ref["P"] >= 2.0
    assert len(warn) == 2, warn
    assert "below --pmin" in warn[0] and "poor" in warn[1]
    print(f"default range: P = {ref['P']:.4f} h, chi2/nu {ref['red_chi2']:.1f}; warnings:")
    for w in warn:
        print("   ", w)


def test_ab_bound():
    # equator-on ellipsoid with a/b = 1.5 at zero phase: A(0) = 2.5 log10(1.5) = 0.440 mag
    d = per.ab_prior_from_amplitude(0.440, 0.001, alpha_mean=0.0, m_slope_err=0.0)
    assert abs(d["ab_min"] - 1.5) < 0.005, d
    print(f"a/b bound: {d['ab_min']:.4f} (expected 1.5)")


if __name__ == "__main__":
    test_right_range()
    test_default_range_is_flagged()
    test_ab_bound()
    print("all period tests passed")
