"""Multiband Fourier period search and lightcurve-derived priors.

Model for reduced magnitudes (m - 5 log10(r*Delta)):
    m_i = o_band(i) + s * alpha_i + sum_k [a_k cos(k w (t_i - t0)) + b_k sin(k w (t_i - t0))]
with w = 2 pi / P_rot.  The Fourier series has the *rotation* frequency as fundamental,
so a double-peaked lightcurve is represented with P = P_rot (even harmonics dominate).
All terms are linear, so chi2(P) is evaluated by weighted least squares on a grid.

Exposure smearing: with per-row exposure times T_i (s), harmonic k of row i is multiplied by
sinc(k w T_i / 2), the exact effect of averaging a sinusoid over the exposure. The fitted
coefficients then describe the un-smeared lightcurve (magnitude averaging; accurate to second
order in the amplitude, ample for period search).
"""
import numpy as np
from scipy.special import ndtri

from lcinv import smear


def _design(t, band_idx, nb, alpha, P_h, K, t0, exptime=None):
    w = 2 * np.pi * 24.0 / P_h
    ph = w * (t - t0)
    cols = [np.eye(nb)[band_idx], alpha[:, None]]
    for k in range(1, K + 1):
        att = 1.0 if exptime is None else smear.sinc_attenuation(k, exptime, P_h)[:, None]
        cols.append(np.c_[np.cos(k * ph), np.sin(k * ph)] * att)
    return np.hstack(cols)


def fit_fourier(t, mag, sig, band_idx, alpha, P_h, K=6, t0=None, exptime=None):
    """Weighted least-squares Fourier fit at one trial period.

    The model is ``m = o_band + s * alpha + sum_k [a_k cos(k w (t - t0)) + b_k sin(k w (t - t0))]``
    with ``w = 2 pi / P``; band offsets ``o_band`` and the linear phase slope ``s`` are fitted with
    the harmonics.

    Parameters
    ----------
    t : ndarray
        Light-time corrected epochs [JD].
    mag : ndarray
        Reduced magnitudes.
    sig : ndarray
        Uncertainties used as weights ``1 / sig`` [mag].
    band_idx : ndarray of int
        Band index of each point (0 .. n_bands - 1).
    alpha : ndarray
        Phase angles [deg].
    P_h : float
        Trial rotation period [h]; the fundamental of the series.
    K : int, optional
        Number of harmonics.
    t0 : float, optional
        Phase reference [JD]; default the mean epoch.
    exptime : float or ndarray, optional
        Exposure times [s]. Harmonic ``k`` is multiplied by its sinc attenuation, so the
        coefficients describe the un-smeared lightcurve. Zero or None: instantaneous.

    Returns
    -------
    chi2 : float
        Weighted sum of squared residuals.
    npar : int
        Number of fitted parameters (``n_bands + 1 + 2 K``).
    x : ndarray
        Coefficients: band offsets, phase slope, then ``a_1, b_1, ..., a_K, b_K``.
    resid : ndarray
        Residuals ``mag - model`` [mag].
    """
    t0 = np.mean(t) if t0 is None else t0
    nb = int(band_idx.max()) + 1
    if exptime is not None:
        exptime = np.broadcast_to(np.nan_to_num(np.asarray(exptime, float)), np.shape(t))
        if not np.any(exptime > 0):
            exptime = None
    A = _design(t, band_idx, nb, alpha, P_h, K, t0, exptime)
    w = 1.0 / sig
    x, *_ = np.linalg.lstsq(A * w[:, None], mag * w, rcond=None)
    r = mag - A @ x
    return float(np.sum((r * w) ** 2)), A.shape[1], x, r


def fourier_curve(x, nb, K, phase):
    """Rotational part of a fitted series at given rotational phases.

    Parameters
    ----------
    x : ndarray
        Coefficients from `fit_fourier`.
    nb : int
        Number of bands (offsets at the start of `x`).
    K : int
        Number of harmonics.
    phase : ndarray
        Rotational phases [rad].

    Returns
    -------
    ndarray
        Magnitude variation without band offsets and phase term.
    """
    y = np.zeros_like(phase)
    for k in range(1, K + 1):
        y += x[nb + 1 + 2 * (k - 1)] * np.cos(k * phase) + x[nb + 2 + 2 * (k - 1)] * np.sin(k * phase)
    return y


def periodogram(t, mag, sig, band_idx, alpha, P_min_h=2.0, P_max_h=30.0, K=6, oversample=10,
                exptime=None):
    """Chi-square of the multiband Fourier fit on a grid of rotation periods.

    The grid is uniform in frequency with step ``1 / (oversample * K * T)``, ``T`` the time span,
    so that each harmonic is sampled finely enough.

    Parameters
    ----------
    t, mag, sig, band_idx, alpha
        Data as in `fit_fourier`.
    P_min_h, P_max_h : float, optional
        Period range [h].
    K : int, optional
        Number of harmonics.
    oversample : int, optional
        Grid oversampling factor.
    exptime : float or ndarray, optional
        Exposure times [s] (see `fit_fourier`).

    Returns
    -------
    periods_h : ndarray
        Trial periods [h], decreasing (increasing frequency).
    chi2 : ndarray
        Chi-square at each period.
    nu : int
        Degrees of freedom of the fit.
    """
    T = t.max() - t.min()
    fmin, fmax = 24.0 / P_max_h, 24.0 / P_min_h  # cycles / day
    df = 1.0 / (oversample * K * T)
    freqs = np.arange(fmin, fmax, df)
    chi2 = np.empty_like(freqs)
    t0 = np.mean(t)
    for i, f in enumerate(freqs):
        chi2[i] = fit_fourier(t, mag, sig, band_idx, alpha, 24.0 / f, K, t0, exptime)[0]
    npar = fit_fourier(t, mag, sig, band_idx, alpha, 24.0 / freqs[0], K, t0, exptime)[1]
    return 24.0 / freqs, chi2, len(t) - npar


def local_minima(periods_h, chi2, n=8, min_sep_frac=0.01):
    """The lowest separated minima of a periodogram.

    Parameters
    ----------
    periods_h, chi2 : ndarray
        Periodogram from `periodogram`.
    n : int, optional
        Number of minima to return.
    min_sep_frac : float, optional
        Minimum fractional separation between returned periods.

    Returns
    -------
    list of (float, float)
        ``(period_h, chi2)`` pairs, best first.
    """
    order = np.argsort(chi2)
    sel = []
    for i in order:
        if all(abs(periods_h[i] - periods_h[j]) / periods_h[j] > min_sep_frac for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return [(periods_h[i], chi2[i]) for i in sel]


def range_warnings(periods_h, chi2, nu, P_min_h, red_chi2, poor_fit=3.0):
    """Signs that the true period lies outside the searched range.

    A sub-hour rotator searched with the default ``--pmin 2`` still yields a "best" period, and
    the jitter term then absorbs the unexplained signal, so a run can look normal at a wrong
    period. Two cheap checks on the periodogram catch this.

    Parameters
    ----------
    periods_h, chi2 : ndarray
        The periodogram as returned by `periodogram`; it may extend below `P_min_h` (the
        scripts compute it from ``min(1 h, --pmin)``).
    nu : int
        Degrees of freedom of the Fourier fit.
    P_min_h : float
        Shortest period that was accepted as a result.
    red_chi2 : float
        Reduced chi-square of the refined best fit (``refine_period(...)["red_chi2"]``).
    poor_fit : float, optional
        Reduced chi-square above which the Fourier fit is called poor.

    Returns
    -------
    list of str
        Human-readable warnings, empty if neither check fires.
    """
    out = []
    periods_h, chi2 = np.asarray(periods_h, float), np.asarray(chi2, float)
    sel = periods_h >= P_min_h
    if np.any(~sel) and np.any(sel) and chi2[~sel].min() < chi2[sel].min():
        i = int(np.argmin(np.where(sel, np.inf, chi2)))
        out.append(f"the periodogram has a lower minimum at {periods_h[i]:.4f} h "
                   f"(chi2/nu {chi2[i] / nu:.2f}) below --pmin {P_min_h:g} h; the period may be "
                   "shorter than the searched range (try a smaller --pmin)")
    if red_chi2 > poor_fit:
        out.append(f"the Fourier fit at the best period is poor (chi2/nu = {red_chi2:.2f}); "
                   "check the folded lightcurve, and widen --pmin/--pmax if it shows no clear "
                   "lightcurve")
    return out


def refine_grid(t, P_h, window=None, half_width=0.004, n=4001, phase_res=0.002, n_max=400001):
    """Half-width and number of points for the fine period scan.

    The default grid (+-0.4 %, 4001 points, i.e. steps of 2e-6 P) is fine for a few nights. Over a
    long baseline with ``N = span / P`` rotations, a step ``dP`` shifts the last rotation by
    ``N dP / P`` cycles, so the step must be below ``phase_res * P / N``. The number of points is
    raised to meet that (short data sets keep the defaults); `window` caps the half-width.

    Parameters
    ----------
    t : ndarray
        Epochs [JD].
    P_h : float
        Centre of the scan [h].
    window : float, optional
        Maximum fractional half-width (``--search-window``).
    half_width : float, optional
        Default fractional half-width.
    n : int, optional
        Default number of points.
    phase_res : float, optional
        Required phase resolution over the whole span [cycles].
    n_max : int, optional
        Upper limit on the number of points.

    Returns
    -------
    half_width : float
    n : int
    """
    t = np.asarray(t, float)
    n_rot = (t.max() - t.min()) * 24.0 / P_h
    hw = half_width if window is None else min(half_width, window)
    need = int(np.ceil(2.0 * hw * n_rot / phase_res)) + 1
    return hw, int(min(max(n, need), n_max))


def refine_period_auto(t, mag, sig, band_idx, alpha, P_guess_h, K=6, window=None, exptime=None,
                       max_zoom=4, min_steps=5):
    """Fine period scan that zooms in until the minimum is resolved.

    Runs `refine_period` on the grid from `refine_grid`, then zooms in (up to `max_zoom` times)
    while the ``Delta chi2 = 1`` interval is narrower than `min_steps` grid steps. For short data
    sets the first scan already resolves it, so the result equals the plain `refine_period`;
    for multi-apparition data it avoids scans of millions of points.

    Parameters
    ----------
    t, mag, sig, band_idx, alpha
        Data as in `fit_fourier`.
    P_guess_h : float
        Starting period, e.g. the best periodogram minimum [h].
    K : int, optional
        Number of harmonics.
    window : float, optional
        Maximum fractional half-width of the first scan.
    exptime : float or ndarray, optional
        Exposure times [s].
    max_zoom : int, optional
        Maximum number of zoom steps.
    min_steps : int, optional
        Grid steps required across the ``Delta chi2 = 1`` interval.

    Returns
    -------
    dict
        As `refine_period`.
    """
    hw, n = refine_grid(t, P_guess_h, window)
    P_c = P_guess_h
    ref = refine_period(t, mag, sig, band_idx, alpha, P_c, K=K, half_width=hw, n=n, exptime=exptime)
    for _ in range(max_zoom):
        step = 2.0 * hw * P_c / (n - 1)
        if ref["sigma"] >= min_steps * step:
            break
        width = max(50.0 * step, 3.0 * (ref["P_hi"] - ref["P_lo"]))
        P_c, n = ref["P"], 2001
        hw = width / P_c
        ref = refine_period(t, mag, sig, band_idx, alpha, P_c, K=K, half_width=hw, n=n,
                            exptime=exptime)
    return ref


def refine_period(t, mag, sig, band_idx, alpha, P_guess_h, K=6, half_width=0.004, n=4001,
                  exptime=None):
    """Fine scan around a periodogram minimum, with period uncertainties.

    Parameters
    ----------
    t, mag, sig, band_idx, alpha
        Data as in `fit_fourier`.
    P_guess_h : float
        Centre of the scan [h].
    K : int, optional
        Number of harmonics.
    half_width : float, optional
        Fractional half-width of the scan.
    n : int, optional
        Number of trial periods.
    exptime : float or ndarray, optional
        Exposure times [s].

    Returns
    -------
    dict
        ``P`` best period [h]; ``sigma`` the ``Delta chi2 = 1`` half-width after rescaling
        ``chi2_min / nu`` to 1; ``P_lo``, ``P_hi`` the more conservative interval
        ``chi2 < chi2_min (1 + sqrt(2 / nu))`` (the acceptance threshold common in lightcurve
        inversion); ``chi2``, ``nu``, ``red_chi2``; the fit (``coeffs``, ``resid``, ``K``, ``t0``);
        and the scan itself (``scan_P``, ``scan_chi2``).
    """
    Ps = np.linspace(P_guess_h * (1 - half_width), P_guess_h * (1 + half_width), n)
    t0 = np.mean(t)
    c = np.array([fit_fourier(t, mag, sig, band_idx, alpha, P, K, t0, exptime)[0] for P in Ps])
    i = int(np.argmin(c))
    chi2min, npar, x, r = fit_fourier(t, mag, sig, band_idx, alpha, Ps[i], K, t0, exptime)
    nu = len(t) - npar
    scale = chi2min / nu
    ok1 = Ps[c <= chi2min + scale]
    okT = Ps[c <= chi2min * (1 + np.sqrt(2.0 / nu))]
    return dict(P=Ps[i], sigma=0.5 * (ok1.max() - ok1.min()),
                P_lo=okT.min(), P_hi=okT.max(), chi2=chi2min, nu=nu,
                red_chi2=scale, coeffs=x, resid=r, K=K, t0=t0,
                scan_P=Ps, scan_chi2=c)


def amplitude(t, mag, sig, band_idx, alpha, P_h, K=6, n_boot=300, seed=1, exptime=None):
    """Peak-to-peak amplitude of the fitted lightcurve with a bootstrap error.

    Parameters
    ----------
    t, mag, sig, band_idx, alpha
        Data as in `fit_fourier`.
    P_h : float
        Rotation period [h].
    K : int, optional
        Number of harmonics.
    n_boot : int, optional
        Residual-bootstrap resamples for the error.
    seed : int, optional
        Random seed.
    exptime : float or ndarray, optional
        Exposure times [s]; the amplitude is then that of the un-smeared series.

    Returns
    -------
    amp : float
        Peak-to-peak amplitude at the mean phase angle [mag].
    amp_err : float
        Bootstrap standard deviation [mag].
    """
    rng = np.random.default_rng(seed)
    t0 = np.mean(t)
    nb = int(band_idx.max()) + 1
    ph = np.linspace(0, 2 * np.pi, 2000)
    chi2, npar, x, r = fit_fourier(t, mag, sig, band_idx, alpha, P_h, K, t0, exptime)
    y = fourier_curve(x, nb, K, ph)
    A0 = y.max() - y.min()
    model = mag - r
    amps = []
    for _ in range(n_boot):
        mb = model + rng.choice(r, size=len(r), replace=True)
        xb = fit_fourier(t, mb, sig, band_idx, alpha, P_h, K, t0, exptime)[2]
        yb = fourier_curve(xb, nb, K, ph)
        amps.append(yb.max() - yb.min())
    return float(A0), float(np.std(amps))


def ab_prior_from_amplitude(amp, amp_err, alpha_mean, m_slope=0.030, m_slope_err=0.010,
                            n_mc=20000, seed=2):
    """Lower bound on the equatorial axis ratio a/b from the amplitude.

    The amplitude is corrected to zero phase angle with the amplitude-phase relation
    ``A(0) = A(alpha) / (1 + m alpha)`` (Zappala et al. 1990), with ``m = 0.030 +- 0.010`` by
    default (typical of S types; about 0.015 for C and 0.013 for M types). For a triaxial
    ellipsoid seen equator-on ``A(0) = 2.5 log10(a/b)``; any other aspect gives a smaller
    amplitude, so the equator-on value is a lower bound. Errors are propagated by Monte Carlo.

    Parameters
    ----------
    amp, amp_err : float
        Amplitude and its error at the mean phase angle [mag].
    alpha_mean : float
        Mean phase angle [deg].
    m_slope, m_slope_err : float, optional
        Amplitude-phase slope and its uncertainty [mag/deg] (``--amp-phase-slope``).
    n_mc : int, optional
        Monte Carlo draws.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        ``A0``, ``A0_err`` (zero-phase amplitude), ``ab_min`` (lower bound on a/b),
        ``ln_ab_min``, ``ln_ab_min_err``.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(amp, amp_err, n_mc)
    m = np.clip(rng.normal(m_slope, m_slope_err, n_mc), 0.0, None)
    A0 = A / (1 + m * alpha_mean)
    ln_ab = 0.4 * np.log(10) * A0
    return dict(A0=float(np.mean(A0)), A0_err=float(np.std(A0)),
                ab_min=float(np.exp(np.mean(ln_ab))), ln_ab_min=float(np.mean(ln_ab)),
                ln_ab_min_err=float(np.std(ln_ab)))


def period_prior_transform(kind, P0, sigma=None, lo=None, hi=None, nsig=5.0):
    """Map the unit interval to the period prior.

    Parameters
    ----------
    kind : {'gaussian', 'uniform'}
        Truncated Gaussian (at +-`nsig` sigma) or uniform.
    P0 : float
        Centre [h].
    sigma : float, optional
        Width [h].
    lo, hi : float, optional
        Bounds of the uniform prior; default ``P0 -+ nsig * sigma``.
    nsig : float, optional
        Truncation (Gaussian) or half-width (uniform) in units of `sigma`.

    Returns
    -------
    callable
        ``u -> P`` for ``u`` in [0, 1].
    """
    if kind == "gaussian":
        from scipy.special import ndtr
        u_lo, u_hi = ndtr(-nsig), ndtr(nsig)
        return lambda u: P0 + sigma * ndtri(u_lo + u * (u_hi - u_lo))
    if kind == "uniform":
        lo = P0 - nsig * sigma if lo is None else lo
        hi = P0 + nsig * sigma if hi is None else hi
        return lambda u: lo + u * (hi - lo)
    raise ValueError(kind)
