"""From a lightcurve file to the arrays every model uses.

`prepare` reads the photometry, obtains the viewing geometry from JPL Horizons (cached),
applies the light-time correction and reduces the magnitudes to unit distances. The convex,
binary and hybrid models are all built from its dictionary; `subset`, `drop_epochs` and
`clip_outliers` filter it consistently.
"""
import numpy as np
import pandas as pd
from . import data, ephem


def prepare(lc_path, designation, location="X05", cache_dir="data", drop_bands=("u",),
            refresh=False, ades_opts=None, observer="station", step="1h", exptime=None,
            obstime="mid"):
    """Load photometry and geometry for one target.

    Parameters
    ----------
    lc_path : str
        Plain lightcurve or ADES file (see `lcinv.data.load_lightcurve`).
    designation : str
        Target designation, packed (``K03Q21N``) or unpacked (``2003 QN21``), or a number.
    location : str, optional
        MPC observatory code for plain lightcurves (ADES files carry one per row).
    cache_dir : str, optional
        Folder of cached Horizons tables; tables covering the epochs are reused.
    drop_bands : sequence of str, optional
        Bands to remove.
    refresh : bool, optional
        Query Horizons even if a cached table exists.
    ades_opts : dict, optional
        Options for `lcinv.ades.read_ades`.
    observer : {'station', 'geocentric', 'local'}, optional
        ``'station'``: one Horizons observer table per observatory code (ADES ``stn``; plain
        lightcurves use `location`). ``'geocentric'``: all rows observed from the geocentre
        (Horizons 500). ``'local'``: the geocentre from astropy's built-in Earth ephemeris, so only
        the asteroid's heliocentric vectors are queried. The geocentric approximations change the
        aspect angles by less than 10 arcsec (parallax < 4e-5 au).
    step : str, optional
        Horizons table step; the tables are interpolated to the epochs.
    exptime : float, optional
        Exposure time [s] for rows without one in the file (ADES ``exp`` or an ``exptime``
        column take precedence). Stored as ``D["exptime"]`` for exposure smearing; 0 means
        instantaneous.
    obstime : {'mid', 'start'}, optional
        Whether epochs are exposure mid-times (the ADES convention) or start times, which are then
        shifted by half the exposure.

    Returns
    -------
    dict
        Per-row arrays (one entry per point, sorted by epoch): ``t`` emission times [JD],
        ``mag`` reduced magnitudes ``m - 5 log10(r Delta)``, ``err`` uncertainties [mag],
        ``exptime`` [s], ``band_idx`` with the sorted labels ``bands``, ``alpha`` phase angles
        [deg], ``e_sun`` and ``e_obs`` unit vectors from the asteroid to the Sun and to the
        observer (ecliptic J2000, shape (N, 3)), ``night``. Also ``df`` (the filtered input rows),
        ``geo`` (full geometry including ``r`` and ``delta`` [au]), ``check`` (largest differences
        to the geometry columns of a plain file, if present), ``helio``, ``topo`` (Horizons
        tables; ``topo`` is a dict per station for several stations) and ``stations``.

    Examples
    --------
    >>> D = prepare("data/K03Q21N.lc", "K03Q21N", location="X05", cache_dir="data")
    >>> len(D["t"]), D["bands"]
    (334, ['g', 'i', 'r'])
    """
    d = data.load_lightcurve(lc_path, drop_bands=drop_bands, ades_opts=ades_opts)
    ex = pd.to_numeric(d["exptime"], errors="coerce") if "exptime" in d.columns else pd.Series(np.nan, index=d.index)
    if exptime is not None:
        ex = ex.fillna(float(exptime))
    d["exptime"] = ex.fillna(0.0).astype(float).values
    if obstime == "start":
        d["epoch"] = d["epoch"] + 0.5 * d["exptime"] / 86400.0
    elif obstime != "mid":
        raise ValueError(f"obstime must be 'mid' or 'start', not {obstime!r}")
    jd = d["epoch"].values
    if "stn" in d.columns and observer == "station":
        stations = sorted(d["stn"].unique())
    else:
        stations = [None]
    kw = dict(cache_dir=cache_dir, refresh=refresh, step=step)
    helio = ephem.get_helio(designation, jd, **kw)
    if observer == "local":
        topo = ephem.earth_topo_from_helio(helio)
        g = ephem.geometry(jd, helio, topo)
        stations = []
        return _finish(d, g, jd, helio, topo, stations)
    parts, topo_by = {}, {}
    for stn in stations:
        m = np.ones(len(d), bool) if stn is None else (d["stn"] == stn).values
        loc = location if stn is None else stn
        if "stn" in d.columns and observer == "geocentric":
            loc = "500"
        suffix = None if (stn is None and "stn" not in d.columns) else loc
        topo = ephem.get_topo(designation, jd[m], location=loc, cache_suffix=suffix, **kw)
        topo_by[loc] = topo
        parts[stn] = (m, ephem.geometry(jd[m], helio, topo))
    keys = ("t_emit", "e_sun", "e_obs", "r", "delta", "alpha")
    g = {k: np.zeros((len(d), 3)) if k in ("e_sun", "e_obs") else np.zeros(len(d)) for k in keys}
    for m, gg in parts.values():
        for k in keys:
            g[k][m] = gg[k]
    topo = topo_by[next(iter(topo_by))] if len(topo_by) == 1 else topo_by
    return _finish(d, g, jd, helio, topo, [s for s in stations if s is not None])


def _finish(d, g, jd, helio, topo, stations):
    bidx, blabels = data.band_index(d["band"].values)
    red = d["mag"].values - 5 * np.log10(g["r"] * g["delta"])
    return dict(df=d, geo=g, t=g["t_emit"], mag=red, err=d["rmsmag"].values,
                exptime=d["exptime"].values if "exptime" in d.columns else np.zeros(len(d)),
                band_idx=bidx, bands=blabels, alpha=g["alpha"], e_sun=g["e_sun"],
                e_obs=g["e_obs"], night=data.night_index(jd),
                check=ephem.check_against_file(g, d), helio=helio, topo=topo,
                stations=stations)


_ROW_KEYS = ("t", "mag", "err", "band_idx", "alpha", "e_sun", "e_obs", "night", "exptime")


def subset(D, keep):
    """Copy of a `prepare` dictionary restricted to some rows.

    Band and night indices are recomputed, so groups that lose all their rows leave no empty
    offset parameter.

    Parameters
    ----------
    D : dict
        Dictionary from `prepare`.
    keep : array_like of bool
        Rows to keep.

    Returns
    -------
    dict
    """
    keep = np.asarray(keep, bool)
    out = dict(D)
    for k in _ROW_KEYS:
        out[k] = np.asarray(D[k])[keep]
    out["df"] = D["df"][keep].reset_index(drop=True)
    out["geo"] = {k: (np.asarray(v)[keep] if np.ndim(v) and len(v) == len(keep) else v)
                  for k, v in D["geo"].items()}
    # re-index bands and nights so that dropped groups leave no empty columns
    bidx, labels = data.band_index(out["df"]["band"].values)
    out["band_idx"], out["bands"] = bidx, labels
    out["night"] = data.night_index(out["df"]["epoch"].values)
    return out


def clip_outliers(D, P_h, K=6, nsig=5.0, sys_err=0.015, max_iter=5):
    """Flag outliers from the multiband Fourier fit at a given period.

    Iteratively removes points whose normalised residual deviates from the median by more than
    `nsig` robust standard deviations (MAD scale), refitting after each pass. This is
    ``--clip-sigma`` of the scripts.

    Parameters
    ----------
    D : dict
        Dictionary from `prepare`.
    P_h : float
        Rotation period [h].
    K : int, optional
        Fourier harmonics.
    nsig : float, optional
        Threshold in robust standard deviations.
    sys_err : float, optional
        Error floor added in quadrature [mag].
    max_iter : int, optional
        Maximum number of passes.

    Returns
    -------
    ndarray of bool
        True for points to keep.
    """
    from . import period as per
    keep = np.ones(len(D["t"]), bool)
    sig = np.hypot(D["err"], sys_err)
    for _ in range(max_iter):
        bidx, _ = data.band_index(np.asarray(D["band_idx"])[keep])
        _, _, _, r = per.fit_fourier(D["t"][keep], D["mag"][keep], sig[keep], bidx,
                                     D["alpha"][keep], P_h, K=K, exptime=D["exptime"][keep])
        z = r / sig[keep]
        s = 1.4826 * np.median(np.abs(z - np.median(z)))
        bad = np.abs(z - np.median(z)) > nsig * max(s, 1e-9)
        if not bad.any():
            break
        idx = np.where(keep)[0]
        keep[idx[bad]] = False
    return keep


def drop_epochs(D, epochs, tol_days=1e-7):
    """Remove rows at given epochs.

    Used to apply the outliers clipped in a global run (``summary.json`` ``clipped_epochs``) to
    later models, so that all of them see identical data.

    Parameters
    ----------
    D : dict
        Dictionary from `prepare`.
    epochs : sequence of float
        Epochs to remove [JD UTC, as in the file].
    tol_days : float, optional
        Matching tolerance [d].

    Returns
    -------
    dict
    """
    if not epochs:
        return D
    e = np.asarray(D["df"]["epoch"].values)
    bad = np.zeros(len(e), bool)
    for x in epochs:
        bad |= np.abs(e - x) < tol_days
    return subset(D, ~bad)


def dense_epochs(jd_lo, jd_hi, P_h, n_min=300, per_rotation=48, n_max=6000):
    """Epoch grid for plotting model curves, fine enough to resolve the rotation.

    Parameters
    ----------
    jd_lo, jd_hi : float
        Time range [JD].
    P_h : float
        Rotation period [h].
    n_min, n_max : int, optional
        Bounds on the number of points.
    per_rotation : int, optional
        Points per rotation.

    Returns
    -------
    ndarray
    """
    n = int(np.clip((jd_hi - jd_lo) * 24.0 / P_h * per_rotation, n_min, n_max))
    return np.linspace(jd_lo, jd_hi, n)


def plot_geometry(D, jd):
    """Geometry at arbitrary epochs, for plotting.

    With several observer tables the first station's is used; the differences are far below
    plotting resolution.

    Parameters
    ----------
    D : dict
        Dictionary from `prepare`.
    jd : ndarray
        Epochs [JD UTC].

    Returns
    -------
    dict
        As ``D["geo"]``: ``t_emit``, ``e_sun``, ``e_obs``, ``r``, ``delta``, ``alpha``.
    """
    topo = D["topo"]
    if isinstance(topo, dict):
        topo = next(iter(topo.values()))
    return ephem.geometry(jd, D["helio"], topo)
