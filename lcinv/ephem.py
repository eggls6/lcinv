"""Viewing/illumination geometry from JPL Horizons.

Horizons is queried with astroquery on a regular TDB grid covering the observations
(heliocentric and observer-centred geometric vectors, ecliptic J2000 frame) and the
vectors are spline-interpolated to the observation epochs. Light-time is solved by
iteration, so the returned geometry refers to the asteroid at emission time.

The tables are cached as CSV (columns jd_tdb,x,y,z), so a run needs network access only once.

Data spanning several apparitions are handled in segments: epochs separated by more than
`gap_days` are queried as separate windows (hourly vectors only where there are data), and
each segment gets its own spline. Several observatories (ADES input) get one observer table
each (cache ..._topo_<stn>.csv); stations with no Horizons site code can be approximated by
the geocentre (500), whose parallax (< 4e-5 au) changes the aspect angles by < 10 arcsec.
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

C_AU_PER_DAY = 173.1446326742403  # speed of light, au / day
UTC_TO_TDB_DAYS = 69.184 / 86400.0  # TT-UTC since 2017 (37 s leap + 32.184 s); TDB-TT < 2 ms

# TAI-UTC steps since 1999 (JD UTC of the step, new value in s)
_LEAPS = [(2451179.5, 32), (2453736.5, 33), (2454832.5, 34), (2456109.5, 35),
          (2457204.5, 36), (2457754.5, 37)]


def utc_to_tdb_days(jd_utc):
    """TT-UTC in days for each epoch (leap-second table from 1999; TDB-TT ignored)."""
    jd = np.asarray(jd_utc, float)
    tai = np.full(jd.shape, 32.0)
    for jd0, v in _LEAPS:
        tai = np.where(jd >= jd0, v, tai)
    return (tai + 32.184) / 86400.0


def segments(jd, gap_days=2.0, pad_days=0.5):
    """Group sorted epochs into windows [start, stop] separated by more than gap_days."""
    jd = np.sort(np.asarray(jd, float))
    cut = np.where(np.diff(jd) > gap_days)[0] + 1
    return [(float(g[0] - pad_days), float(g[-1] + pad_days)) for g in np.split(jd, cut)]


class SegmentSpline:
    """Cubic splines of a Horizons vector table, one per contiguous segment.

    Gaps longer than `max_step_days` in the table's time grid start a new segment, so tables of
    several apparitions interpolate only within each window.

    Parameters
    ----------
    df : pandas.DataFrame
        Vector table with columns ``jd_tdb``, ``x``, ``y``, ``z`` [au].
    max_step_days : float, optional
        Largest grid step within a segment [d].
    """

    def __init__(self, df, max_step_days=1.0):
        t = df["jd_tdb"].values
        xyz = df[["x", "y", "z"]].values
        o = np.argsort(t)
        t, xyz = t[o], xyz[o]
        keep = np.r_[True, np.diff(t) > 1e-9]
        t, xyz = t[keep], xyz[keep]
        cut = np.where(np.diff(t) > max_step_days)[0] + 1
        self.parts = []
        for idx in np.split(np.arange(len(t)), cut):
            if len(idx) >= 4:
                self.parts.append((t[idx[0]], t[idx[-1]], CubicSpline(t[idx], xyz[idx])))

    def covers(self, tt, margin=0.0):
        """Whether each epoch lies inside a segment (with `margin` [d] to its ends).

        Parameters
        ----------
        tt : array_like
            Epochs [JD TDB].
        margin : float, optional

        Returns
        -------
        ndarray of bool
        """
        tt = np.atleast_1d(tt)
        ok = np.zeros(tt.shape, bool)
        for a, b, _ in self.parts:
            ok |= (tt >= a + margin) & (tt <= b - margin)
        return ok

    def __call__(self, tt):
        """Interpolated vectors at epochs `tt` [JD TDB], shape (N, 3) [au].

        Raises
        ------
        ValueError
            If an epoch lies outside the table; refresh the cache (``refresh=True``).
        """
        tt = np.atleast_1d(np.asarray(tt, float))
        out = np.full(tt.shape + (3,), np.nan)
        for a, b, f in self.parts:
            m = (tt >= a) & (tt <= b)
            out[m] = f(tt[m])
        if np.isnan(out).any():
            raise ValueError("epoch outside the Horizons tables (refresh the cache)")
        return out


def unpack_designation(des):
    """Turn an MPC packed designation into a Horizons search string.

    ``'K03Q21N'`` -> ``'2003 QN21'``; ``'91984'`` -> ``'91984'``; packed numbers and
    extended cycle counts are handled. Unpacked inputs are returned unchanged.

    Parameters
    ----------
    des : str

    Returns
    -------
    str
    """
    des = str(des).strip()
    if des.isdigit():
        return des
    m = re.fullmatch(r"([IJK])(\d\d)([A-HJ-Y])([0-9A-Za-z])(\d)([A-HJ-Z])", des)
    if not m:
        return des
    century = {"I": "18", "J": "19", "K": "20"}[m.group(1)]
    year = century + m.group(2)
    c = m.group(4)
    if c.isdigit():
        cycle = int(c) * 10 + int(m.group(5))
    elif c.isupper():
        cycle = (ord(c) - ord("A") + 10) * 10 + int(m.group(5))
    else:
        cycle = (ord(c) - ord("a") + 36) * 10 + int(m.group(5))
    return f"{year} {m.group(3)}{m.group(6)}{cycle if cycle else ''}"


def _horizons_vectors(target, location, start_tdb, stop_tdb, step):
    from astroquery.jplhorizons import Horizons

    # numbered asteroid -> 'smallbody' search on the number; otherwise a provisional designation
    id_type = "smallbody" if target.isdigit() else "designation"
    obj = Horizons(id=target, id_type=id_type, location=location,
                   epochs={"start": start_tdb, "stop": stop_tdb, "step": step})
    tab = obj.vectors(refplane="ecliptic", aberrations="geometric")
    return pd.DataFrame({"jd_tdb": np.asarray(tab["datetime_jd"], float),
                         "x": np.asarray(tab["x"], float),
                         "y": np.asarray(tab["y"], float),
                         "z": np.asarray(tab["z"], float)})


def _query_segments(target, location, segs, step):
    from astropy.time import Time
    fmt = "%Y-%m-%d %H:%M"
    out = []
    for t0, t1 in segs:
        a = Time(t0, format="jd", scale="utc").tdb
        b = Time(t1, format="jd", scale="utc").tdb
        out.append(_horizons_vectors(target, location, a.strftime(fmt), b.strftime(fmt), step))
    return pd.concat(out, ignore_index=True)


def _write_cache(path, df, target, center):
    with open(path, "w") as fh:
        fh.write(f"# JPL Horizons geometric vectors, target {target}, center {center}, "
                 "ecliptic J2000, au, JD TDB\n")
        df.to_csv(fh, index=False, float_format="%.16e")


def _cached_or_query(path, target, center, jd_utc, need_tdb, refresh, gap_days, pad_days, step):
    if path is not None and os.path.exists(path) and not refresh:
        df = pd.read_csv(path, comment="#")
        if SegmentSpline(df).covers(need_tdb).all():
            return df
    df = _query_segments(target, center, segments(jd_utc, gap_days, pad_days), step)
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _write_cache(path, df, target, center)
    return df


def get_helio(designation, jd_utc, cache_dir=None, step="1h", pad_days=0.5, refresh=False,
              gap_days=2.0):
    """Heliocentric asteroid vectors covering all epochs.

    Cached as ``<cache_dir>/horizons_<designation>_helio.csv``; a cache that does not cover the
    epochs (with a margin for the light time) is queried again.

    Parameters
    ----------
    designation : str
        Target, packed or unpacked.
    jd_utc : array_like
        Observation epochs [JD UTC].
    cache_dir : str, optional
        Cache folder; None disables caching.
    step : str, optional
        Horizons table step.
    pad_days : float, optional
        Padding of each queried window [d].
    refresh : bool, optional
        Ignore an existing cache.
    gap_days : float, optional
        Epochs further apart start a separate query window.

    Returns
    -------
    pandas.DataFrame
        Columns ``jd_tdb``, ``x``, ``y``, ``z`` (ecliptic J2000, au).
    """
    jd_utc = np.asarray(jd_utc, float)
    tt = jd_utc + utc_to_tdb_days(jd_utc)
    path = os.path.join(cache_dir, f"horizons_{designation}_helio.csv") if cache_dir else None
    # needed at emission times (up to ~0.03 d before reception): keep a margin
    return _cached_or_query(path, unpack_designation(designation), "500@10", jd_utc, tt - 0.05,
                            refresh, gap_days, pad_days, step)


def get_topo(designation, jd_utc, location="X05", cache_dir=None, step="1h", pad_days=0.5,
             refresh=False, gap_days=2.0, cache_suffix=None):
    """Observer-centred asteroid vectors covering all epochs.

    Cached as ``<cache_dir>/horizons_<designation>_topo[_<cache_suffix>].csv``. Parameters as
    `get_helio`, plus `location`, the MPC observatory code (``X05`` Rubin, ``500`` geocentre),
    and `cache_suffix`, which distinguishes the tables of several stations.

    Returns
    -------
    pandas.DataFrame
        Columns ``jd_tdb``, ``x``, ``y``, ``z`` (ecliptic J2000, au).
    """
    jd_utc = np.asarray(jd_utc, float)
    tt = jd_utc + utc_to_tdb_days(jd_utc)
    name = f"horizons_{designation}_topo" + (f"_{cache_suffix}" if cache_suffix else "") + ".csv"
    path = os.path.join(cache_dir, name) if cache_dir else None
    return _cached_or_query(path, unpack_designation(designation), location, jd_utc, tt,
                            refresh, gap_days, pad_days, step)


OBLIQUITY_J2000_RAD = np.radians(84381.448 / 3600.0)  # IAU 1976, as used by Horizons


def earth_topo_from_helio(helio):
    """Geocentric asteroid vectors without a Horizons observer query.

    The Earth's heliocentric position comes from astropy's built-in ephemeris (ERFA ``epv00``,
    accurate to a few km) on the time grid of the heliocentric table. Used by
    ``observer="local"``.

    Parameters
    ----------
    helio : pandas.DataFrame
        Heliocentric table from `get_helio`.

    Returns
    -------
    pandas.DataFrame
        Geocentric vectors in the same format.
    """
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric
    t = Time(helio["jd_tdb"].values, format="jd", scale="tdb")
    e = get_body_barycentric("earth", t, ephemeris="builtin").xyz.to("au").value.T
    sun = get_body_barycentric("sun", t, ephemeris="builtin").xyz.to("au").value.T
    eq = e - sun                                   # heliocentric Earth, ICRS equatorial
    c, s = np.cos(OBLIQUITY_J2000_RAD), np.sin(OBLIQUITY_J2000_RAD)
    ecl = np.c_[eq[:, 0], c * eq[:, 1] + s * eq[:, 2], -s * eq[:, 1] + c * eq[:, 2]]
    ast = helio[["x", "y", "z"]].values
    return pd.DataFrame({"jd_tdb": helio["jd_tdb"].values, "x": ast[:, 0] - ecl[:, 0],
                         "y": ast[:, 1] - ecl[:, 1], "z": ast[:, 2] - ecl[:, 2]})


def get_vectors(designation, jd_utc, location="X05", cache_dir=None, step="1h", pad_days=0.5,
                refresh=False, gap_days=2.0, cache_suffix=None):
    """Heliocentric and observer-centred tables for a single observer.

    Parameters as `get_topo`.

    Returns
    -------
    helio, topo : pandas.DataFrame
    """
    kw = dict(cache_dir=cache_dir, step=step, pad_days=pad_days, refresh=refresh, gap_days=gap_days)
    return (get_helio(designation, jd_utc, **kw),
            get_topo(designation, jd_utc, location=location, cache_suffix=cache_suffix, **kw))


def geometry(jd_utc, helio, topo, n_iter=3):
    """Light-time corrected viewing and illumination geometry.

    The light time is solved by iteration, so all vectors refer to the asteroid at emission time.

    Parameters
    ----------
    jd_utc : array_like
        Observation epochs [JD UTC].
    helio, topo : pandas.DataFrame
        Vector tables from `get_helio` and `get_topo`.
    n_iter : int, optional
        Light-time iterations.

    Returns
    -------
    dict
        ``t_emit`` emission times [JD, UTC scale: epoch minus light time]; ``e_sun``, ``e_obs``
        unit vectors from the asteroid to the Sun and to the observer (ecliptic J2000, (N, 3));
        ``r``, ``delta`` heliocentric and observer distances [au]; ``alpha`` phase angle [deg].
    """
    hs, ts = SegmentSpline(helio), SegmentSpline(topo)
    t = np.asarray(jd_utc, float) + utc_to_tdb_days(jd_utc)
    obs_helio = hs(t) - ts(t)  # observer heliocentric position at reception
    tau = np.linalg.norm(ts(t), axis=1) / C_AU_PER_DAY
    for _ in range(n_iter):
        ast = hs(t - tau)
        vec = ast - obs_helio
        tau = np.linalg.norm(vec, axis=1) / C_AU_PER_DAY
    r = np.linalg.norm(ast, axis=1)
    delta = np.linalg.norm(vec, axis=1)
    e_sun = -ast / r[:, None]
    e_obs = -vec / delta[:, None]
    alpha = np.degrees(np.arccos(np.clip(np.sum(e_sun * e_obs, axis=1), -1, 1)))
    return dict(t_emit=np.asarray(jd_utc, float) - tau, e_sun=e_sun, e_obs=e_obs,
                r=r, delta=delta, alpha=alpha)


def check_against_file(geo, df):
    """Largest differences to the geometry columns of a plain lightcurve file.

    Parameters
    ----------
    geo : dict
        Output of `geometry`.
    df : pandas.DataFrame
        The lightcurve; columns ``topodist``, ``heliodist`` [au] and ``phase_angle`` [deg] are
        compared when present.

    Returns
    -------
    dict
        ``delta_au``, ``r_au``, ``alpha_deg`` maximum absolute differences (empty if the file has
        no geometry).
    """
    out = {}
    if "topodist" in df:
        out["delta_au"] = float(np.max(np.abs(geo["delta"] - df["topodist"].values)))
    if "heliodist" in df:
        out["r_au"] = float(np.max(np.abs(geo["r"] - df["heliodist"].values)))
    if "phase_angle" in df:
        out["alpha_deg"] = float(np.max(np.abs(geo["alpha"] - df["phase_angle"].values)))
    return out
