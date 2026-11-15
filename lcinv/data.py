"""Reading lightcurve files.

Two formats are accepted and detected automatically:

* a plain CSV lightcurve (the Rubin-style ``.lc`` files) with the columns ``designation``,
  ``epoch`` (JD UTC, exposure mid-time), ``band``, ``mag``, ``rmsmag`` and ``flag``, optionally
  ``exptime`` [s] and the geometry columns ``heliodist``, ``topodist``, ``phase_angle`` used for a
  consistency check;
* an MPC ADES file (CSV, PSV or XML), read by `lcinv.ades`.
"""
import numpy as np
import pandas as pd


def load_lightcurve(path, drop_bands=("u",), max_err=None, flag_ok=0, ades_opts=None):
    """Read a lightcurve file into a DataFrame sorted by epoch.

    Parameters
    ----------
    path : str
        Plain CSV lightcurve or ADES file (see the module docstring).
    drop_bands : sequence of str, optional
        Bands to remove. For ADES files the ``L``-prefixed Rubin codes are dropped too
        (``u`` also removes ``Lu``).
    max_err : float, optional
        Drop rows with a larger uncertainty [mag].
    flag_ok : int or None, optional
        Keep only rows with this ``flag`` (plain files); None keeps all.
    ades_opts : dict, optional
        Keyword arguments for `lcinv.ades.read_ades` (station and band selection, uncertainty
        rules); see `lcinv.ades.opts_from_args`.

    Returns
    -------
    pandas.DataFrame
        At least ``designation``, ``epoch``, ``band``, ``mag``, ``rmsmag``, ``flag``; ADES input
        adds ``stn`` (observatory code per row), ``band_raw``, ``mag_decimals``, ``group_size``
        and ``exptime``.
    """
    from . import ades
    if ades.is_ades(path):
        opts = dict(ades_opts or {})
        opts.setdefault("drop_bands", tuple(drop_bands or ()) + tuple("L" + b for b in (drop_bands or ())))
        if max_err is not None:
            opts.setdefault("max_rms", max_err)
        return ades.read_ades(path, **opts)
    d = pd.read_csv(path)
    if "flag" in d.columns and flag_ok is not None:
        d = d[d["flag"] == flag_ok]
    if drop_bands:
        d = d[~d["band"].isin(drop_bands)]
    if max_err is not None:
        d = d[d["rmsmag"] <= max_err]
    d = d.sort_values("epoch").reset_index(drop=True)
    return d


def band_index(bands):
    """Map band labels to integer indices.

    Parameters
    ----------
    bands : array_like of str
        Band label of each row.

    Returns
    -------
    idx : ndarray of int
        Index of each row's band in `labels`.
    labels : list of str
        Sorted unique labels.
    """
    labels = sorted(np.unique(bands))
    lut = {b: i for i, b in enumerate(labels)}
    return np.array([lut[b] for b in bands], dtype=int), labels


def night_index(jd, split_hour_utc=12.0):
    """Integer label of the observing night of each epoch.

    Parameters
    ----------
    jd : ndarray
        Epochs [JD UTC].
    split_hour_utc : float, optional
        Hour (UTC) at which one night ends and the next begins; 12 h suits observatories in the
        Americas, such as Rubin in Chile.

    Returns
    -------
    ndarray of int
        Night labels 0, 1, ... in time order.
    """
    shifted = jd + 0.5 - split_hour_utc / 24.0  # days since 00 UT, shifted to the split hour
    n = np.floor(shifted).astype(int)
    _, idx = np.unique(n, return_inverse=True)
    return idx
