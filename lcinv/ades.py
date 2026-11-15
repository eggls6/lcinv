"""Reading photometry from MPC ADES files (Astrometry Data Exchange Standard).

Supported encodings: CSV (as exported by the MPC or astropy tables), PSV (pipe-separated, with
``#`` header lines) and XML (``<optical>`` records). Column names are matched case-insensitively
(``obsTime``/``obstime``, ``rmsMag``/``rmsmag``, ``permID``/``permid``, ...). Rows without a
magnitude and deprecated rows are dropped.

Each (station, band) pair is its own photometric group and gets its own magnitude offset in the
models: stations use different reference catalogues, apertures and filter systems. With one
station the band codes are used as labels (``Lr``), with several the group (``X05:Lr``).

Uncertainties are built in three steps:

1. ``rmsMag`` when given, otherwise `default_rms` (``--ades-default-rms``, 0.10 mag);
2. plus, in quadrature, the rounding error of the reported precision, ``step / sqrt(12)``,
   so a magnitude reported to 0.1 mag gets at least 0.029 mag;
3. plus `sparse_floor` (``--ades-sparse-floor``, 0.05 mag) in quadrature for groups whose
   median uncertainty exceeds `dense_rms` (``--ades-dense-rms``, 0.03 mag): sparse survey
   photometry whose calibration errors are far above its formal ``rmsMag``.

Groups with fewer than `min_group` rows (``--ades-min-group``, 5) are dropped, because their
free offset would absorb them entirely. The result has the columns of the plain lightcurve
format plus ``stn``, ``band_raw``, ``mag_decimals``, ``group_size`` and ``exptime`` (from
ADES ``exp``).
"""
import os
import re

import numpy as np
import pandas as pd

_ALIASES = {"obstime": "obstime", "rmsmag": "rmsmag", "permid": "permid", "provid": "provid",
            "trksub": "trksub", "stn": "stn", "band": "band", "mag": "mag", "photcat": "photcat",
            "notes": "notes", "mode": "mode", "deprecated": "deprecated"}


def is_ades(path):
    """True if a file looks like ADES: XML root, PSV header, or CSV with ``obsTime`` and ``stn``."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, errors="replace") as f:
        head = f.read(4096)
    if ext == ".xml" or "<ades" in head.lower():
        return True
    if head.lstrip().startswith("# version") or ("|" in head and "obstime" in head.lower()):
        return True
    first = head.splitlines()[0].lower() if head else ""
    return "obstime" in first and "stn" in first


def _read_table(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, errors="replace") as f:
        head = f.read(4096)
    if ext == ".xml" or "<ades" in head.lower():
        return pd.read_xml(path, xpath=".//optical")
    lines = [l for l in open(path, errors="replace") if l.strip() and not l.startswith("#")]
    if "|" in lines[0]:
        from io import StringIO
        df = pd.read_csv(StringIO("".join(lines)), sep="|", skipinitialspace=True, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        return df.apply(lambda s: s.str.strip() if s.dtype == object else s)
    return pd.read_csv(path, dtype={"mag": str})


def _decimals(s):
    s = str(s).strip()
    return len(s.split(".")[1]) if "." in s else 0


def read_ades(path, stations=None, exclude_stations=None, bands=None, max_rms=None,
              default_rms=0.10, sparse_floor=0.05, dense_rms=0.03, min_group=5,
              designation=None, drop_bands=("u", "Lu")):
    """Read an ADES file into a lightcurve DataFrame.

    Parameters
    ----------
    path : str
        ADES file (CSV, PSV or XML).
    stations, exclude_stations : list of str, optional
        MPC codes to keep or to drop (None: all).
    bands : list of str, optional
        ADES band codes to keep (None: all).
    max_rms : float, optional
        Drop rows whose final uncertainty exceeds this [mag].
    default_rms, sparse_floor, dense_rms : float, optional
        Uncertainty rules, see the module docstring.
    min_group : int, optional
        Drop (station, band) groups with fewer rows.
    designation : str, optional
        Designation to record; default the first ``permID``, ``provID`` or ``trkSub``.
    drop_bands : sequence of str, optional
        Band codes to drop.

    Returns
    -------
    pandas.DataFrame
        Columns ``designation``, ``epoch`` (JD UTC), ``band``, ``mag``, ``rmsmag``, ``flag``,
        ``stn``, ``band_raw``, ``mag_decimals``, ``group_size``, ``exptime``; sorted by epoch.

    Raises
    ------
    ValueError
        If a required column is missing or no data are left after the selection.
    """
    raw = _read_table(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    d = raw.rename(columns={c: _ALIASES.get(c, c) for c in raw.columns})
    for col in ("obstime", "stn", "mag"):
        if col not in d.columns:
            raise ValueError(f"{path}: ADES column '{col}' not found")
    d = d[d["mag"].notna() & (d["mag"].astype(str).str.strip() != "")].copy()
    if "deprecated" in d.columns:
        d = d[d["deprecated"].isna() | (d["deprecated"].astype(str).str.strip() == "")]
    d["mag_decimals"] = d["mag"].map(_decimals)
    d["mag"] = d["mag"].astype(float)
    d["band_raw"] = d.get("band", pd.Series("", index=d.index)).fillna("").astype(str).str.strip()
    d.loc[d["band_raw"] == "", "band_raw"] = "?"
    d["stn"] = d["stn"].astype(str).str.strip()
    if stations:
        d = d[d["stn"].isin(stations)]
    if exclude_stations:
        d = d[~d["stn"].isin(exclude_stations)]
    if bands:
        d = d[d["band_raw"].isin(bands)]
    if drop_bands:
        d = d[~d["band_raw"].isin(drop_bands)]
    if d.empty:
        raise ValueError(f"{path}: no photometry left after station/band selection")

    t = pd.to_datetime(d["obstime"].astype(str).str.strip(), utc=True)
    d["epoch"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().values / 86400.0 + 2440587.5

    rms = pd.to_numeric(d["rmsmag"], errors="coerce") if "rmsmag" in d.columns else pd.Series(np.nan, index=d.index)
    rms = rms.fillna(default_rms).values
    rounding = 10.0 ** (-d["mag_decimals"].values) / np.sqrt(12.0)
    err = np.hypot(rms, rounding)
    d["group"] = d["stn"] + ":" + d["band_raw"]
    med = pd.Series(err, index=d.index).groupby(d["group"]).transform("median").values
    err = np.where(med > dense_rms, np.hypot(err, sparse_floor), err)
    d["rmsmag"] = err
    if max_rms is not None:
        d = d[d["rmsmag"] <= max_rms]
    n = d["group"].map(d["group"].value_counts())
    d = d[n >= min_group].copy()
    if d.empty:
        raise ValueError(f"{path}: no (station, band) group has >= {min_group} usable rows")

    if designation is None:
        for col in ("permid", "provid", "trksub"):
            if col in d.columns and d[col].notna().any():
                designation = str(d[col].dropna().astype(str).iloc[0]).strip()
                break
    # one station: plain band labels; several: station-qualified labels
    d["band"] = d["band_raw"] if d["stn"].nunique() == 1 else d["group"]
    d["designation"] = designation
    d["flag"] = 0
    d["exptime"] = pd.to_numeric(d["exp"], errors="coerce") if "exp" in d.columns else np.nan
    d["group_size"] = d["group"].map(d["group"].value_counts())
    cols = ["designation", "epoch", "band", "mag", "rmsmag", "flag", "stn", "band_raw",
            "mag_decimals", "group_size", "exptime"]
    return d[cols].sort_values("epoch").reset_index(drop=True)


def summary(df):
    """Per-group overview of an ADES DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `read_ades`.

    Returns
    -------
    pandas.DataFrame
        Indexed by (``stn``, ``band_raw``): rows ``n``, ``nights``, ``first_jd``, ``last_jd``,
        median uncertainty ``med_err`` and reported precision ``decimals``.
    """
    nights = np.floor(df["epoch"] - 0.5).astype(int)
    g = df.assign(night=nights).groupby(["stn", "band_raw"])
    out = g.agg(n=("mag", "size"), nights=("night", "nunique"),
                first_jd=("epoch", "min"), last_jd=("epoch", "max"),
                med_err=("rmsmag", "median"), decimals=("mag_decimals", "max"))
    return out


def parse_list(s):
    """Split a comma- or space-separated string: ``'X05,I41'`` -> ``['X05', 'I41']``; empty or None
    -> None.
    """
    if not s:
        return None
    return [x for x in re.split(r"[,\s]+", s.strip()) if x]


# ------------------------------------------------------------------------------- CLI helpers
def add_cli_arguments(parser):
    """Add the ADES-input and exposure-smearing options to an argument parser.

    Adds the groups "exposure smearing" (``--exptime``, ``--obstime``, ``--smear-nodes``) and
    "ADES input" (``--stations``, ``--exclude-stations``, ``--ades-bands``, the uncertainty
    rules, ``--ades-min-group``, ``--observer``). Shared by all model scripts.

    Parameters
    ----------
    parser : argparse.ArgumentParser

    Returns
    -------
    argparse.ArgumentParser
    """
    add_timing_arguments(parser)
    g = parser.add_argument_group("ADES input (detected automatically)")
    g.add_argument("--stations", default=None,
                   help="comma-separated MPC codes to use, e.g. X05 or X05,I41,F51 (default all)")
    g.add_argument("--exclude-stations", default=None, help="comma-separated MPC codes to drop")
    g.add_argument("--ades-bands", default=None, help="comma-separated ADES band codes to keep")
    g.add_argument("--ades-default-rms", type=float, default=0.10,
                   help="uncertainty when rmsMag is missing [mag]")
    g.add_argument("--ades-sparse-floor", type=float, default=0.05,
                   help="extra error (quadrature) for groups with median error > --ades-dense-rms")
    g.add_argument("--ades-dense-rms", type=float, default=0.03,
                   help="groups whose median error exceeds this are treated as sparse and get "
                        "--ades-sparse-floor [mag]")
    g.add_argument("--ades-max-rms", type=float, default=None, help="drop rows with larger error")
    g.add_argument("--ades-min-group", type=int, default=5,
                   help="drop (station, band) groups with fewer rows")
    g.add_argument("--observer", choices=["station", "geocentric", "local"], default="station",
                   help="observer: Horizons table per station; Horizons geocentre; or geocentre "
                        "from astropy's built-in Earth ephemeris (only heliocentric vectors queried)")
    return parser


def add_timing_arguments(parser):
    """Add the exposure-smearing options (``--exptime``, ``--obstime``, ``--smear-nodes``).

    Parameters
    ----------
    parser : argparse.ArgumentParser

    Returns
    -------
    argparse.ArgumentParser
    """
    g = parser.add_argument_group("exposure smearing")
    g.add_argument("--exptime", type=float, default=None,
                   help="exposure time [s] for rows without one (ADES 'exp' or a lightcurve "
                        "'exptime' column take precedence); default: no smearing")
    g.add_argument("--obstime", choices=["mid", "start"], default="mid",
                   help="whether epochs are exposure mid-times (ADES convention) or start times")
    g.add_argument("--smear-nodes", type=int, default=None,
                   help="Gauss-Legendre nodes per exposure (default: chosen from P, T and lmax)")
    return parser


def timing_from_args(a):
    """``exptime`` and ``obstime`` keyword arguments for `lcinv.setup_data.prepare`.

    Parameters
    ----------
    a : argparse.Namespace or dict
        Parsed options, or the ``args`` stored in a results folder.

    Returns
    -------
    dict
    """
    get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    return dict(exptime=get("exptime"), obstime=get("obstime", "mid"))


def opts_from_args(a):
    """``ades_opts`` for `lcinv.setup_data.prepare` from parsed options.

    Parameters
    ----------
    a : argparse.Namespace or dict
        Parsed options, or the ``args`` stored in a results folder.

    Returns
    -------
    dict
        Keyword arguments of `read_ades`.
    """
    get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    return dict(stations=parse_list(get("stations")), exclude_stations=parse_list(get("exclude_stations")),
                bands=parse_list(get("ades_bands")), default_rms=get("ades_default_rms", 0.10),
                sparse_floor=get("ades_sparse_floor", 0.05), dense_rms=get("ades_dense_rms", 0.03),
                max_rms=get("ades_max_rms"), min_group=get("ades_min_group", 5))


def designation_from_file(path):
    """The ``permID`` / ``provID`` of an ADES file, or None for other formats."""
    if not is_ades(path):
        return None
    return read_ades(path, min_group=1).loc[0, "designation"]
