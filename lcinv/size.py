"""Approximate diameter from the absolute magnitude H, as in the CNEOS asteroid size estimator
(https://cneos.jpl.nasa.gov/tools/ast_size_est.html):

    D [km] = 10^(3.1236 - 0.5 log10(p) - 0.2 H)          (= 1329 km / sqrt(p) * 10^(-H/5))

with p the geometric albedo. The CNEOS reference table spans p = 0.05 ... 0.30; the nominal
albedo here is 0.15 (their S-type example), and D_min / D_max are given for 0.30 / 0.05.

H comes from (in this order) an explicit value, a table column (H, H_V, Hmag), or the JPL
Small-Body Database API (https://ssd-api.jpl.nasa.gov/sbdb.api, phys_par "H"), cached as JSON.

    python -m lcinv.size 566130 2013_DV23 --albedo 0.15
"""
import argparse
import json
import os
import sys

import numpy as np

SBDB_URL = os.environ.get("LCINV_SBDB_URL", "https://ssd-api.jpl.nasa.gov/sbdb.api")
ALBEDO_NOMINAL = 0.15
ALBEDO_RANGE = (0.05, 0.30)
H_COLUMNS = ("H", "H_V", "Hmag", "h", "H_mag", "absmag")


def diameter_km(H, albedo=ALBEDO_NOMINAL):
    """Diameter from H and geometric albedo (CNEOS): ``D = 10^(3.1236 - 0.5 log10(p) - 0.2 H)`` km.

    Parameters
    ----------
    H : float or array_like
        Absolute magnitude.
    albedo : float, optional
        Geometric albedo p.

    Returns
    -------
    float or ndarray
        Diameter [km].
    """
    return 10 ** (3.1236 - 0.5 * np.log10(albedo) - 0.2 * np.asarray(H, float))


def size_estimate(H, albedo=ALBEDO_NOMINAL, albedo_range=ALBEDO_RANGE):
    """Nominal diameter and its range over the CNEOS albedo range.

    Parameters
    ----------
    H : float
        Absolute magnitude.
    albedo : float, optional
        Nominal geometric albedo.
    albedo_range : (float, float), optional
        Albedo range for ``diameter_min_km`` (highest albedo) and ``diameter_max_km`` (lowest).

    Returns
    -------
    dict
        ``H``, ``albedo``, ``diameter_km``, ``diameter_min_km``, ``diameter_max_km``,
        ``albedo_range``, ``method``.

    Examples
    --------
    >>> round(size_estimate(16.0)["diameter_km"], 3)
    2.165
    """
    H = float(H)
    return dict(H=H, albedo=float(albedo), diameter_km=float(diameter_km(H, albedo)),
                diameter_min_km=float(diameter_km(H, max(albedo_range))),
                diameter_max_km=float(diameter_km(H, min(albedo_range))),
                albedo_range=list(albedo_range),
                method="CNEOS size estimator: D = 10^(3.1236 - 0.5 log10 p - 0.2 H) km")


def _designation(num):
    from lcinv.mpc import mpc_designation
    return mpc_designation(num)


def query_H(desig, cache_dir="data/sbdb", refresh=False, timeout=60):
    """Absolute magnitude H from the JPL Small-Body Database API.

    Answers are cached as JSON in `cache_dir`. The API URL can be changed with the environment
    variable ``LCINV_SBDB_URL``.

    Parameters
    ----------
    desig : str
        Number or designation.
    cache_dir : str, optional
    refresh : bool, optional
        Ignore the cache.
    timeout : float, optional
        Request timeout [s].

    Returns
    -------
    H : float or None
        None if the object has no H.
    name : str
        Full name reported by the SBDB.
    """
    d = _designation(desig)
    path = os.path.join(cache_dir, d.replace(" ", "_") + ".json") if cache_dir else None
    if path and os.path.exists(path) and not refresh:
        j = json.load(open(path))
    else:
        import urllib.parse
        url = SBDB_URL + "?" + urllib.parse.urlencode({"sstr": d, "phys-par": 1})
        try:
            import requests
            r = requests.get(url, timeout=timeout)
            j = r.json()
        except ImportError:
            import urllib.request
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                j = json.loads(resp.read().decode())
        if str(j.get("code", "")) == "300":
            raise LookupError(f"{d}: SBDB matched several objects: "
                              + ", ".join(f"{x.get('pdes')} ({x.get('name')})" for x in j.get("list", [])[:5]))
        if "object" not in j:
            raise LookupError(f"{d}: SBDB: {j.get('message', 'object not found')}")
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(j, f)
    for p in j.get("phys_par", []) or []:
        if p.get("name") == "H" and p.get("value") not in (None, ""):
            return float(p["value"]), j["object"].get("fullname", d)
    return None, j.get("object", {}).get("fullname", d)


def H_from_row(row):
    """First finite H-like column (``H``, ``H_V``, ``Hmag``, ...) of a table row.

    Returns
    -------
    H : float or None
    column : str or None
    """
    for c in H_COLUMNS:
        if c in row.index:
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                return v, c
    return None, None


def estimate_for(num, row=None, H=None, albedo=ALBEDO_NOMINAL, albedo_column=None, lookup=True,
                 cache_dir="data/sbdb"):
    """Size estimate for one target.

    H is taken from, in this order: `H`, an H-like column of `row`, the SBDB (if `lookup`). An
    albedo column of the row, if named and valid, overrides `albedo`.

    Parameters
    ----------
    num : str
        Number or designation (for the SBDB lookup).
    row : pandas.Series, optional
        Target-table row.
    H : float, optional
    albedo : float, optional
    albedo_column : str, optional
    lookup : bool, optional
        Allow the SBDB query.
    cache_dir : str, optional

    Returns
    -------
    dict
        As `size_estimate`, plus ``H_source``.

    Raises
    ------
    LookupError
        If no H is available.
    """
    src = "given"
    if H is None and row is not None:
        H, col = H_from_row(row)
        src = f"table column {col}" if H is not None else src
    if H is None and lookup:
        H, name = query_H(num, cache_dir=cache_dir)
        src = f"JPL SBDB ({name})"
    if H is None:
        raise LookupError(f"{num}: no absolute magnitude H available")
    if albedo_column and row is not None and albedo_column in row.index:
        try:
            a = float(row[albedo_column])
            if np.isfinite(a) and a > 0:
                albedo = a
        except (TypeError, ValueError):
            pass
    out = size_estimate(H, albedo)
    out["H_source"] = src
    return out


def main(argv=None):
    """Command line: ``python -m lcinv.size TARGET [TARGET ...] [--H H] [--albedo P]``."""
    p = argparse.ArgumentParser(description="Diameter from H (CNEOS size estimator), H from JPL SBDB")
    p.add_argument("targets", nargs="+",
                   help="numbers or designations")
    p.add_argument("--H", type=float, default=None, help="use this H instead of looking it up")
    p.add_argument("--albedo", type=float, default=ALBEDO_NOMINAL,
                   help="geometric albedo (CNEOS reference range 0.05-0.30)")
    p.add_argument("--cache-dir", default="data/sbdb",
                   help="cache of SBDB answers")
    a = p.parse_args(argv)
    for t in a.targets:
        try:
            e = estimate_for(t, H=a.H, albedo=a.albedo, cache_dir=a.cache_dir)
            print(f"{t}: H = {e['H']:.2f} ({e['H_source']}), p = {e['albedo']:g} -> D = {e['diameter_km']:.3f} km "
                  f"(range {e['diameter_min_km']:.3f}-{e['diameter_max_km']:.3f} km for p = "
                  f"{ALBEDO_RANGE[1]:g}-{ALBEDO_RANGE[0]:g})")
        except Exception as e:  # noqa: BLE001
            print(f"{t}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
