"""Download ADES observations from the Minor Planet Center Observations API.

    https://data.minorplanetcenter.net/api/get-obs   (GET with a JSON body)
    request:  {"desigs": ["2013 DV23"], "output_format": ["ADES_DF"], "ades_version": "2022"}
    response: [ {"ADES_DF": [ {obstime: ..., stn: ..., mag: ..., band: ..., rmsmag: ..., ...}, ... ]} ]

The records are written as CSV (pandas DataFrame.to_csv), the same layout as an ADES CSV
exported from the MPC by hand, so lcinv.ades.read_ades reads them directly. The file holds
all observations of the object (astrometry-only rows included); station/band selection
happens when the file is read (--stations etc.).

Command line:
    python -m lcinv.mpc 2013_DV23 722574 --outdir data/mpc
"""
import argparse
import json
import os
import sys
import threading
import time

import pandas as pd

API_URL = os.environ.get("LCINV_MPC_URL", "https://data.minorplanetcenter.net/api/get-obs")
_lock = threading.Lock()
_last_request = [0.0]


class DownloadError(RuntimeError):
    """Raised when the MPC API cannot be reached or returns an error."""
    pass


def mpc_designation(num):
    """'2013_DV23' -> '2013 DV23'; 722574.0 -> '722574' (the API accepts names, numbers,
    packed and unpacked provisional designations)."""
    s = str(num).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s.replace("_", " ")


def _get(url, payload, timeout):
    body = json.dumps(payload).encode()
    try:
        import requests
        r = requests.get(url, data=body, headers={"Content-Type": "application/json"}, timeout=timeout)
        if r.status_code >= 400:
            raise DownloadError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    except ImportError:
        import urllib.request
        req = urllib.request.Request(url, data=body, method="GET", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())


def query_ades(desig, ades_version="2022", timeout=120, retries=3, min_interval=1.0, url=API_URL):
    """ADES records of one object from the MPC Observations API.

    Parameters
    ----------
    desig : str
        Name, number or designation (see `mpc_designation`).
    ades_version : str, optional
    timeout : float, optional
        Request timeout [s].
    retries : int, optional
    min_interval : float, optional
        Minimum time between requests [s], to be polite to the API.
    url : str, optional

    Returns
    -------
    pandas.DataFrame
        Empty if the MPC has no observations.

    Raises
    ------
    DownloadError
    """
    payload = {"desigs": [mpc_designation(desig)], "output_format": ["ADES_DF"],
               "ades_version": str(ades_version)}
    err = None
    for attempt in range(retries):
        with _lock:                                   # be polite: spaced requests across threads
            wait = _last_request[0] + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.time()
        try:
            j = _get(url, payload, timeout)
            if isinstance(j, dict) and "ADES_DF" not in j:
                raise DownloadError(f"unexpected response: {str(j)[:300]}")
            rec = (j[0] if isinstance(j, list) else j).get("ADES_DF") or []
            return pd.DataFrame(rec)
        except DownloadError as e:
            if "HTTP 4" in str(e) and "HTTP 429" not in str(e):
                raise                                 # bad request / unknown object: no retry
            err = e
        except Exception as e:  # noqa: BLE001 - network errors: retry with backoff
            err = e
        time.sleep(5 * 2 ** attempt)
    raise DownloadError(f"{desig}: download failed after {retries} attempts: {err}")


def fetch_ades(desig, path, refresh=False, require_photometry=True, **kw):
    """Download the ADES observations of an object to a CSV file, unless it exists.

    Parameters
    ----------
    desig : str
    path : str
    refresh : bool, optional
        Download even if `path` exists.
    require_photometry : bool, optional
        Raise if no row has a magnitude.
    **kw
        Passed to `query_ades`.

    Returns
    -------
    path : str
    n_rows : int
    n_photometric : int
    downloaded : bool
    """
    if os.path.exists(path) and not refresh:
        d = pd.read_csv(path, dtype={"mag": str})
        return path, len(d), _n_phot(d), False
    d = query_ades(desig, **kw)
    if d.empty:
        raise DownloadError(f"{desig}: the MPC returned no observations")
    n_phot = _n_phot(d)
    if require_photometry and n_phot == 0:
        raise DownloadError(f"{desig}: {len(d)} observations but none with a magnitude")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    d.to_csv(tmp)
    os.replace(tmp, path)                             # never leave a half-written file
    return path, len(d), n_phot, True


def _n_phot(d):
    cols = {c.lower(): c for c in d.columns}
    if "mag" not in cols:
        return 0
    m = d[cols["mag"]].astype(str).str.strip()
    return int((d[cols["mag"]].notna() & (m != "") & (m.str.lower() != "nan")).sum())


def main(argv=None):
    """Command line: ``python -m lcinv.mpc DESIG [DESIG ...] --outdir data``."""
    p = argparse.ArgumentParser(description="Download ADES observations from the MPC")
    p.add_argument("targets", nargs="+", help="names, numbers or designations (2013_DV23 or '2013 DV23')")
    p.add_argument("--outdir", default="data/mpc",
                   help="folder for the CSV files")
    p.add_argument("--ades-version", default="2022", choices=["2017", "2022"],
                   help="ADES version requested from the MPC")
    p.add_argument("--refresh", action="store_true", help="download again even if the file exists")
    a = p.parse_args(argv)
    bad = 0
    for t in a.targets:
        path = os.path.join(a.outdir, mpc_designation(t).replace(" ", "_") + "_ades.csv")
        try:
            path, n, nph, new = fetch_ades(t, path, refresh=a.refresh, ades_version=a.ades_version)
            print(f"{t}: {n} observations, {nph} with magnitudes -> {path}" + ("" if new else " (cached)"))
        except DownloadError as e:
            bad += 1
            print(f"{t}: {e}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
