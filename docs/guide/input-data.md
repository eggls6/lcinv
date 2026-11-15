# Input data

Every model reads its photometry through `lcinv.setup_data.prepare()`, so the rules on this page
apply to `run_inversion.py`, `run_hybrid.py` and `run_binary.py` alike. Two formats are
accepted, and ADES files are recognised automatically.

## Plain lightcurve files

A CSV file with one row per observation, as produced by Rubin-style pipelines. The bundled
`data/K03Q21N.lc` starts like this:

```text
designation,epoch,band,mag,rmsmag,flag,topodist,heliodist,phase_angle
K03Q21N,2461094.694874841,i,20.449988202618783,0.014340379560914236,0,2.0219916413211574,2.912370099618761,10.20183277130127
K03Q21N,2461094.6958364686,i,20.493843331232434,0.015445883820954498,0,2.021984732101937,2.912370365357962,10.201513290405273
```

| Column | Required | Meaning |
|---|---|---|
| `epoch` | yes | JD (UTC) of the observation; exposure mid-time unless `--obstime start` |
| `band` | yes | filter label; each band gets its own magnitude offset |
| `mag` | yes | apparent magnitude |
| `rmsmag` | yes | 1-sigma uncertainty, mag |
| `flag` | no | rows with `flag` other than 0 are dropped |
| `exptime` | no | exposure time in s, for [exposure smearing](smearing.md) |
| `topodist`, `heliodist`, `phase_angle` | no | if present, compared with the Horizons geometry as a check |
| `designation` | no | informational; pass `--designation` on the command line |

Rows in `--drop-bands` (default `u`) are removed. The designation is required for plain files:
`--designation K03Q21N` (packed MPC form) or a number such as `--designation 91984`. The
observer is `--location` (default `X05`, Rubin).

## ADES files

ADES is the MPC's Astrometry Data Exchange Standard. lcinv reads its CSV, PSV and XML encodings;
column names are matched case-insensitively. An MPC export such as the bundled
`data/566130_ades.csv` works as is, and the designation comes from `permID`/`provID`.

```bash
python run_inversion.py data/566130_ades.csv --stations X05 --outdir results/566130_X05
python run_inversion.py data/566130_ades.csv --exclude-stations 691,J04 --observer local ...
```

**Groups.** Each (station, band) pair is a photometric group with its own magnitude offset,
since stations use different catalogues, apertures and filters. With one station the labels are
plain bands (`g`, `r`); with several they are qualified (`X05:r`, `I41:g`). Groups with fewer than
`--ades-min-group` rows (default 5) are dropped, since their offset would absorb all their
information.

**Uncertainties** are built in three steps:

1. `rmsMag` if given, else `--ades-default-rms` (0.1 mag);
2. plus the rounding error of the reported precision in quadrature, step/\(\sqrt{12}\): a
   magnitude reported to 0.1 mag gets at least 0.029 mag;
3. plus `--ades-sparse-floor` (0.05 mag) in quadrature for groups whose median error exceeds
   `--ades-dense-rms` (0.03 mag). Sparse survey photometry carries calibration errors far above
   its formal `rmsMag`.

Rows above `--ades-max-rms` are dropped. The `exp` column is used as the exposure time.

| Option | Default | Effect |
|---|---|---|
| `--stations` | all | comma-separated MPC codes to keep, e.g. `X05` or `X05,I41,F51` |
| `--exclude-stations` | none | codes to drop |
| `--ades-bands` | all | band codes to keep |
| `--ades-default-rms` | 0.1 | error when `rmsMag` is missing |
| `--ades-sparse-floor` | 0.05 | extra error for sparse groups |
| `--ades-dense-rms` | 0.03 | median error above which a group counts as sparse |
| `--ades-max-rms` | none | drop rows with larger error |
| `--ades-min-group` | 5 | minimum rows per group |

## Geometry and the Horizons cache

The first run of a target queries JPL Horizons (through astroquery) for heliocentric and
observer-centred vectors on an hourly grid covering the observations, in windows around them
for multi-apparition data. Tables are cached in `--cache-dir` (default `data`):

| File | Content |
|---|---|
| `horizons_<designation>_helio.csv` | asteroid relative to the Sun |
| `horizons_<designation>_topo.csv` | asteroid relative to `--location` (plain files) |
| `horizons_<designation>_topo_<stn>.csv` | asteroid relative to one ADES station |

All tables are JD TDB, ecliptic J2000, au. If the cache exists, nothing is queried; delete it to
query again. The bundled caches let both example targets run offline.

**Observer modes** (`--observer`):

`station` (default)
:   one observer table per observatory code (ADES `stn`; plain files use `--location`).

`geocentric`
:   all rows observed from the geocentre (Horizons code 500).

`local`
:   the geocentre from Astropy's built-in Earth ephemeris; only the heliocentric table is
    queried. Fastest for ADES files with many stations.

The geocentric options change the aspect angles by less than 10 arcsec, since the parallax is
below \(4\times10^{-5}\) au. For K03Q21N against Horizons with observer X05 the difference is
4 arcsec, 0.0007° in phase angle.

**Checking the geometry.** When the file has `topodist`, `heliodist` and `phase_angle`, the run
prints the largest differences. For K03Q21N:

```text
334 points, bands ['g', 'i', 'r'], nights 5; geometry vs file: {'delta_au': 1.454731914929397e-07, 'r_au': 2.5232026690247267e-07, 'alpha_deg': 3.6327119943280195e-06}
```

## Outliers

`--clip-sigma` (default 5) drops points more than 5 robust standard deviations from the
multiband Fourier fit at the adopted period, iterating up to five times. Clipped epochs are
listed in the output and in `summary.json` (`clipped_epochs`), and `run_hybrid.py` removes the
same rows, so both layers see identical data. `--clip-sigma 0` turns clipping off; do that for
multi-apparition data, where one Fourier series does not describe all apparitions.

## Using the data from Python

```python
from lcinv.setup_data import prepare

D = prepare("data/K03Q21N.lc", "K03Q21N", location="X05", cache_dir="data")
print(len(D["t"]), D["bands"], D["night"].max() + 1)
```

```text
334 ['g', 'i', 'r'] 5
```

[The `prepare()` dictionary](../reference/api/setup_data.md#the-prepare-dictionary) lists every
key. Example [Period search from Python](../examples/period-search-api.md) continues from here.
