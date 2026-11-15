## The `prepare()` dictionary

`lcinv.setup_data.prepare()` returns a plain dictionary. Every per-row array has one entry per
photometric point, sorted by epoch after filtering; `subset()` and `drop_epochs()` keep all of them
consistent.

| Key | Shape | Content |
|---|---|---|
| `df` | DataFrame | the filtered input rows (`epoch`, `band`, `mag`, `rmsmag`, `exptime`, and `stn` for ADES) |
| `t` | (N,) | emission times, JD (UTC epoch minus light time); the rotation phase is computed from these |
| `mag` | (N,) | reduced magnitudes, \(m - 5\log_{10}(r\Delta)\) |
| `err` | (N,) | quoted uncertainties in mag (ADES rules applied, see [Input data](../../guide/input-data.md)) |
| `exptime` | (N,) | exposure times in s (0 = instantaneous) |
| `band_idx`, `bands` | (N,), list | integer group index per row and the sorted group labels (`g`; for ADES the band code, `Lr`, or `X05:Lr` when several stations are used) |
| `alpha` | (N,) | solar phase angle, deg |
| `e_sun`, `e_obs` | (N, 3) | ecliptic J2000 unit vectors from the asteroid to the Sun and to the observer |
| `night` | (N,) | integer night label (nights split at 12 h UTC) |
| `geo` | dict | the full geometry: `t_emit`, `e_sun`, `e_obs`, `r`, `delta`, `alpha` |
| `check` | dict | largest differences to `topodist`, `heliodist`, `phase_angle` columns of the file, if present |
| `helio`, `topo` | DataFrame / dict | the Horizons vector tables (one `topo` table per station for ADES) |
| `stations` | list | MPC observatory codes used (empty for plain files and `observer="local"`) |
