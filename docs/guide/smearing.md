# Exposure smearing

A detector integrates the flux over the exposure. For a fast rotator the body turns appreciably
during it, by \(\Delta\varphi = 2\pi T/P\): 17° for a 30 s exposure and a 10.7 min period. Sharp
lightcurve features are flattened, and a model that ignores this underestimates the amplitude and
biases the shape.

## What lcinv does

The model magnitude of an exposure of length \(T\) centred on \(t_i\) is the average flux over it,

\[
m_i = -2.5\log_{10}\Big[\frac1T\int_{t_i-T/2}^{t_i+T/2} L(t)\,dt\Big] + \text{offsets},
\]

evaluated by Gauss-Legendre quadrature with \(Q\) nodes in time; the Sun and observer directions
are held at their mid-exposure values (they move by about \(10^{-5}\) of their daily arc in
30 s). Rotational harmonic \(k\) is attenuated by \(\mathrm{sinc}(k\Delta\varphi/2)\).

- **Convex model:** brightness is linear in the facet areas, so only the scattering matrix is
  averaged; the shape optimisation costs the same.
- **Hybrid model:** only the facet-sum step grows by a factor \(Q\).
- **Period search:** the Fourier design matrix uses the exact sinc attenuation per row, so the
  fitted series and amplitude describe the un-smeared lightcurve.
- **Choosing \(Q\):** `lcinv.smear.choose_nodes` picks the smallest \(Q\) whose quadrature error
  stays below \(10^{-4}\) in relative flux for the highest harmonic the shape model can produce.
  `--smear-nodes` overrides it. Rows with \(T = 0\) recover the instantaneous model exactly.

## Options

| Option | Default | Effect |
|---|---|---|
| `--exptime T` | none | exposure time in s for rows that have none |
| `--obstime` | `mid` | `mid`: epochs are exposure mid-times (ADES convention); `start`: shifted by \(T/2\) |
| `--smear-nodes Q` | automatic | Gauss-Legendre nodes per exposure |

Exposure times come first from the file (ADES `exp`, or an `exptime` column of a plain
lightcurve), then from `--exptime`. Without any, nothing changes. A run with smearing prints a
line such as

```text
exposure smearing: T = 30.0 s (median), rotation swept 16.9 deg, 4 node(s) per exposure
```

This is the 566130 reference run: a 0.178 h rotator turns by 16.9° in a 30 s Rubin exposure, and
four nodes per exposure keep the quadrature error below the tolerance.

## When it matters

The swept angle \(2\pi T/P\) decides. For K03Q21N (\(P = 3.25\) h) a 30 s exposure sweeps 0.9°
and smearing is negligible; for 566130 it is 16.9°, enough to flatten the higher harmonics. The
evidences of the 566130 reference runs (same data, [model comparison](../examples/reference-results.md#566130-model-comparison)):

| Run | ln Z |
|---|---|
| \(c_L = 0.1\), no smearing | 318.63 ± 0.25 |
| \(c_L = 0.1\), 30 s exposures | 314.85 ± 0.25 |
| \(c_L\) fitted, 30 s exposures | 319.63 ± 0.29 |

With the scattering law fixed, accounting for the exposures lowers the evidence; with \(c_L\)
free, the smeared model is the better one. Smearing interacts with the other modelling choices,
so compare complete models rather than switching it off because one comparison favours that.

## Validation

`tests/test_smearing.py` checks the quadrature against the exact attenuation, the model accuracy
against a 64-node reference (below 1 mmag), and that the period search recovers the un-smeared
harmonics.
