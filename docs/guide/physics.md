# Cohesion and size

Lightcurves fix shape and spin but not size. With a size estimate, lcinv can report how much
cohesive strength a rubble pile would need to hold together at the fitted spin, and can use a
cohesion limit as a soft constraint in every model.

## Size from H

Shapes are scaled with the approximate diameter from the absolute magnitude, as in the CNEOS
asteroid size estimator:

\[
D\,[\mathrm{km}] = 10^{\,3.1236 - 0.5\log_{10} p - 0.2H},
\]

with geometric albedo \(p\) (default 0.15) and a range \(D_\min, D_\max\) for \(p = 0.30\) and
\(0.05\). H comes from an explicit value, a table column (`H`, `H_V`, `Hmag`) or the JPL
Small-Body Database (`phys_par` "H", cached in `data/sbdb/`).

```bash
python -m lcinv.size 566130 2013_DV23 --albedo 0.15          # H from SBDB -> D and range
python export_obj.py results/566130_X05_cl --size-from-H      # OBJ in km from the SBDB H
python export_obj.py results/566130_X05_cl --H 18.4 --albedo 0.2
python run_inversion.py ... --size-from-H --cohesion-max 25   # size for the cohesion limit and report
```

## Required cohesion

For a rubble pile with semi-axes \(\alpha \ge \beta \ge \gamma\) spinning about \(\gamma\), the
Drucker-Prager criterion at the centre (Sánchez & Scheeres 2014, MAPS 49, 788, eq. 23) requires

\[
\sigma_Y \ \ge\ \rho\,\alpha^2\,(\omega^2 - \omega_\alpha^2)\,\frac{3 + \sin\phi}{2(3 - \sin\phi)},
\]

with \(\omega_\alpha^2\) the self-gravity term (their eq. 16) and friction angle \(\phi\). The
default \(\phi = 90°\) is their conservative eq. 24. Their estimate for fine-regolith rubble
piles is about 25 Pa.

- `--diameter-km` alone reports the required cohesion of the posterior shapes
  (`summary["cohesion"]`).
- `--cohesion-max S` adds a soft constraint
  \(\ln g = -\tfrac12\big(\max(0, \ln(\sigma_\mathrm{req}/S))/0.1\big)^2\) to the model: convex from
  the projected-area axis ratios, hybrid from the principal moments of the radius field, binary
  from the equivalent ellipsoid of the two lobes. It acts like an observation "the body holds
  together with cohesion at most S", enters every model's evidence the same way, and keeps model
  comparisons fair.

If no shape can satisfy the limit at the fitted period, the evidence drops sharply. That says the
diameter must be smaller or the body is monolithic: 566130 at 0.18 h would need about 400 Pa as a
hypothetical 100 m sphere.

```bash
python run_inversion.py data/X.csv ... --diameter-km 0.35 --density 1700 --cohesion-max 25
python run_hybrid.py    data/X.csv ... --diameter-km 0.35 --cohesion-max 25
python run_binary.py --global-dir ... --diameter-km 0.35 --cohesion-max 25
python batch_shapes.py targets.csv --cohesion-max 25     # uses the size estimate per target
```

| Option | Default | Effect |
|---|---|---|
| `--diameter-km` | none | volume-equivalent diameter; enables the cohesion report |
| `--H`, `--albedo` | none, 0.15 | diameter from H when `--diameter-km` is not given |
| `--size-from-H` | off | look up H in the SBDB |
| `--density` | 1700 kg/m³ | bulk density |
| `--friction-angle` | 90° | internal friction angle |
| `--cohesion-max` | none | soft limit in Pa (needs a diameter) |
| `--cohesion-softness` | 0.1 | width of the soft limit in \(\ln(\sigma_\mathrm{req}/S)\) |
