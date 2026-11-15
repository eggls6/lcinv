# How lcinv works

A lightcurve inversion asks which shape, spin axis and rotation period reproduce the brightness
variations observed from changing viewing geometries. lcinv answers it in two layers, each used
for what it does well, and reports uncertainties for all of these quantities rather than a single
best fit.

| | Global layer (`run_inversion.py`) | Local layer (`run_hybrid.py`) |
|---|---|---|
| Purpose | find all spin solutions and aliases, and their probabilities | joint spin and shape posterior within one spin solution |
| Shape | convex Gaussian image, marginalised by a Laplace approximation | star-shaped log-radius field, sampled |
| Sampler | dynesty nested sampling over \((\lambda, \beta, P, \ln\sigma_\mathrm{jit})\) | NumPyro NUTS over spin, shape and hyperparameters |
| Strength | multimodality, evidence, period aliases | non-Gaussian shape uncertainty, physical priors |
| Weakness | approximate shape marginal; no inertia constraint | cannot move between solutions; no evidence of its own |

The distinct pole solutions are called **spin states** (spin state 1, 2, ...). The global layer
finds them; the local layer samples each one; their weights come from the local model's Laplace
evidence.

## Geometry

For every observation lcinv needs the directions from the asteroid to the Sun and to the
observer, at the time the light left the asteroid. `lcinv.ephem` queries heliocentric and
observer-centred state vectors from JPL Horizons on an hourly TDB grid, splines them, applies the
leap-second offset per epoch and iterates the light time three times. Epochs in the input are
taken as UTC.

Observers can be one Horizons table per observatory code (`--observer station`, the default), the
geocentre for everyone (`geocentric`), or the geocentre computed locally from Astropy's built-in
Earth ephemeris (`local`, only the asteroid's heliocentric vectors are queried). For the bundled
Rubin data the geocentric options change aspect angles by less than 10 arcsec.

## Magnitudes and nuisance parameters

The models work with reduced magnitudes \(m - 5\log_{10}(r\Delta)\). Each photometric group (a
band, or a station and band for ADES) gets its own magnitude offset, and a linear phase-angle
term absorbs the phase function. These are linear parameters: the global layer profiles them by
weighted least squares inside every likelihood call; the local layer marginalises them
analytically.

Survey photometry has errors beyond the quoted ones (calibration, star contamination,
model imperfection). Both layers fit a **jitter** \(\sigma_\mathrm{jit}\), white noise added in
quadrature to the quoted errors, with a log-uniform prior from 1 to 100 mmag.

## The global layer

**Shape.** A convex body is described by its Gaussian image: 400 fixed facet normals on a
Fibonacci sphere with areas

\[
A_i = w_i \exp\Big(\sum_{l=1}^{l_\max}\sum_m c_{lm} Y_{lm}(\hat n_i)\Big),
\]

positive by construction, with \(l_\max = 4\) (24 coefficients) by default. A penalty enforces
closure, \(\sum_i A_i \hat n_i = 0\). Brightness is the sum over lit and visible facets of the
scattering law \(S = \mu\mu_0\,[1/(\mu + \mu_0) + c_L]\), Lommel-Seeliger plus a Lambert term.
For a convex body this sum is exact: there is no self-shadowing.

**Laplace marginalisation.** For each spin state \(\theta\) the shape coefficients are optimised
with a jitted L-BFGS (coarse to fine: three starts (`--n-starts`) using only \(l \le 2\), then all terms) and
integrated out:

\[
\ln L(\theta) = -f(\hat c) + \tfrac{k}{2}\ln 2\pi - \tfrac12 \ln\det H(\hat c),
\]

where \(f\) is the negative log posterior of the coefficients under a Gaussian prior with scale
\(\tau_l = \tau_0 / l\). Nested sampling then explores only the 4 spin parameters (5 or 6 with
\(c_L\) or \(\tau_0\) fitted), which keeps it fast enough to map every mode and compute the
evidence.

**The rotational phase is a gauge.** Rotating a free shape about its spin axis by \(\delta\)
gives the same lightcurve as shifting the initial phase \(\varphi_0\) by \(\delta\), and the shape
prior is rotation invariant. lcinv therefore fixes \(\varphi_0 = 0\) at a reference time
\(t_\mathrm{ref}\) (the data mid-time) and lets the shape orientation absorb it. Sampling
\(\varphi_0\) would only add a flat direction.

**Shape uncertainty.** Each posterior spin draw gets one draw of the coefficients from the
Laplace Gaussian \(\mathcal N(\hat c, H^{-1})\). The facet areas are turned into a polyhedron by
solving the Minkowski problem (`lcinv.minkowski`). These are the shapes in `shapes/*.obj` and the
3D viewer.

## The local layer

Within each spin state the hybrid model samples spin and shape jointly with NUTS:

- **Shape** is a log-radius field \(r(\hat u) = \exp(\sum_{l \ge 2} a_{lm} Y_{lm})\) with a
  GP-like, non-centred prior; \(\sigma_f\) (pointwise spread of \(\ln r\)) and \(\ell\) (angular
  scale) are hyperparameters with log-normal priors. It is star-shaped, not necessarily convex.
- **Physics:** volume, centre of mass and inertia tensor are explicit integrals, and a soft prior
  asks for rotation about the axis of maximum inertia. The global convex shapes do not know
  about inertia.
- **Spin coordinates** are a gnomonic chart around the spin state's centre, free of the
  coordinate singularity at the ecliptic poles. The body frame is built by parallel transport
  (quaternions), which removes a near-flat ridge between the pole longitude and the shape
  orientation that the Euler-angle frame created.
- **Checks** after sampling: convexity of the posterior shapes, ray-traced self-shadowing error,
  R-hat, ESS and divergences.

[Hybrid model](../guide/hybrid.md) has the details and options.

## What the data can and cannot say

- **Mirror solutions.** Photometry from one apparition constrains the pole only up to the
  \((\lambda, \beta) \leftrightarrow (\lambda + 180°, -\beta)\) ambiguity; lcinv reports both
  spin states with their weights instead of picking one.
- **Period aliases.** Half and double periods can fit the Fourier series equally well. The
  period search lists the competing minima; the period prior and physical arguments (spin
  barrier, amplitude) decide.
- **Axis ratios.** The amplitude gives a lower bound on \(a/b\); the projected-area ratios
  reported by the global layer are exact only for ellipsoids, so `summary.json` also lists the
  extent ratios of the reconstructed polyhedra.
- **Multiple apparitions** constrain the pole far better but turn the period into a
  rotation-counting problem; see [Period search](../guide/period-search.md#data-spanning-several-apparitions).
