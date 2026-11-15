# Validation

How we know the numbers are right. The package ships seven test scripts that check the physics
against independent calculations, plus diagnostic scripts used while developing the method. All
of them pass with lcinv 0.2.0 on CPU; the output below is copied from those runs.

## Running the tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests -q
```

or, to see the printed comparisons below, run the scripts one by one:

```bash
export PYTHONPATH=. JAX_PLATFORMS=cpu
for t in test_period test_cohesion_fitstats test_quaternion_frame test_hybrid_basics \
         test_smearing test_cl test_binary; do
    python tests/$t.py || echo "FAILED: $t"
done
```

| Test | What it checks | Time | Result |
|---|---|---|---|
| `test_period` | period search, refinement, amplitude, a/b bound, search-range warnings | 10 s | passed |
| `test_cohesion_fitstats` | cohesion formulas, effective parameters, spin-state naming, size from H | 1 s | passed |
| `test_quaternion_frame` | the parallel-transport body frame of the hybrid layer | 3 s | passed |
| `test_hybrid_basics` | quadrature, orthonormality, mesh and prior of the radius field | 3 s | passed |
| `test_smearing` | exposure smearing in the period search, convex and hybrid models | 16 s | passed |
| `test_cl` | the fitted Lambert weight \(c_L\) | 32 s | passed |
| `test_binary` | the contact-binary model against a z-buffer renderer | 32 s | passed |

Each script exits with an error on the first failed assertion, so `|| echo` reports exactly the
failing ones. They need no network: all data and geometry are in `data/` and `results/`. The
same tests run on every push through `.github/workflows/tests.yml`.

## Period search

`test_period` builds a double-peaked lightcurve of a 0.5 h rotator in two bands over five
nights and searches it twice:

```text
--8<-- "examples/docs/test_period.out"
```

With a range that contains the period, it is recovered to \(2\times10^{-6}\) of its value, the
interval contains the truth and no warning is raised. With the default range (\(P \ge 2\) h)
the search returns a meaningless 20 h, and both checks of
[`period.range_warnings`](reference/api/period.md) fire: a deeper periodogram minimum below
`--pmin` and a poor Fourier fit. For an equator-on ellipsoid of \(a/b = 1.5\) the amplitude bound
recovers 1.5.

## Cohesion, goodness of fit and sizes

`test_cohesion_fitstats` compares `lcinv.cohesion` with the closed-form cases of Sánchez &
Scheeres (2014): the critical spin rate of a sphere \(\omega^2 = \tfrac43\pi G\rho\), zero
cohesion below it, the stress at \(\varphi = 90°\) and its halving at \(\varphi = 0°\), and
Carlson's elliptic integral \(R_D\) of the JAX implementation against SciPy to \(10^{-12}\). It
then checks the effective-parameter count of [`lcinv.fitstats`](reference/api/fitstats.md) on
linear models where it is known exactly, the renaming of legacy hybrid results, and the CNEOS size
relation.

```text
cohesion ok
fitstats ok
naming ok
size ok
```

## The hybrid body frame

The hybrid layer samples the pole in a gnomonic chart around each spin-state centre. The body
frame must follow the pole without twisting about the spin axis, or pole and rotational phase
become entangled. `test_quaternion_frame` measures that twist for a small pole step (0.5°):

```text
geometry: proper rotations, pole -> +z, and Euler/quaternion frames differ by a z-rotation  OK
twist about the spin axis per unit pole displacement (eps = 0.5 deg):
  beta =  0 deg:  Euler    0.002   quaternion  0.00000
  beta = 45 deg:  Euler    1.000   quaternion  0.00000
  beta = 70 deg:  Euler    2.739   quaternion  0.00000
  beta = 80 deg:  Euler    5.658   quaternion  0.00000
  beta = 85 deg:  Euler   11.444   quaternion  0.00000
  beta = 89 deg:  Euler   59.995   quaternion  0.00000
  (Euler grows like tan(beta); the quaternion frame has no twist at the mode centre)
off-centre twist of the quaternion frame vs the bound tan(d/2):
  d = 0.10 rad: max twist rate 0.0500, tan(d/2) = 0.0500
  d = 0.25 rad: max twist rate 0.1257, tan(d/2) = 0.1257
  d = 0.50 rad: max twist rate 0.2553, tan(d/2) = 0.2553
```

Euler angles twist like \(\tan\beta\), 60 times the pole step at \(\beta = 89°\), which is
exactly where the poles of K03Q21N lie. The parallel-transport frame has no twist at the centre
and stays at the theoretical bound \(\tan(d/2)\) away from it. `--laplace` adds a comparison of
the MAP and Laplace curvature in both frames (slower).

## Radius-field basics

```text
grid 28 x 56 k = 45 orthonormality error 8.69e-15
volume rel err 7.77e-16, inertia rel err [ 0. -0. -0.], centroid [-0.  0.  0.]
mesh area 12.5342 (4pi=12.5664), closure 2.3e-15
pointwise prior sd of log r: 0.1500 (target 0.15)
all basic checks passed
```

Spherical harmonics are orthonormal under the Gauss–Legendre grid to machine precision; volume
and inertia of an ellipsoid from the radial integrals are exact; the triangulated unit sphere is
closed; and the prior on \(\log r\) has the standard deviation that `--sigma-f` asks for.

## Exposure smearing

```text
rotation swept in 30 s: 16.9 deg
  k= 2: attenuation 0.9856, Q=2, GL error 2.8e-05
  k= 6: attenuation 0.8751, Q=3, GL error 1.5e-05
  k=10: attenuation 0.6764, Q=4, GL error 3.0e-06
  k=14: attenuation 0.4288, Q=4, GL error 4.2e-05
566130 Fourier (K=10) amplitude, instantaneous  : 0.214 +- 0.007 mag
566130 Fourier (K=10) amplitude, smeared (30 s) : 0.232 +- 0.007 mag
Fourier without sinc correction: k=2: 0.0592 (true 0.06), k=4: 0.0369 (true 0.04), k=6: 0.0263 (true 0.03), k=8: 0.0167 (true 0.02)
Fourier with    sinc correction: k=2: 0.0600 (true 0.06), k=4: 0.0392 (true 0.04), k=6: 0.0300 (true 0.03), k=8: 0.0213 (true 0.02)
convex: synthetic amplitude 0.32 mag; auto Q = 4
  max |model - reference| [mmag]: instantaneous 3.2, auto-Q 0.325, exptime=0 vs instantaneous 0.0e+00
  fit with instantaneous model: ln L   577.65, residual rms  10.6 mmag, best jitter 2 mmag
  fit with smeared       model: ln L   577.52, residual rms  10.6 mmag, best jitter 2 mmag
hybrid: auto Q = 4; max |model - reference| [mmag]: instantaneous 5.7, auto-Q 0.228
all smearing tests passed
```

- The automatically chosen number of Gauss–Legendre nodes \(Q\) averages harmonic \(k\) over the
  exposure to \(< 5\times10^{-5}\) of the exact sinc attenuation.
- In the period search, the sinc correction restores the true Fourier amplitudes; without it the
  8th harmonic loses 17 %. For 566130 the corrected amplitude is 0.232 mag instead of 0.214.
- In the convex and hybrid models, the smeared model with automatic \(Q\) matches a 64-node
  reference integral to 0.3 mmag; ignoring smearing would be off by 3–6 mmag. For this smooth
  synthetic shape both models fit equally well; the effect matters for sharper shapes and for the
  amplitude prior. See [Exposure smearing](guide/smearing.md).

## Scattering law

```text
convex: stored ln L 325.813619, fixed c_L 325.813619, fitted c_L=0.1 325.813619
hybrid: ndim 38 -> 39; log-likelihood fixed -3423.500557, fitted c_L=0.1 -3423.500557; d logp / d ln c_L = -347.634
recovery, true c_L = 0.1: profile ln L - max = 0.01:0.0, 0.03:-0.0, 0.1:-0.4, 0.3:-1.7, 1:-5.0, 3:-10.9, 10:-15.0, 30:-15.7  -> best 0.01
   c_L values within 2 of max ln L: 0.01 - 0.3
recovery, true c_L = 3.0: profile ln L - max = 0.01:-61.8, 0.03:-57.2, 0.1:-34.6, 0.3:-10.5, 1:-2.6, 3:-0.8, 10:-0.2, 30:0.0  -> best 30
   c_L values within 2 of max ln L: 3 - 30
all c_L tests passed
```

Fitting \(c_L\) and fixing it at the same value give identical likelihoods, to the last printed
digit, in both layers, and the stored reference value is reproduced. On synthetic data with
566130's geometry the true \(c_L\) is always inside \(\Delta\ln L < 2\), but at phase angles of
4–7° the data separate only Lommel–Seeliger-dominated (\(c_L \lesssim 0.3\)) from
Lambert-dominated (\(c_L \gtrsim 3\)) scattering. See [Scattering law](guide/scattering.md).

## Contact-binary model

The analytic two-lobe model (occlusion and shadowing of ellipsoids) is compared with a
point-sampled z-buffer render of the same bodies, for three shapes and four geometries:

| Lobes \(q, b, c/b, s\) | Aspect, phase angle | Amplitude, model / reference [mag] | rms difference [mmag] |
|---|---|---|---|
| 0.8, 0.85, 0.94, 0.9 | 90°, 5° | 0.807 / 0.831 | 4.9 |
| | 90°, 40° | 1.107 / 1.121 | 6.1 |
| | 60°, 20° | 0.380 / 0.387 | 2.6 |
| | 30°, 10° | 0.071 / 0.075 | 1.6 |
| 0.5, 0.7, 0.8, 0.95 | 90°, 5° | 0.704 / 0.725 | 5.7 |
| | 90°, 40° | 1.309 / 1.345 | 9.7 |
| | 60°, 20° | 0.381 / 0.389 | 2.0 |
| | 30°, 10° | 0.069 / 0.072 | 0.9 |
| 1.0, 0.9, 0.9, 0.8 | 90°, 5° | 0.822 / 0.823 | 4.6 |
| | 90°, 40° | 0.893 / 0.895 | 5.0 |
| | 60°, 20° | 0.343 / 0.350 | 3.4 |
| | 30°, 10° | 0.066 / 0.068 | 1.3 |

The largest differences, 10 mmag rms (34 mmag at the deepest minimum), occur equator-on at large
phase angle, where shadows cast by one lobe onto the other are longest; at the phase angles of
survey data they are a few mmag, below typical jitter. On synthetic data with 566130's epochs and
geometry, the likelihood at the true parameters beats a wrong mass ratio (−4103), a deeper
overlap (−178) and the mirror pole (−1019):

```text
recovery: ln L at truth 609.9; q=0.4: -4102.9, s=0.75 (deep overlap): -178.1, mirror pole: -1019.2
all contact-binary tests passed
```

## End-to-end checks

The tests are complemented by checks on real data, documented in the examples:

- **Reproducibility.** A shortened run of K03Q21N with 60 live points recovers the period and
  jitter of the 250-live-point reference run and its retrograde pole to 5°
  ([Quickstart](getting-started/quickstart.md)); a shortened 566130 run agrees with its reference
  in period, jitter, axis ratio and \(c_L\) ([ADES example](examples/ades-566130.md)).
- **Model comparison.** Rerunning `compare_models.py` on the shipped 566130 runs reproduced the
  shipped table exactly ([Working with finished runs](examples/post-processing.md#comparing-models)).
- **Geometry.** For plain lightcurve files every run compares Horizons with the file's distances
  and phase angle; for K03Q21N they agree to \(10^{-7}\) au and \(4\times10^{-6}\) degrees.

## Development diagnostics

These scripts in `tests/` are not pass/fail tests. They answer questions that shaped the method,
and are worth rerunning when changing the model:

| Script | Question |
|---|---|
| `synthetic_check.py` | Does the gauge-fixed likelihood recover a known spin state injected into the real K03Q21N geometry? |
| `smoothness_check.py` | Is the Laplace-marginalised likelihood smooth in pole and period? |
| `width_check.py` | How narrow is the likelihood in phase and period at fixed pole? |
| `optimizer_check.py` | Do the inner-optimiser variants reach the same optimum? |
| `speed_accuracy.py` | How much accuracy do cheaper optimiser settings cost? |
| `inspect_checkpoint.py` | What do the live points of a dynesty checkpoint look like? |

The injection–recovery and facet-resolution results are discussed in section 10 of the
[methods note](papers.md).

## Issues fixed in 0.2.0

Both were found while writing these pages; `CHANGELOG.md` lists all changes.

- **`--resume` after a finished or `--maxcall`-limited run** failed with *"The remaining live
  points have already been added to the list of samples!"*. `finalize_from_checkpoint` now
  removes the final live points from such a checkpoint before continuing.
- **Fast rotators and the default period range.** With `--pmin` at its default of 2 h, a
  sub-hour rotator returned a wrong period without any message. The period search now warns
  when a deeper minimum lies below `--pmin` or the Fourier fit is poor, and stores the warnings
  in `summary.json`. See
  [ADES example](examples/ades-566130.md#what-happens-with-the-default-search-range).
