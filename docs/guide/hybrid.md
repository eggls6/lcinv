# Hybrid model

`run_hybrid.py` is the local layer. It takes the spin states found by a finished global run and
samples spin and shape **jointly** within each of them with NUTS, using a richer shape model with
physical priors. It is the step to run when you need shape uncertainties you can trust, or
whether the body spins about its axis of maximum inertia.

```bash
python run_inversion.py data/K03Q21N.lc --designation K03Q21N --outdir results/K03Q21N       # global
python run_hybrid.py   data/K03Q21N.lc --designation K03Q21N --global-dir results/K03Q21N \
       --outdir results/K03Q21N_hybrid --sensitivity                                         # local
```

`--global-dir` supplies the spin-state centres, the period prior, the clipped outliers and the
data selection. Without a global run, give the pole centres yourself with
`--poles "lam,bet;lam,bet"` (degrees) and the period with `--period`/`--period-sigma`.

## How spin states are found

`lcinv.hybrid.modes.find_modes` clusters the global posterior poles on the sphere: cores within
20° (`r_core`), members within 45° (`r_member`), clusters below 5 % of the mass dropped, at most
four. Each becomes a spin state. In each one lcinv then

1. finds the maximum a posteriori point from `--starts` starting points (`--map-iter` L-BFGS
   iterations, Optax),
2. computes the Laplace evidence there, which sets the spin state's **weight**
   \(w_k = Z_k / \sum_j Z_j\),
3. runs NUTS with a dense mass matrix initialised from the Laplace covariance.

## The model

**Shape.** A star-shaped log-radius field on a Gauss-Legendre × uniform-longitude grid:

\[
r(\hat u) = \exp\Big(\sum_{l=2}^{l_\max}\sum_m a_{lm} Y_{lm}(\hat u)\Big),\qquad
a_{lm} = \tau_l(\sigma_f, \ell)\,z_{lm},\quad z_{lm}\sim\mathcal N(0,1),
\]

with \(\tau_l^2 \propto \sigma_f^2\,(1 + l(l+1)/\ell^2)^{-\nu}\), normalised so that \(\sigma_f\) is
the pointwise standard deviation of \(\ln r\). The degree \(l = 0\) (scale) is absorbed by the
offsets and \(l = 1\) (a translation) is weakly identified, so both are dropped. The real
spherical harmonics are exactly orthonormal on this grid (error about \(10^{-15}\)), which keeps
the per-degree damping and the rotation symmetry about the spin axis intact.

**Physical priors.** Volume, centre of mass and the inertia tensor are explicit radial integrals.
A soft prior asks for small off-diagonal inertia (\(I_{xz}/I_{zz}\), \(I_{yz}/I_{zz}\)) and for
\(I_{zz}\) to be the largest moment, i.e. rotation about the maximum-inertia axis. A weak penalty
discourages local dents; whether the posterior shapes are convex enough for the facet sum is
checked afterwards.

**Spin.** The pole lives in gnomonic tangent-plane coordinates around the spin state's centre,
with the Jacobian of a uniform-on-sphere prior. This chart has no singularity at \(|\beta| = 90°\)
and cannot reach the mirror solution. The period is non-centred, \(P = P_0 + \sigma_P z_P\), and
the phase is gauge-fixed as in the global layer.

**Body frame.** By default (`--frame parallel`) the frame is carried from the spin state's centre
by the minimal rotation taking each pole there (quaternions). Moving the pole then twists the
frame by at most \(\tan(d/2)\) per unit displacement, against \(\tan\beta\) for Euler angles, which
diverges near the ecliptic poles. That divergence created a near-flat ridge between \(\lambda\) and
the shape orientation and spoiled pole mixing. `--frame euler` restores the old behaviour.

**Likelihood.** Lommel-Seeliger plus Lambert scattering summed over lit and visible facets, with
limb and terminator softened (\(\mu \to T\,\mathrm{softplus}(\mu/T)\), \(T = 0.005\)) for smooth
gradients. Band offsets and the phase slope are **marginalised** analytically, and the pointwise
log-likelihood is stored for LOO.

## Options

| Option | Default | Effect |
|---|---|---|
| `--lmax` | 5 | maximum degree of the radius field |
| `--nlat` | 16 | latitude nodes of the grid (and mesh) |
| `--nu` | 2.0 | spectral slope: how strongly high degrees are damped |
| `--sigma-f` | 0.15 | prior median of the spread of \(\ln r\) |
| `--ell` | 3.0 | prior median of the angular scale |
| `--inertia-offdiag`, `--inertia-order` | 0.005, 0.005 | widths of the principal-axis prior |
| `--convexity-scale` | 0.02 | dent penalty; ≤ 0 disables it |
| `--frame` | `parallel` | `parallel` (quaternions) or `euler` |
| `--fit-cl`, `--cl-prior` | off, `0.3 2.0` | sample \(c_L\) with a log-normal prior (median, sd of \(\ln c_L\)) |
| `--chains`, `--warmup`, `--samples` | 2, 400, 400 | NUTS chains and iterations per spin state |
| `--tree-depth`, `--target-accept` | 7, 0.9 | NUTS tuning |
| `--starts`, `--map-iter` | 4, 2000 | MAP search per spin state |
| `--sensitivity` | off | refit MAP and weights at \(\nu = 1.5\), \(\nu = 3\) and a tighter \(\sigma_f\) |
| `--check-draws`, `--shadow-draws`, `--shadow-epochs` | 20, 4, 30 | post-sampling checks |

All options: [`run_hybrid.py` reference](../reference/cli/run_hybrid.md).

## Checks after sampling

For each spin state the report gives

- **convergence**: split rank-normalised R-hat and bulk ESS on the chart coordinates, period and
  jitter, and the number of divergences (see [Convergence](convergence.md));
- **convexity**: the largest protrusion of the posterior shapes beyond any facet plane, in units
  of the equatorial radius;
- **self-shadowing**: a ray-traced (Möller-Trumbore) occlusion test on selected draws gives the
  magnitude error made by the convex facet sum. This number says whether a non-convex posterior
  shape invalidates the scattering approximation;
- **prior sensitivity** with `--sensitivity`.

## Outputs

| File | Content |
|---|---|
| `summary_hybrid.json` | per spin state: weight, pole, period, jitter, checks, convergence; global weights for comparison |
| `posterior_spin_state_<k>.npz` | draws of spin state k (read by all scripts) |
| `posterior_spin_state_<k>.nc` | ArviZ InferenceData with pointwise log-likelihood (needs h5netcdf) |
| `fig_pole_hybrid.png`, `fig_traces.png`, `fig_diagnostics.png` | pole posterior, traces, diagnostics |
| `fig_radius_maps.png` | mean radius and its uncertainty per spin state |
| `fig_lightcurves_hybrid.png`, `fig_folded_lightcurve_hybrid.png` | data and posterior-predictive curves |
| `viewer_spin_state_<k>.html` | 3D radius-uncertainty viewer |
| `shape_viewer_spin_state_<k>.html` | 3D shape viewer |

`make_hybrid_report.py <outdir>` rebuilds every figure and viewer from the saved posteriors
without resampling.

!!! note "Older results folders"
    Spin states used to be called modes (`posterior_mode1.npz`, `"modes"`). Every script still
    reads those folders; `python -m lcinv.hybrid.naming <outdir>` renames them in place.

## Cost

One log-density gradient costs about 10 ms on a 16 × 32 mesh (\(l_\max = 5\), 37 parameters per
spin state). Against a 48 × 96 reference mesh the coarser mesh adds about 2.4 mmag rms of model
error, well below a typical jitter of 12 mmag. NUTS needs 60 to 130 leapfrog steps per iteration.
Chains run in parallel on CPU host devices, or vectorised on one GPU.

The [hybrid example](../examples/hybrid-k03q21n.md) runs this on the bundled data, and
[Reference results](../examples/reference-results.md#k03q21n-hybrid-model) shows the full-length
run.
