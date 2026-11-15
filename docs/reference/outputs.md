# Output files

What each results folder contains, and the keys of the summaries. Example values are from the
566130 reference run (`results/566130_X05_cl`, \(c_L\) fitted, 30 s exposures) and the K03Q21N
hybrid reference run.

## Global run: `summary.json`

Percentile triples are `[16th, 50th, 84th]`.

| Key | Content | Example |
|---|---|---|
| `designation`, `horizons_target` | target as given and as queried | `"566130"` |
| `n_points`, `bands`, `n_nights` | data after filtering and clipping | `194`, `["Lg", "Li", "Lr"]`, `5` |
| `phase_angle_range` | min and max phase angle, deg | `[3.93, 6.79]` |
| `geometry_check_vs_file` | largest differences to file geometry (empty for ADES) | `{}` |
| `clipped_epochs` | JD of points removed by `--clip-sigma` | `[2460794.7727]` |
| `line_of_sight_lonlat` | mean direction to the observer, ecliptic deg | `[212.8, 7.0]` |
| `data_fingerprint` | hash of epochs, magnitudes, errors, bands | `"bc74198abb64"` |
| `runtime_s` | wall-clock time | `4299.6` |

`period_search`

| Key | Content |
|---|---|
| `P_best_h` | refined best period |
| `sigma_dchi2_1` | \(\Delta\chi^2 = 1\) error after rescaling |
| `threshold_interval` | \(\chi^2 < \chi^2_\min(1 + \sqrt{2/\nu})\) interval |
| `red_chi2`, `harmonics` | reduced \(\chi^2\) of the Fourier fit, \(K\) |
| `other_minima` | `[period, chi2/nu]` of the six lowest separated minima |
| `warnings` | search-range warnings (empty if none); see [Period search](../guide/period-search.md#warnings-about-the-search-range) |

`amplitude`: `A`, `A_err` (at the mean phase angle), `A0`, `A0_err` (corrected to zero phase),
`ab_min`, `ln_ab_min`, `ln_ab_min_err`.

`priors`: `period` (`["gaussian", P0, sigma, nsig]` or `["uniform", lo, hi]`), `ab`, `bc`
(`[kind, value, sigma_ln]` or null), `pole`, `phi0`, `ln_jitter`, `shape_coeff_tau0`, `lmax`.

`sampler`: `nlive`, `sample`, `niter`, `ncall`, `logZ`, `logZ_err`,
`finalised_from_checkpoint`, `seed`.

`convergence`

| Key | Content |
|---|---|
| `lnZ`, `lnZ_err` | evidence and its error |
| `ess` | Kish effective sample size |
| `evidence_fraction_in_live`, `dlogz_live` | evidence left in the final live points |
| `frac_prograde`, `frac_prograde_err` | posterior mass with \(\beta > 0\) and its error |
| `resumed`, `maxcall_hit` | how the run ended |
| `checks` | `ok`/`warn`/`fail` per check |
| `verdict` | `CONVERGED`, `CONVERGED WITH WARNINGS` or `NOT CONVERGED` |
| `advice` | what to change |

`max_likelihood`: `theta` (the sampled parameters), `logL`, `chi2`, `t_ref_jd` (phase-zero
reference time).

`posterior`

| Key | Content | Example |
|---|---|---|
| `P_h` | sidereal period, h | `[0.1779755, 0.1779773, 0.1779792]` |
| `jitter_mag` | jitter, mag | `[0.0301, 0.0325, 0.0347]` |
| `beta_deg` | pole latitude | `[-50.9, 23.0, 62.0]` |
| `frac_prograde` | mass with \(\beta > 0\) | `0.604` |
| `pole_hpd_sky_fraction` | sky fraction of the 68 % and 95 % highest-density pole regions | |
| `ab`, `bc` | projected-area axis ratios | `[1.194, 1.201, 1.210]` |
| `extent_ratios_a_b`, `extent_ratios_b_c` | extent ratios of the reconstructed polyhedra | |
| `chi2_per_point` | \(\chi^2/N\) of the posterior draws | |
| `c_lambert` | with `--fit-cl` | `[0.011, 0.014, 0.022]` |
| `tau0` | with `--fit-tau` | |

`fit_quality` (best fit): `n_points`, `p_eff`, `p_nominal`, `n_spin`, `n_linear`, `dof`,
`chi2_nominal`, `chi2_red_nominal`, `chi2_jitter`, `chi2_red_jitter`, `rms_mag`,
`median_err_mag`, `jitter_mag`, `p_eff_shape`, `k_shape`, `lmax`. See
[Goodness of fit](../guide/model-comparison.md#goodness-of-fit).

`cohesion` (with a diameter): required cohesion percentiles and the settings; see
[Cohesion and size](../guide/physics.md).

### Other files

| File | Content |
|---|---|
| `nested_results.pkl` | `samples`, `logwt`, `logl`, `logz`, `logzerr`, equal-weight `post`, run `args`, and the posterior shape `ensemble`; read with `lcinv.sampler.load("<run>/nested_results.pkl")` |
| `dynesty.save` | sampler checkpoint for `--resume` |
| `shapes/shape_maxlike.obj`, `shapes/shape_post_###.obj` | convex shapes, z = spin axis, volume 1 |
| `shape_viewer.html`, `shape_uncertainty_viewer.html` | 3D viewers (`make_viewer.py`, `make_uncertainty_viewer.py`) |
| `obj/` | `export_obj.py` meshes |
| `complexity_scan.json`, `fig_complexity.png` | `complexity_scan.py` |
| `resolution_check.json` | `reweight_resolution.py` |

## Binary run: `summary.json`

The keys of a global run, plus

| Key | Content |
|---|---|
| `model` | `"contact binary"` |
| `settings` | `nfacets`, `soft`, `c_lambert`, `fit_cl`, `smear_nodes`, `global_dir` |
| `evidence_comparison` | `lnZ_convex`, `lnZ_binary`, `delta_lnZ_binary_minus_convex`, the Bayes factor and a reading |

## Hybrid run: `summary_hybrid.json`

Top level: `designation`, `n_points`, `settings`, `grid`, `period_prior`, `sensitivity`,
`runtime_s`, `data_fingerprint`, and `spin_states`, a list with one record per spin state:

| Key | Content |
|---|---|
| `name` | `"spin state 1"`, ... |
| `global_center` | centre from the global layer, deg |
| `weight_global`, `weight_local_laplace` | weight in the global posterior and from the local Laplace evidence |
| `lnZ_laplace`, `lnZ_profiled`, `laplace_nonpositive` | Laplace evidence (native and profiled-offset convention), curvature warning |
| `map_logp`, `map_grad_norm` | MAP log density and gradient norm |
| `pole`, `P_h`, `jitter_mag` | posterior summaries |
| `sigma_f`, `ell` | shape hyperparameters |
| `inertia_angle_deg` | angle between spin axis and maximum-inertia axis |
| `principal_moments_median`, `moment_gap_top2`, `extents_median` | mass properties |
| `radius_sd_median_pct_Req` | median radius uncertainty, % of R_eq |
| `convergence` | R-hat, ESS, divergences, verdict |
| `loo` | PSIS-LOO, if ArviZ succeeded |
| `checks` | convexity and self-shadowing |
| `nuts_seconds`, `fit_quality` | timing and goodness of fit at the MAP |

Per spin state: `posterior_spin_state_<k>.npz` (draws; read by every script),
`posterior_spin_state_<k>.nc` (ArviZ, needs h5netcdf), `viewer_spin_state_<k>.html`,
`shape_viewer_spin_state_<k>.html`, `obj/spin_state_<k>/`.
