# `run_inversion`

*Global layer: convex shape, nested sampling over the spin state.*

```text
python run_inversion.py [-h] [--designation DESIGNATION] [--location LOCATION]
                        [--cache-dir CACHE_DIR] [--outdir OUTDIR]
                        [--drop-bands DROP_BANDS] [--pmin PMIN] [--pmax PMAX]
                        [--harmonics HARMONICS] [--sys-err SYS_ERR]
                        [--clip-sigma CLIP_SIGMA]
                        [--period-prior {gaussian,uniform}] [--period PERIOD]
                        [--period-sigma PERIOD_SIGMA]
                        [--period-nsig PERIOD_NSIG] [--search-window W]
                        [--ab-prior {none,lower,lognormal}] [--ab AB]
                        [--ab-sigma AB_SIGMA] [--ab-model-err AB_MODEL_ERR]
                        [--bc-prior {none,lower,lognormal}] [--bc BC]
                        [--bc-sigma BC_SIGMA]
                        [--amp-phase-slope AMP_PHASE_SLOPE] [--lmax LMAX]
                        [--nfacets NFACETS] [--n-inner N_INNER]
                        [--n-starts N_STARTS] [--n-stage1 N_STAGE1]
                        [--tau0 TAU0] [--fit-tau] [--tau-range LO HI]
                        [--c-lambert C_LAMBERT] [--fit-cl] [--cl-range LO HI]
                        [--nlive NLIVE] [--dlogz DLOGZ]
                        [--sample {rslice,rwalk,slice,unif}]
                        [--bootstrap BOOTSTRAP] [--slices SLICES]
                        [--maxcall MAXCALL] [--seed SEED] [--npool NPOOL]
                        [--exptime EXPTIME] [--obstime {mid,start}]
                        [--smear-nodes SMEAR_NODES] [--stations STATIONS]
                        [--exclude-stations EXCLUDE_STATIONS]
                        [--ades-bands ADES_BANDS]
                        [--ades-default-rms ADES_DEFAULT_RMS]
                        [--ades-sparse-floor ADES_SPARSE_FLOOR]
                        [--ades-dense-rms ADES_DENSE_RMS]
                        [--ades-max-rms ADES_MAX_RMS]
                        [--ades-min-group ADES_MIN_GROUP]
                        [--observer {station,geocentric,local}]
                        [--cohesion-max PA] [--diameter-km DIAMETER_KM]
                        [--H H] [--albedo ALBEDO] [--size-from-H]
                        [--density DENSITY] [--friction-angle FRICTION_ANGLE]
                        [--cohesion-softness COHESION_SOFTNESS]
                        [--device {auto,cpu,gpu}]
                        [--gpu-mem-fraction GPU_MEM_FRACTION]
                        [--gpu-preallocate] [--worker-device {cpu,gpu,auto}]
                        [--worker-threads WORKER_THREADS] [--resume]
                        [--extra-calls EXTRA_CALLS] [--n-post N_POST]
                        [--n-obj N_OBJ]
                        lc
```

```text
End-to-end Bayesian convex inversion of a single lightcurve file.

  python run_inversion.py data/K03Q21N.lc --designation K03Q21N --location X05       --nlive 400 --period-prior gaussian --ab-prior lower --outdir results/K03Q21N

Steps
  1. geometry from JPL Horizons (astroquery; cached CSV in --cache-dir), light-time corrected
  2. multiband Fourier period search, period uncertainty, amplitude -> a/b prior
  3. JAX convex model, Laplace-marginalised shape, jitter
  4. dynesty nested sampling over (lambda, beta, P, ln sigma_jit)
  5. posterior shape ensemble, Minkowski reconstruction, OBJ files, figures, summary.json
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `lc` | – | lightcurve file: Rubin-style .lc/.csv with epoch, mag, err, band (and optional geometry), or an MPC ADES CSV/PSV/XML file |

## Options

| Option | Default | Description |
|---|---|---|
| `--designation`&nbsp;<code>DESIGNATION</code> | – | e.g. K03Q21N (packed) or 2003 QN21 |
| `--location`&nbsp;<code>LOCATION</code> | `X05` | MPC observatory code (X05 = Rubin) |
| `--cache-dir`&nbsp;<code>CACHE_DIR</code> | `data` | folder for cached JPL Horizons tables (reused when they cover the epochs) |
| `--outdir`&nbsp;<code>OUTDIR</code> | `results` | results folder (created if needed) |
| `--drop-bands`&nbsp;<code>DROP_BANDS</code> | `u` | comma-separated bands to remove before fitting (empty string keeps all) |
| `--pmin`&nbsp;<code>PMIN</code> | `2.0` | min rotation period searched [h] |
| `--pmax`&nbsp;<code>PMAX</code> | `30.0` | max rotation period searched [h] |
| `--harmonics`&nbsp;<code>HARMONICS</code> | `6` | Fourier harmonics K of the period search (rotation frequency as fundamental) |
| `--sys-err`&nbsp;<code>SYS_ERR</code> | `0.015` | error floor for period search [mag] |
| `--clip-sigma`&nbsp;<code>CLIP_SIGMA</code> | `5.0` | drop points beyond this many robust sigma from the Fourier fit (0 = off) |
| `--period-prior`&nbsp;<code>PERIOD_PRIOR</code> | `gaussian` | prior on P: truncated Gaussian around the period-search result, or uniform over +- --period-nsig sigma One of `gaussian`, `uniform`. |
| `--period`&nbsp;<code>PERIOD</code> | – | override prior centre [h] |
| `--period-sigma`&nbsp;<code>PERIOD_SIGMA</code> | – | override prior sigma [h] |
| `--period-nsig`&nbsp;<code>PERIOD_NSIG</code> | `5.0` | half-width of the period prior in units of its sigma |
| `--search-window`&nbsp;<code>W</code> | – | period search only within --period x (1 +- W) instead of the full 1 h .. --pmax grid; needed for multi-apparition data, whose full grid has ~1e7 trial periods |
| `--ab-prior`&nbsp;<code>AB_PRIOR</code> | `lower` | a/b prior from the amplitude: one-sided lower bound or lognormal One of `none`, `lower`, `lognormal`. |
| `--ab`&nbsp;<code>AB</code> | – | override a/b value |
| `--ab-sigma`&nbsp;<code>AB_SIGMA</code> | – | sigma of ln(a/b) |
| `--ab-model-err`&nbsp;<code>AB_MODEL_ERR</code> | `0.05` | extra ln(a/b) uncertainty for non-ellipsoidal shapes/scattering |
| `--bc-prior`&nbsp;<code>BC_PRIOR</code> | `lower` | b/c prior; default: soft lower bound b/c &gt;= 1, i.e. spin about the shortest axis (relaxed principal-axis rotator) One of `none`, `lower`, `lognormal`. |
| `--bc`&nbsp;<code>BC</code> | `1.0` | b/c prior value |
| `--bc-sigma`&nbsp;<code>BC_SIGMA</code> | `0.05` | sigma of ln(b/c) |
| `--amp-phase-slope`&nbsp;<code>AMP_PHASE_SLOPE</code> | `0.03` | amplitude-phase slope m [mag/deg] for correcting the amplitude to zero phase angle (a/b bound) |
| `--lmax`&nbsp;<code>LMAX</code> | `4` | maximum spherical-harmonic degree of the log facet areas (shape resolution) |
| `--nfacets`&nbsp;<code>NFACETS</code> | `400` | number of Gaussian-image facets (Fibonacci sphere) |
| `--n-inner`&nbsp;<code>N_INNER</code> | – | L-BFGS iterations of the shape optimisation per likelihood call (default 80; 400 with --fit-tau or tau0 &gt; 0.8, where 80 does not converge) |
| `--n-starts`&nbsp;<code>N_STARTS</code> | `3` | starts of the inner shape optimisation per likelihood call; the best is kept |
| `--n-stage1`&nbsp;<code>N_STAGE1</code> | `40` | L-BFGS iterations of the coarse l &lt;= 2 stage that seeds each start |
| `--tau0`&nbsp;<code>TAU0</code> | `0.6` | prior scale of the shape coefficients, tau_l = tau0 / l (fixed) |
| `--fit-tau` | off | infer tau0 from the data (log-uniform prior over --tau-range): data-driven shape complexity within one run |
| `--tau-range`&nbsp;<code>LO HI</code> | `0.2 5.0` | log-uniform prior range of tau0 for --fit-tau |
| `--c-lambert`&nbsp;<code>C_LAMBERT</code> | `0.1` | Lambert weight c_L of the scattering law S = mu mu0 [1/(mu+mu0) + c_L] (fixed) |
| `--fit-cl` | off | sample ln c_L (log-uniform prior over --cl-range) instead of fixing it |
| `--cl-range`&nbsp;<code>LO HI</code> | `0.01 100.0` | prior range of c_L for --fit-cl (0.01 ~ pure Lommel-Seeliger, 100 ~ pure Lambert) |
| `--nlive`&nbsp;<code>NLIVE</code> | `400` | dynesty live points (more: smoother evidence and posterior, proportionally slower) |
| `--dlogz`&nbsp;<code>DLOGZ</code> | `0.5` | stop when the evidence left in the live points is below this (in ln Z) |
| `--sample`&nbsp;<code>SAMPLE</code> | `rslice` | dynesty proposal (rslice: robust on thin pole ridges; unif: fast early, but collapses on curved posteriors) One of `rslice`, `rwalk`, `slice`, `unif`. |
| `--bootstrap`&nbsp;<code>BOOTSTRAP</code> | `0` | dynesty bootstrap (only matters for unif) |
| `--slices`&nbsp;<code>SLICES</code> | – | slices per proposal for (r)slice |
| `--maxcall`&nbsp;<code>MAXCALL</code> | – | stop after this many likelihood calls (finish later with --resume --extra-calls) |
| `--seed`&nbsp;<code>SEED</code> | `42` | sampler seed (repeat with another seed to check) |
| `--npool`&nbsp;<code>NPOOL</code> | `2` | parallel likelihood workers |
| `--resume` | off | finish from &lt;outdir&gt;/dynesty.save instead of starting a new run |
| `--extra-calls`&nbsp;<code>EXTRA_CALLS</code> | `0` | with --resume: continue sampling for this many calls before adding live points |
| `--n-post`&nbsp;<code>N_POST</code> | `150` | posterior draws for shapes/curves |
| `--n-obj`&nbsp;<code>N_OBJ</code> | `30` | posterior shape OBJ files to write |

## Exposure smearing

| Option | Default | Description |
|---|---|---|
| `--exptime`&nbsp;<code>EXPTIME</code> | – | exposure time [s] for rows without one (ADES 'exp' or a lightcurve 'exptime' column take precedence); default: no smearing |
| `--obstime`&nbsp;<code>OBSTIME</code> | `mid` | whether epochs are exposure mid-times (ADES convention) or start times One of `mid`, `start`. |
| `--smear-nodes`&nbsp;<code>SMEAR_NODES</code> | – | Gauss-Legendre nodes per exposure (default: chosen from P, T and lmax) |

## ADES input (detected automatically)

| Option | Default | Description |
|---|---|---|
| `--stations`&nbsp;<code>STATIONS</code> | – | comma-separated MPC codes to use, e.g. X05 or X05,I41,F51 (default all) |
| `--exclude-stations`&nbsp;<code>EXCLUDE_STATIONS</code> | – | comma-separated MPC codes to drop |
| `--ades-bands`&nbsp;<code>ADES_BANDS</code> | – | comma-separated ADES band codes to keep |
| `--ades-default-rms`&nbsp;<code>ADES_DEFAULT_RMS</code> | `0.1` | uncertainty when rmsMag is missing [mag] |
| `--ades-sparse-floor`&nbsp;<code>ADES_SPARSE_FLOOR</code> | `0.05` | extra error (quadrature) for groups with median error &gt; --ades-dense-rms |
| `--ades-dense-rms`&nbsp;<code>ADES_DENSE_RMS</code> | `0.03` | groups whose median error exceeds this are treated as sparse and get --ades-sparse-floor [mag] |
| `--ades-max-rms`&nbsp;<code>ADES_MAX_RMS</code> | – | drop rows with larger error |
| `--ades-min-group`&nbsp;<code>ADES_MIN_GROUP</code> | `5` | drop (station, band) groups with fewer rows |
| `--observer`&nbsp;<code>OBSERVER</code> | `station` | observer: Horizons table per station; Horizons geocentre; or geocentre from astropy's built-in Earth ephemeris (only heliocentric vectors queried) One of `station`, `geocentric`, `local`. |

## Cohesion limit (Sanchez & Scheeres 2014)

| Option | Default | Description |
|---|---|---|
| `--cohesion-max`&nbsp;<code>PA</code> | – | penalise shapes/spins that need more cohesive strength than this (Pa); ~25 Pa is the Sanchez & Scheeres (2014) estimate for fine-regolith rubble piles. Needs --diameter-km |
| `--diameter-km`&nbsp;<code>DIAMETER_KM</code> | – | volume-equivalent diameter (km); also enables the required-cohesion report |
| `--H`&nbsp;<code>H</code> | – | absolute magnitude: diameter from the CNEOS size estimator if --diameter-km is not given |
| `--albedo`&nbsp;<code>ALBEDO</code> | `0.15` | geometric albedo for the H-based diameter |
| `--size-from-H` | off | diameter from H looked up in the JPL Small-Body Database (CNEOS formula) |
| `--density`&nbsp;<code>DENSITY</code> | `1700.0` | bulk density (kg/m^3) |
| `--friction-angle`&nbsp;<code>FRICTION_ANGLE</code> | `90.0` | internal friction angle (deg); 90 = conservative limit (their eq. 24) |
| `--cohesion-softness`&nbsp;<code>COHESION_SOFTNESS</code> | `0.1` | width of the soft limit in ln(sigma_req / cohesion_max) |

## JAX device and memory

| Option | Default | Description |
|---|---|---|
| `--device`&nbsp;<code>DEVICE</code> | `auto` | JAX platform of the main process (auto = GPU if available) One of `auto`, `cpu`, `gpu`. |
| `--gpu-mem-fraction`&nbsp;<code>GPU_MEM_FRACTION</code> | – | cap the GPU memory fraction per process (sets XLA_PYTHON_CLIENT_MEM_FRACTION) |
| `--gpu-preallocate` | off | re-enable JAX's default GPU preallocation (off by default) |
| `--worker-device`&nbsp;<code>WORKER_DEVICE</code> | `cpu` | JAX platform of the likelihood pool workers (default cpu; with gpu, all workers share one GPU without preallocation) One of `cpu`, `gpu`, `auto`. |
| `--worker-threads`&nbsp;<code>WORKER_THREADS</code> | `1` | threads per CPU pool worker (default 1; npool x threads should not exceed the number of physical cores) |

<small>72 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
