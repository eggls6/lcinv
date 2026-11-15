# `run_hybrid`

*Local layer: radius-field shape and spin, NUTS within each spin state.*

```text
python run_hybrid.py [-h] [--designation DESIGNATION] [--location LOCATION]
                     [--cache-dir CACHE_DIR] [--global-dir GLOBAL_DIR]
                     [--poles POLES] [--period PERIOD]
                     [--period-sigma PERIOD_SIGMA] [--outdir OUTDIR]
                     [--lmax LMAX] [--nlat NLAT] [--nu NU] [--sigma-f SIGMA_F]
                     [--ell ELL] [--inertia-offdiag INERTIA_OFFDIAG]
                     [--inertia-order INERTIA_ORDER]
                     [--convexity-scale CONVEXITY_SCALE]
                     [--c-lambert C_LAMBERT] [--fit-cl]
                     [--cl-prior MEDIAN LNSD] [--frame {parallel,euler}]
                     [--starts STARTS] [--map-iter MAP_ITER] [--warmup WARMUP]
                     [--samples SAMPLES] [--chains CHAINS]
                     [--tree-depth TREE_DEPTH] [--target-accept TARGET_ACCEPT]
                     [--check-draws CHECK_DRAWS] [--shadow-draws SHADOW_DRAWS]
                     [--shadow-epochs SHADOW_EPOCHS] [--sensitivity]
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
                     [--cohesion-max PA] [--diameter-km DIAMETER_KM] [--H H]
                     [--albedo ALBEDO] [--size-from-H] [--density DENSITY]
                     [--friction-angle FRICTION_ANGLE]
                     [--cohesion-softness COHESION_SOFTNESS]
                     [--device {auto,cpu,gpu}]
                     [--gpu-mem-fraction GPU_MEM_FRACTION] [--gpu-preallocate]
                     lc
```

```text
Hybrid inversion: spin states (pole solutions) of the global nested-sampling run ->
radius-field NUTS per spin state.

  python run_hybrid.py data/K03Q21N.lc --designation K03Q21N       --global-dir results/K03Q21N --outdir results/K03Q21N_hybrid

Requires a finished global run (run_inversion.py) in --global-dir, or explicit --poles.
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `lc` | – | lightcurve file (plain or ADES), the same data as the global run |

## Options

| Option | Default | Description |
|---|---|---|
| `--designation`&nbsp;<code>DESIGNATION</code> | – | target designation (default: from an ADES file) |
| `--location`&nbsp;<code>LOCATION</code> | `X05` | MPC observatory code for plain lightcurves |
| `--cache-dir`&nbsp;<code>CACHE_DIR</code> | `data` | folder of cached JPL Horizons tables |
| `--global-dir`&nbsp;<code>GLOBAL_DIR</code> | – | output folder of run_inversion.py |
| `--poles`&nbsp;<code>POLES</code> | – | explicit spin-state pole centres 'lam,bet;lam,bet' (deg) |
| `--period`&nbsp;<code>PERIOD</code> | – | centre of the Gaussian period prior [h] (default: the global run's period-search result) |
| `--period-sigma`&nbsp;<code>PERIOD_SIGMA</code> | – | width of the period prior [h] (default: half the global run's period threshold interval) |
| `--outdir`&nbsp;<code>OUTDIR</code> | `results/hybrid` | results folder |
| `--lmax`&nbsp;<code>LMAX</code> | `5` | maximum spherical-harmonic degree of the log-radius field |
| `--nlat`&nbsp;<code>NLAT</code> | `16` | mesh/quadrature latitude nodes |
| `--nu`&nbsp;<code>NU</code> | `2.0` | Matern-like spectral slope (damping) |
| `--sigma-f`&nbsp;<code>SIGMA_F</code> | `0.15` | prior median of sd(log r) |
| `--ell`&nbsp;<code>ELL</code> | `3.0` | prior median of angular scale |
| `--inertia-offdiag`&nbsp;<code>INERTIA_OFFDIAG</code> | `0.005` | scale of the soft constraint on the products of inertia I_xz, I_yz (principal-axis rotation) |
| `--inertia-order`&nbsp;<code>INERTIA_ORDER</code> | `0.005` | scale of the soft constraint that z is the axis of maximum inertia |
| `--convexity-scale`&nbsp;<code>CONVEXITY_SCALE</code> | `0.02` | &lt;=0 disables |
| `--c-lambert`&nbsp;<code>C_LAMBERT</code> | `0.1` | fixed Lambert weight c_L |
| `--fit-cl` | off | sample ln c_L (lognormal prior, --cl-prior) |
| `--cl-prior`&nbsp;<code>MEDIAN LNSD</code> | `0.3 2.0` | lognormal prior of c_L: median and sd of ln c_L (default 0.3, 2.0) |
| `--frame`&nbsp;<code>FRAME</code> | `parallel` | body frame: quaternion parallel transport (default) or legacy Euler angles One of `parallel`, `euler`. |
| `--starts`&nbsp;<code>STARTS</code> | `4` | MAP starts per spin state (elongation at different angles) |
| `--map-iter`&nbsp;<code>MAP_ITER</code> | `2000` | L-BFGS iterations of each MAP start |
| `--warmup`&nbsp;<code>WARMUP</code> | `400` | NUTS warm-up iterations per chain |
| `--samples`&nbsp;<code>SAMPLES</code> | `400` | NUTS samples per chain |
| `--chains`&nbsp;<code>CHAINS</code> | `2` | NUTS chains per spin state (run in parallel on CPU host devices) |
| `--tree-depth`&nbsp;<code>TREE_DEPTH</code> | `7` | maximum NUTS tree depth |
| `--target-accept`&nbsp;<code>TARGET_ACCEPT</code> | `0.9` | NUTS target acceptance probability (raise to 0.95 against divergences) |
| `--check-draws`&nbsp;<code>CHECK_DRAWS</code> | `20` | draws for the convexity check |
| `--shadow-draws`&nbsp;<code>SHADOW_DRAWS</code> | `4` | posterior draws for the self-shadowing check |
| `--shadow-epochs`&nbsp;<code>SHADOW_EPOCHS</code> | `30` | epochs per draw for the self-shadowing check |
| `--sensitivity` | off | MAP+Laplace for nu in {1.5, 3} |

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

<small>55 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
