# `run_binary`

*Contact-binary model (two ellipsoidal lobes), nested sampling.*

```text
python run_binary.py [-h] [--global-dir GLOBAL_DIR]
                     [--designation DESIGNATION] [--location LOCATION]
                     [--cache-dir CACHE_DIR] [--drop-bands DROP_BANDS]
                     [--outdir OUTDIR] [--period PERIOD]
                     [--period-sigma PERIOD_SIGMA] [--period-nsig PERIOD_NSIG]
                     [--q-range Q_RANGE Q_RANGE] [--b-range B_RANGE B_RANGE]
                     [--cb-range CB_RANGE CB_RANGE]
                     [--s-range S_RANGE S_RANGE] [--nfacets NFACETS]
                     [--soft SOFT] [--c-lambert C_LAMBERT] [--fit-cl]
                     [--cl-range CL_RANGE CL_RANGE] [--nlive NLIVE]
                     [--dlogz DLOGZ] [--sample {rslice,rwalk,slice,unif}]
                     [--slices SLICES] [--maxcall MAXCALL] [--seed SEED]
                     [--npool NPOOL] [--resume] [--n-post N_POST]
                     [--n-obj N_OBJ] [--diameter-km DIAMETER_KM] [--replot]
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
                     [--cohesion-max PA] [--H H] [--albedo ALBEDO]
                     [--size-from-H] [--density DENSITY]
                     [--friction-angle FRICTION_ANGLE]
                     [--cohesion-softness COHESION_SOFTNESS]
                     [--device {auto,cpu,gpu}]
                     [--gpu-mem-fraction GPU_MEM_FRACTION] [--gpu-preallocate]
                     [--worker-device {cpu,gpu,auto}]
                     [--worker-threads WORKER_THREADS]
                     [lc]
```

```text
Contact-binary inversion: two ellipsoidal lobes in contact, fitted by nested sampling.

  python run_binary.py --global-dir results/566130_X05_cl --outdir results/566130_X05_binary

With --global-dir the data selection (file, stations, observer, exposure time, clipped
outliers), the period prior and c_L settings are taken from that run_inversion.py run, so
the evidences of the two models are directly comparable (summary.json: evidence_comparison).
Without it give the lightcurve and the period:

  python run_binary.py data/566130_ades.csv --stations X05 --observer local --exptime 30 \
      --period 0.1779775 --period-sigma 2e-6 --outdir results/566130_binary

Outputs (like run_inversion.py): summary.json, nested_results.pkl, shapes/*.obj (watertight
union meshes, unit volume-equivalent radius, or km with --diameter-km), fig_pole_posterior.png,
fig_corner.png, fig_lightcurves.png, fig_folded_lightcurve.png, shape_viewer.html.
check_convergence.py works on the folder; --replot rebuilds the products of an existing
--outdir without sampling.
```

## Positional arguments

| Option | Default | Description |
|---|---|---|
| `lc` | – | lightcurve / ADES file (optional with --global-dir) |

## Options

| Option | Default | Description |
|---|---|---|
| `--global-dir`&nbsp;<code>GLOBAL_DIR</code> | – | run_inversion.py output: data, period prior, c_L |
| `--designation`&nbsp;<code>DESIGNATION</code> | – | target designation (default: from --global-dir or an ADES file) |
| `--location`&nbsp;<code>LOCATION</code> | `X05` | MPC observatory code for plain lightcurves |
| `--cache-dir`&nbsp;<code>CACHE_DIR</code> | `data` | folder of cached JPL Horizons tables |
| `--drop-bands`&nbsp;<code>DROP_BANDS</code> | `u` | comma-separated bands to remove |
| `--outdir`&nbsp;<code>OUTDIR</code> | `results/binary` | results folder |
| `--period`&nbsp;<code>PERIOD</code> | – | period prior centre [h] |
| `--period-sigma`&nbsp;<code>PERIOD_SIGMA</code> | – | period prior sigma [h] |
| `--period-nsig`&nbsp;<code>PERIOD_NSIG</code> | `5.0` | truncation of the Gaussian period prior, in sigma |
| `--nfacets`&nbsp;<code>NFACETS</code> | `800` | facets per lobe |
| `--soft`&nbsp;<code>SOFT</code> | `0.02` | soft-edge width of occlusion and neck |
| `--c-lambert`&nbsp;<code>C_LAMBERT</code> | – | fixed c_L (default: from --global-dir or 0.1) |
| `--fit-cl` | off | sample ln c_L |
| `--cl-range`&nbsp;<code>CL_RANGE CL_RANGE</code> | `0.01 100.0` | prior range of c_L for --fit-cl (log-uniform) |
| `--nlive`&nbsp;<code>NLIVE</code> | `400` | dynesty live points |
| `--dlogz`&nbsp;<code>DLOGZ</code> | `0.5` | stop when the evidence left in the live points is below this |
| `--sample`&nbsp;<code>SAMPLE</code> | `rslice` | dynesty proposal One of `rslice`, `rwalk`, `slice`, `unif`. |
| `--slices`&nbsp;<code>SLICES</code> | – | slices per proposal for (r)slice |
| `--maxcall`&nbsp;<code>MAXCALL</code> | – | stop after this many likelihood calls |
| `--seed`&nbsp;<code>SEED</code> | `42` | sampler seed |
| `--npool`&nbsp;<code>NPOOL</code> | `2` | parallel likelihood workers |
| `--resume` | off | continue from outdir/dynesty.save if it exists |
| `--n-post`&nbsp;<code>N_POST</code> | `150` | posterior draws for curves and statistics |
| `--n-obj`&nbsp;<code>N_OBJ</code> | `30` | posterior shape OBJ files |
| `--diameter-km`&nbsp;<code>DIAMETER_KM</code> | – | scale OBJ files to this D_eq [km] |
| `--replot` | off | rebuild products of an existing --outdir |

## Contact-binary shape priors (uniform)

| Option | Default | Description |
|---|---|---|
| `--q-range`&nbsp;<code>Q_RANGE Q_RANGE</code> | `0.3 1.0` | lobe size ratio |
| `--b-range`&nbsp;<code>B_RANGE B_RANGE</code> | `0.5 1.0` | lobe b/a |
| `--cb-range`&nbsp;<code>CB_RANGE CB_RANGE</code> | `0.5 1.0` | lobe c/b |
| `--s-range`&nbsp;<code>S_RANGE S_RANGE</code> | `0.7 0.98` | centre distance / (a1 + a2); 1 = touching, smaller = more overlap |

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

<small>55 options. Generated from the argparse definition by `tools/gen_cli_docs.py`.</small>
