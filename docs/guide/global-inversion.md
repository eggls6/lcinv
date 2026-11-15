# Global convex inversion

`run_inversion.py` is the first and most important step: it finds every spin state compatible
with the data, their relative probabilities, the period and the noise level, and a convex shape
with uncertainties for each. Everything else (hybrid model, binary model, comparisons) starts from
its output folder.

```bash
python run_inversion.py data/K03Q21N.lc --designation K03Q21N --location X05 \
       --nlive 400 --npool 16 --outdir results/K03Q21N
```

## What a run does

1. **Data and geometry**: reads the file, queries or loads the Horizons cache, computes
   light-time corrected geometry ([Input data](input-data.md)).
2. **Period search** and priors: Fourier periodogram, refinement, amplitude, \(a/b\) bound,
   outlier clipping ([Period search](period-search.md)).
3. **Model**: builds the JAX convex model with the Laplace-marginalised likelihood and compiles it.
4. **Nested sampling** over \((\lambda, \beta, P, \ln\sigma_\mathrm{jit})\) and optionally
   \(\ln c_L\) and \(\ln\tau_0\), with a checkpoint in `<outdir>/dynesty.save`.
5. **Shapes**: optimal and Laplace-drawn shape coefficients for posterior draws, Minkowski
   reconstruction, OBJ files.
6. **Figures and summary**: pole map, corner plot, lightcurves per night, folded lightcurve,
   shape figure, `summary.json`, and a convergence report.

## Parameters and priors

| Parameter | Prior | Options |
|---|---|---|
| pole \((\lambda, \beta)\) | uniform on the sphere | – |
| period \(P\) | truncated Gaussian around the period search (or uniform) | `--period-prior`, `--period`, `--period-sigma`, `--period-nsig` |
| jitter \(\sigma_\mathrm{jit}\) | log-uniform, 1–100 mmag | – |
| \(c_L\) (optional) | log-uniform over `--cl-range` (0.01–100) | `--fit-cl`, `--c-lambert` |
| \(\tau_0\) (optional) | log-uniform over `--tau-range` (0.2–5) | `--fit-tau`, `--tau0` |
| shape coefficients | Gaussian, \(\tau_l = \tau_0/l\), marginalised | `--lmax`, `--nfacets`, `--tau0` |
| axis ratios | soft bounds from the amplitude (\(a/b\)) and \(b/c \ge 1\) | `--ab-prior`, `--bc-prior` |

## The shape model

| Option | Default | Effect |
|---|---|---|
| `--lmax` | 4 | degree of the log-area expansion; 24 coefficients at 4 |
| `--nfacets` | 400 | Gaussian-image facets (Fibonacci sphere) |
| `--tau0` | 0.6 | prior scale of the coefficients, \(\tau_l = \tau_0/l\) |
| `--n-inner` | 80 (400 with `--fit-tau` or `--tau0` > 0.8) | L-BFGS iterations per likelihood call |
| `--n-starts`, `--n-stage1` | 3, 40 | starts and iterations of the coarse \(l \le 2\) stage |

A larger `--tau0` or `--lmax` allows more shape detail. Whether the data support it is a
question for the evidence: see [Shape complexity and model comparison](model-comparison.md).

## The sampler

| Option | Default | Effect |
|---|---|---|
| `--nlive` | 400 | live points; more gives a better-resolved posterior and evidence |
| `--dlogz` | 0.5 | stop when the evidence left in the live points is below this; 0.1 for publication |
| `--sample` | `rslice` | random slice sampling; robust on thin, curved pole ridges |
| `--slices` | dynesty default | slices per proposal for `rslice`/`slice` |
| `--bootstrap` | 0 | ellipsoid enlargement by bootstrap (only relevant for `unif`) |
| `--maxcall` | none | hard limit on likelihood calls |
| `--npool` | 2 | parallel likelihood workers |
| `--seed` | 42 | change it to check reproducibility |

!!! note "Why rslice"
    Once the pole posterior narrows to thin curved ridges, multi-ellipsoid bounds fit it badly
    and `unif` sampling needs more than 100 likelihood calls per iteration. `rslice` costs about
    5 × `slices` calls per iteration whatever the shape, and each worker runs its own slice
    chain, so it scales with `--npool`.

**Resuming.** Every run checkpoints to `<outdir>/dynesty.save`. `--resume` finishes from it,
and `--resume --extra-calls N` first continues sampling for N more calls. Use it when a run
stopped at `--maxcall` or was interrupted; [the quickstart](../getting-started/quickstart.md)
shows a full example.

## Reading the output

A run prints, in this order: the JAX backend, the data summary and geometry check, the period
line, the sampler progress, a convergence report, and a goodness-of-fit line.

```text
nested sampling: 247 iterations, 4066 likelihood calls (efficiency 6.1%), 256 s, ln Z = 729.37 +- 1.63
== convergence: nested sampling: NOT CONVERGED ==
   ln Z = 729.37 +- 1.63 [fail]; evidence left in live points 100% (dlogz 11.77) [fail]; ESS 6 [fail]
   prograde fraction 0.221 +- 0.175 [fail]
   -> 100% of the evidence is still in the final live points: the run stopped early (--maxcall reached); rerun to completion or continue with --resume --extra-calls N
best fit (convex): N = 334, p_eff = 14.9 (nominal 31), chi2_red = 3.47 (quoted errors) / 0.92 (with jitter 19.4 mmag), rms 21.9 mmag
```

This is a deliberately short run (`--maxcall 4000`) from the [quickstart](../getting-started/quickstart.md):
the report says exactly what is missing. [Convergence](convergence.md) explains each check, and
`fit_quality` (the last line) is explained in
[Shape complexity and model comparison](model-comparison.md#goodness-of-fit).

Files in `--outdir`:

| File | Content |
|---|---|
| `summary.json` | period search, priors, sampler, convergence, posterior summaries, fit quality |
| `nested_results.pkl` | dynesty results and the posterior shape ensemble |
| `dynesty.save` | sampler checkpoint (for `--resume`) |
| `fig_periodogram.png` | Fourier \(\chi^2(P)\) and the fine scan |
| `fig_pole_posterior.png` | pole posterior (Mollweide) with period and jitter marginals |
| `fig_corner.png` | corner plot of the sampled parameters |
| `fig_lightcurves.png` | data per night with posterior-predictive curves |
| `fig_folded_lightcurve.png` | data and model over two rotations, with residuals |
| `fig_shapes.png` | max-likelihood silhouettes with posterior draws; \(a/b\), \(b/c\) |
| `shapes/*.obj` | max-likelihood and posterior-draw shapes (z = spin axis, volume 1) |

[Output files](../reference/outputs.md) lists the keys of `summary.json`.

## Optional physics

- `--fit-cl`: sample the Lambert weight of the scattering law ([Scattering law](scattering.md)).
- `--fit-tau`: infer the shape prior scale ([model complexity](model-comparison.md)).
- `--exptime`, `--obstime`, `--smear-nodes`: exposure smearing ([Exposure smearing](smearing.md)).
- `--diameter-km`, `--cohesion-max`, `--size-from-H`: required cohesion and a cohesion limit
  ([Cohesion and size](physics.md)).

All options: [`run_inversion.py` reference](../reference/cli/run_inversion.md).
