# Model comparison

Data fingerprint `2b0a31a485aa` · equal prior odds · Jeffreys / Kass–Raftery scale (|Δ ln Z| < 1 not worth mentioning, 1–2.3 weak, 2.3–5 moderate, > 5 strong)

| Model | ln Z method | N | p_eff | χ²_red (errors) | χ²_red (jitter) | rms [mmag] | jitter [mmag] | ln Z | Δ ln Z | reading | P(model) | P(best) | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid_quat] | Laplace (per spin state, summed) | 334 | 36.7 | 2.25 | 1.14 | 17.2 | 12.2 | 795.30 ± 1.00 | 0.00 | best | 0.948 | 0.98 | NOT CONVERGED |
| hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid] | Laplace (per spin state, summed) | 334 | 36.8 | 2.21 | 1.14 | 17.1 | 12.2 | 792.40 ± 1.00 | -2.90 | moderate (worse) | 0.052 | 0.02 | NOT CONVERGED |
| convex lmax 4, τ0 0.6, c_L 0.1 [K03Q21N_rslice] | nested sampling | 334 | 15.1 | 3.37 | 1.09 | 21.6 | 17.3 | 732.52 ± 0.35 | -62.78 | strong (worse) | 0.000 | 0.00 | CONVERGED |
| convex lmax 4, τ0 0.6, c_L 0.1 [K03Q21N] | nested sampling | 334 | 15.1 | 3.34 | 1.10 | 21.5 | 17.2 | 732.21 ± 0.25 | -63.10 | strong (worse) | 0.000 | 0.00 | NOT CONVERGED |

## Recommendation

**Most probable model: hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid_quat]**

- *hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid_quat]* is moderately favoured over *hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid]* (Δ ln Z = 2.9 ± 1.4, odds ≈ 18:1).
- Caution: the best model's run is **NOT CONVERGED**; rerun it to convergence before trusting the comparison.
- Caution: the best model's evidence is from Laplace (per spin state, summed); confirm it with a full nested run.
- Fit adequacy: χ²_red = 2.3 with the quoted errors (median 10.9 mmag); the model needs 11 mmag extra scatter. Either the errors are underestimated or there is signal no model here captures (shape detail, scattering, non-principal-axis rotation, calibration).
- The data constrain p_eff ≈ 37 parameters of the best model (nominal 41).

Spin states of *hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid_quat]*: spin state 1 (pole 241°, -66°, weight 0.01, NOT CONVERGED); spin state 2 (pole 69°, 59°, weight 0.99, NOT CONVERGED)

Spin states of *hybrid star-shaped, lmax 5 (2 spin state(s)) [K03Q21N_hybrid]*: spin state 1 (pole 247°, -80°, weight 0.18, NOT CONVERGED); spin state 2 (pole 69°, 57°, weight 0.82, NOT CONVERGED)

Notes: ln Z values are only meaningful as differences between runs on the same data. χ²_red uses dof = N − p_eff, where p_eff = tr(H_lik H_post⁻¹) counts the parameters the data actually constrain (shape coefficients shrunk by their prior count less than 1). Importance-sampled rows with low ESS are lower bounds (≥) and get no probability. Hybrid evidences are Laplace approximations in the profiled-offset convention of the convex/binary runs (±1 added).
