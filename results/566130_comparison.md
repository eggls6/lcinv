# Model comparison

Data fingerprint `bc74198abb64` · equal prior odds · Jeffreys / Kass–Raftery scale (|Δ ln Z| < 1 not worth mentioning, 1–2.3 weak, 2.3–5 moderate, > 5 strong)

| Model | ln Z method | N | p_eff | χ²_red (errors) | χ²_red (jitter) | rms [mmag] | jitter [mmag] | ln Z | Δ ln Z | reading | P(model) | P(best) | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convex lmax 6, τ0 1.2, c_L fit, 30 s exp. [scan] | importance sampling (ESS 2) | 194 | 17.8 | 5.94 | 0.98 | 27.1 | 26.2 | ≥ 364.04 ± 0.97 | 41.53 | ≥ strong (better) | – | – | – |
| convex lmax 4, τ0 1.2, c_L fit, 30 s exp. [scan] | importance sampling (ESS 2) | 194 | 16.5 | 6.31 | 1.05 | 28.1 | 26.2 | ≥ 359.12 ± 0.79 | 36.61 | ≥ strong (better) | – | – | – |
| convex lmax 8, τ0 0.6, c_L fit, 30 s exp. [scan] | importance sampling (ESS 76) | 194 | 15.7 | 8.77 | 1.01 | 33.2 | 32.3 | 322.50 ± 0.30 | 0.00 | best | 0.641 | 0.97 | – |
| convex lmax 6, τ0 0.6, c_L fit, 30 s exp. [scan] | importance sampling (ESS 86) | 194 | 15.5 | 8.82 | 1.02 | 33.3 | 32.3 | 321.78 ± 0.29 | -0.73 | not worth mentioning (worse) | 0.309 | 0.03 | – |
| convex lmax 4, τ0 0.6, c_L fit, 30 s exp. | nested sampling | 194 | 14.6 | 8.96 | 1.04 | 33.7 | 32.5 | 319.63 ± 0.29 | -2.87 | moderate (worse) | 0.036 | 0.00 | CONVERGED |
| convex lmax 4, τ0 0.6, c_L 0.1 | nested sampling | 194 | 13.7 | 9.51 | 1.02 | 34.8 | 33.7 | 318.63 ± 0.25 | -3.88 | moderate (worse) | 0.013 | 0.00 | CONVERGED |
| convex lmax 4, τ0 0.6, c_L 0.1, 30 s exp. | nested sampling | 194 | 13.5 | 9.98 | 1.05 | 35.7 | 34.5 | 314.85 ± 0.25 | -7.65 | strong (worse) | 0.000 | 0.00 | CONVERGED |
| contact binary (q = 0.34, s = 0.77) | nested sampling | 194 | 13.0 | 14.83 | 1.03 | 43.6 | 43.2 | 310.42 ± 0.36 | -12.08 | strong (worse) | 0.000 | 0.00 | CONVERGED |
| convex lmax 2, τ0 0.6, c_L fit, 30 s exp. [scan] | importance sampling (ESS 1) | 194 | 12.0 | 19.77 | 1.67 | 50.5 | 38.6 | ≥ 266.15 ± 2.39 | -56.36 | inconclusive (lower bound) | – | – | – |
| convex lmax 2, τ0 1.2, c_L fit, 30 s exp. [scan] | importance sampling (ESS 1) | 194 | 12.1 | 19.77 | 1.67 | 50.5 | 38.6 | ≥ 262.90 ± 2.21 | -59.61 | inconclusive (lower bound) | – | – | – |
| convex lmax 2, τ0 2.4, c_L fit, 30 s exp. [scan] | importance sampling (ESS 1) | 194 | 12.2 | 19.77 | 1.67 | 50.5 | 38.6 | ≥ 258.64 ± 2.35 | -63.87 | inconclusive (lower bound) | – | – | – |

Not shown (Laplace step of the importance-sampling scan not converged, rerun with a larger --n-inner or fully): convex lmax 6, τ0 2.4, c_L fit, 30 s exp. [scan]; convex lmax 8, τ0 2.4, c_L fit, 30 s exp. [scan]; convex lmax 4, τ0 2.4, c_L fit, 30 s exp. [scan]; convex lmax 8, τ0 1.2, c_L fit, 30 s exp. [scan]

## Recommendation

**Most probable model: convex lmax 8, τ0 0.6, c_L fit, 30 s exp. [scan]**

- **No decision** between *convex lmax 8, τ0 0.6, c_L fit, 30 s exp. [scan]* and *convex lmax 6, τ0 0.6, c_L fit, 30 s exp. [scan]* (Δ ln Z = 0.7 ± 0.4); report both.
- Caution: the best model's evidence is from importance sampling (ESS 76); confirm it with a full nested run.
- A setting with only a lower bound on ln Z already exceeds the best by +41.5: *convex lmax 6, τ0 1.2, c_L fit, 30 s exp. [scan]*. The data support more shape freedom; run it fully (complexity_scan.py --full, or run_inversion.py --fit-tau) and compare again.
- Fit adequacy: χ²_red = 8.8 with the quoted errors (median 11.8 mmag); the model needs 32 mmag extra scatter. Either the errors are underestimated or there is signal no model here captures (shape detail, scattering, non-principal-axis rotation, calibration).
- The data constrain p_eff ≈ 16 parameters of the best model (nominal 88).

Notes: ln Z values are only meaningful as differences between runs on the same data. χ²_red uses dof = N − p_eff, where p_eff = tr(H_lik H_post⁻¹) counts the parameters the data actually constrain (shape coefficients shrunk by their prior count less than 1). Importance-sampled rows with low ESS are lower bounds (≥) and get no probability. Hybrid evidences are Laplace approximations in the profiled-offset convention of the convex/binary runs (±2 assumed).
