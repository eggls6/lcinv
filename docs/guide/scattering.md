# Scattering law

All three models use the same disk-integrated scattering law per facet,

\[
S(\mu, \mu_0) = \mu\mu_0\Big[\frac{1}{\mu + \mu_0} + c_L\Big],
\]

where \(\mu\) and \(\mu_0\) are the cosines of the emission and incidence angles: Lommel-Seeliger
plus a Lambert term with weight \(c_L\). Only facets that are both lit and visible contribute.
The default is \(c_L = 0.1\) fixed, as in Kaasalainen et al. (2001).

## Options

| Model | Fixed value | Fitted |
|---|---|---|
| global (`run_inversion.py`) | `--c-lambert` (0.1) | `--fit-cl`: \(\ln c_L\) with a log-uniform prior over `--cl-range` (0.01–100) |
| hybrid (`run_hybrid.py`) | `--c-lambert` (0.1) | `--fit-cl`: \(\ln c_L\) with a log-normal prior, `--cl-prior MEDIAN LNSD` (0.3, 2.0) |
| binary (`run_binary.py`) | from `--global-dir`, else 0.1 | `--fit-cl`, `--cl-range` |

The range 0.01 to 100 spans Lommel-Seeliger-dominated to Lambert-dominated scattering.

- **Global layer:** \(\ln c_L\) becomes a fifth nested-sampling parameter. \(S\) is linear in
  \(c_L\), \(S = S_\mathrm{LS} + c_L S_\mathrm{L}\), so fitting it costs one extra matrix per
  likelihood call and the shape optimisation is unchanged. The posterior is reported as
  `posterior.c_lambert` in `summary.json`.
- **Hybrid layer:** \(\ln c_L\) is an extra parameter, started from the global posterior median
  when the global run fitted it.

## What the data can tell

At small phase angles the data separate the two regimes, but values within one regime are nearly
degenerate. In `tests/test_cl.py`, with the geometry of 566130, a synthetic \(c_L = 0.1\) gives a
profile interval of 0.01 to 0.3, and \(c_L = 3\) gives 3 to 30. Fit \(c_L\) when you want the
evidence to include this freedom; expect a wide posterior.

For 566130, fitting \(c_L\) raises \(\ln Z\) by 4.8 over \(c_L = 0.1\) with the same exposure
treatment ([reference results](../examples/reference-results.md#566130-model-comparison)).
