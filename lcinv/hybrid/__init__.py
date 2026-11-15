"""Hybrid Bayesian lightcurve inversion.

Global layer : nested sampling over the spin state with a Laplace-marginalised convex
               shape (lcinv.sampler / run_inversion.py) -> discovers spin modes and their
               mass, robust to multimodality and period aliases.
Local layer  : a star-shaped log-radius field with a Matern-like (GP-like) truncated
               spherical-harmonic prior, Lommel-Seeliger + Lambert scattering, an explicit
               principal-axis (inertia) prior, sampled jointly with the spin state by NUTS
               within each mode -> exact joint shape/spin uncertainties within a mode.
Combination  : per-mode posteriors are mixed with weights from a Laplace evidence of the
               local model (consistent priors), reported next to the global-layer weights.

Modules
  quadrature : Gauss-Legendre x uniform grid, exactly orthonormal real SH, surface mesh
  shape      : radius field, non-centred prior, volume/inertia integrals, priors
  model      : local spin coordinates, likelihood with marginalised offsets and jitter
  modes      : mode discovery from global samples, multi-start MAP, Laplace evidence
  inference  : NumPyro NUTS per mode, ArviZ InferenceData with pointwise log-likelihood
  checks     : convexity and ray-traced self-shadowing checks, prior sensitivity
  report     : figures (with fallbacks), HTML viewers, summary
"""
