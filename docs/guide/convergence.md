# Convergence

A posterior is only as good as the sampler that produced it. Both layers end with a convergence
report graded **ok**, **warn** or **fail**, with advice on what to change. The report is also
stored under `convergence` in `summary.json` and in each spin state of `summary_hybrid.json`.

## Global layer (nested sampling)

| Check | ok | warn |
|---|---|---|
| evidence left in the final live points, as dlogz | ≤ 0.5 | ≤ 1.0 |
| Kish effective sample size of the weighted posterior | ≥ 400 | ≥ 150 |
| error of \(\ln Z\) | ≤ 0.5 | ≤ 1.0 |
| error of the prograde fraction | ≤ 0.05 | ≤ 0.10 |

Anything worse is a fail. A run that stopped at `--maxcall` says so:

```text
== convergence: nested sampling: NOT CONVERGED ==
   ln Z = 729.37 +- 1.63 [fail]; evidence left in live points 100% (dlogz 11.77) [fail]; ESS 6 [fail]
   prograde fraction 0.221 +- 0.175 [fail]
   -> 100% of the evidence is still in the final live points: the run stopped early (--maxcall reached); rerun to completion or continue with --resume --extra-calls N
   -> posterior ESS is low: increase --nlive (e.g. x2) or lower --dlogz to 0.1
   -> ln Z is noisy: increase --nlive
```

The fixes, in order of cost: continue the run (`--resume --extra-calls N`), increase `--nlive`,
lower `--dlogz`. For publication-grade global results use `--dlogz 0.1` and confirm one repeat
with another `--seed`.

## Hybrid layer (NUTS, per spin state)

| Check | ok | warn |
|---|---|---|
| split R-hat of the chart coordinates \(d_1, d_2\), period, jitter | ≤ 1.01 | ≤ 1.05 |
| bulk ESS of the same parameters | ≥ 400 | ≥ 100 |
| divergences | 0 | ≤ 10 |

The chart coordinates are used rather than the ecliptic longitude, whose R-hat is inflated near
the ecliptic poles. Fixes: more chains and longer runs (`--chains 4 --warmup 1000 --samples 1500`),
a higher `--target-accept` for divergences.

## Checking finished runs: `check_convergence.py`

```bash
python check_convergence.py results/X                              # one run
python run_inversion.py ... --seed 7 --outdir results/X_seed7      # a repeat
python check_convergence.py results/X results/X_seed7              # + reproducibility
python check_convergence.py results/X_hybrid                       # NUTS spin states
```

Global runs are read from `nested_results.pkl` and `summary.json`, hybrid runs from
`posterior_spin_state_*.npz`. Repeat runs agree when \(\ln Z\) differs by less than about 2
combined sigma and the period, jitter and prograde fraction overlap within their errors.

## Resolution checks

The convex model's 400 facets are a discretisation. `reweight_resolution.py results/X --nfacets 800`
importance-reweights the posterior with a finer facet set and reports the ESS and how far the
spin-state centres move. For K03Q21N: ESS 80 % at 800 facets, centres moved by less than 1.3°.
