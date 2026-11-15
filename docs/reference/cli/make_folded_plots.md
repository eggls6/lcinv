# `make_folded_plots`

*Phase-folded lightcurve figures for finished runs.*

This command reads its arguments directly (no option parser).

```text
Phase-folded (0-4 pi) lightcurve figures for existing results folders, without resampling.

  python make_folded_plots.py results/566130_X05_cl              -> fig_folded_lightcurve.png
  python make_folded_plots.py results/K03Q21N_hybrid_quat        -> fig_folded_lightcurve_hybrid.png
                                                                     (via make_hybrid_report.py)
New runs of run_inversion.py / run_hybrid.py make these figures themselves.
```
