# Command-line reference

Every entry point with all of its options and defaults. The pages are generated from the code (`python tools/gen_cli_docs.py`), so they match the installed version.

| Command | Purpose |
|---|---|
| [`run_inversion.py`](run_inversion.md) | Global layer: convex shape, nested sampling over the spin state |
| [`run_hybrid.py`](run_hybrid.md) | Local layer: radius-field shape and spin, NUTS within each spin state |
| [`run_binary.py`](run_binary.md) | Contact-binary model (two ellipsoidal lobes), nested sampling |
| [`batch_shapes.py`](batch_shapes.md) | Whole pipeline for every row of a target table |
| [`check_convergence.py`](check_convergence.md) | Convergence report for finished runs, agreement between repeats |
| [`compare_models.py`](compare_models.md) | Evidence-based comparison of runs on the same data |
| [`complexity_scan.py`](complexity_scan.md) | Evidence versus shape resolution (lmax) and prior scale (tau0) |
| [`export_obj.py`](export_obj.md) | Watertight .obj meshes of the posterior shapes |
| [`make_folded_plots.py`](make_folded_plots.md) | Phase-folded lightcurve figures for finished runs |
| [`make_hybrid_report.py`](make_hybrid_report.md) | Rebuild all hybrid figures and viewers without resampling |
| [`make_viewer.py`](make_viewer.md) | Self-contained interactive 3D shape viewer (HTML) |
| [`make_uncertainty_viewer.py`](make_uncertainty_viewer.md) | 3D viewer of the shape uncertainty (radius spread) |
| [`reweight_resolution.py`](reweight_resolution.md) | Facet-resolution check by importance reweighting |
| [`par_to_targets.py`](par_to_targets.md) | Convert a .par target list into a batch table |
| [`python -m lcinv.mpc`](lcinv_mpc.md) | Download ADES observations from the MPC |
| [`python -m lcinv.size`](lcinv_size.md) | Diameter from H (CNEOS size estimator) |
| [`python -m lcinv.hybrid.naming`](lcinv_hybrid_naming.md) | Rename legacy hybrid results ('mode' to 'spin state') |
