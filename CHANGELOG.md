# Changelog

## 0.2.0

### Added

- `--search-window W` in `run_inversion.py`: search the period only within
  `--period x (1 +- W)`. Needed for data spanning several apparitions, whose full grid would
  have millions of trial periods.
- Automatic zooming of the fine period scan (`lcinv.period.refine_grid`,
  `refine_period_auto`): the scan resolution follows the number of rotations in the data.
  Results for data of a few nights are unchanged.
- Search-range warnings (`lcinv.period.range_warnings`): the period search warns when the
  periodogram has a deeper minimum below `--pmin` or the Fourier fit at the adopted period is
  poor (chi2/nu > 3). Printed as `WARNING (period search): ...` and stored in `summary.json`
  under `period_search.warnings`.
- Documentation site (MkDocs Material) in `docs/`, with tested examples, the command-line and
  Python API reference and the validation results; published by `.github/workflows/docs.yml`.
- `pyproject.toml` (`pip install -e .`), pytest support (`python -m pytest tests`), a test of
  the period search (`tests/test_period.py`) and a CI workflow running the tests.

### Fixed

- `--resume` failed on a run that had ended normally or at `--maxcall` ("The remaining live
  points have already been added to the list of samples!"). `finalize_from_checkpoint` now
  removes the final live points from such a checkpoint before continuing.
- `run_binary.py --help` printed an empty description.
- `make_hybrid_report.py` defaulted to the K03Q21N data when a folder had no
  `summary_hybrid.json`; it now requires `--lc` and `--designation` in that case.

### Changed

- Docstrings of the whole package in NumPy style, matching the documentation; help texts for
  every command-line option; all scripts print their full description with `--help`.
- `README.md` shortened; the former README sections and `HYBRID.md` are now in the
  documentation site.
- Unused imports and dead code removed.

## 0.1.2

- `--sample rslice` (random slice sampling) became the default proposal of the global layer.

## 0.1.1

- JAX GPU memory preallocation is off by default; `--device`, `--gpu-mem-fraction`,
  `--gpu-preallocate`, `--worker-device` and `--worker-threads` added.
