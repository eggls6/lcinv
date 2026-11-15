# `make_viewer`

*Self-contained interactive 3D shape viewer (HTML).*

This command reads its arguments directly (no option parser).

```text
Build a self-contained interactive 3D viewer (HTML + inlined three.js r128, works offline).

  python make_viewer.py results/K03Q21N            -> results/K03Q21N/shape_viewer.html

Reads shapes/*.obj, summary.json and nested_results.pkl. For display, every shape is
rotated about its spin axis so that its longest equatorial axis lies along x (the
rotational phase is a gauge freedom, so posterior draws are otherwise arbitrarily rotated
relative to each other). The OBJ files keep the physical orientation.
```
