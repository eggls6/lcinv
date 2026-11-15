#!/usr/bin/env python3
"""Draw the home-page hero: the K03Q21N lightcurve folded at its period (real data, lcinv fit).

    python tools/make_hero.py            -> docs/assets/hero_lightcurve.svg

Points are the reduced magnitudes minus the fitted band offsets and phase term, coloured by
band; the curve is the 6-harmonic Fourier series from lcinv.period at the period search's
best period. Colours come from CSS classes so the figure follows the light/dark theme.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from lcinv import period as per          # noqa: E402
from lcinv.setup_data import prepare     # noqa: E402

D = prepare(os.path.join(ROOT, "data", "K03Q21N.lc"), "K03Q21N", location="X05",
            cache_dir=os.path.join(ROOT, "data"))
sig = np.hypot(D["err"], 0.015)
P = 3.248258                                       # period search, results/K03Q21N/summary.json
t0 = float(np.mean(D["t"]))
_, _, x, r = per.fit_fourier(D["t"], D["mag"], sig, D["band_idx"], D["alpha"], P, K=6, t0=t0)
nb = len(D["bands"])
ph = (2 * np.pi * 24.0 / P * (D["t"] - t0)) % (2 * np.pi)
y = r + per.fourier_curve(x, nb, 6, ph)            # rotational signal + residual
grid = np.linspace(0, 4 * np.pi, 721)
curve = per.fourier_curve(x, nb, 6, grid)

W, H, padx, pady = 1200, 300, 8, 26
ymin, ymax = min(y.min(), curve.min()) - 0.03, max(y.max(), curve.max()) + 0.03
sx = lambda p: padx + (W - 2 * padx) * p / (4 * np.pi)
sy = lambda m: pady + (H - 2 * pady) * (m - ymin) / (ymax - ymin)   # brighter = up (mag down)

out = [f'<svg class="hero-lc" viewBox="0 0 {W} {H}" role="img" '
       f'aria-label="Lightcurve of asteroid 2003 QN21 from Rubin g, r and i photometry, folded over two '
       f'rotations of {P:.4f} hours, with the fitted model curve">']
for k in range(1, 4):
    xx = sx(k * np.pi)
    out.append(f'<line class="hero-rot{" hero-rot-major" if k == 2 else ""}" x1="{xx:.1f}" y1="{pady - 12}" '
               f'x2="{xx:.1f}" y2="{H - pady + 12}"/>')
for rot in (0, 1):
    for b, lab in enumerate(D["bands"]):
        m = D["band_idx"] == b
        pts = " ".join(f"M{sx(p + 2 * np.pi * rot):.1f},{sy(v):.1f}h0" for p, v in zip(ph[m], y[m]))
        out.append(f'<path class="hero-pts band-{lab}" d="{pts}"/>')
d = "M" + " L".join(f"{sx(p):.1f},{sy(v):.1f}" for p, v in zip(grid, curve))
out.append(f'<path class="hero-model" pathLength="1" d="{d}"/>')
out.append("</svg>")
path = os.path.join(ROOT, "docs", "assets", "hero_lightcurve.svg")
os.makedirs(os.path.dirname(path), exist_ok=True)
open(path, "w").write("\n".join(out) + "\n")
amp = curve.max() - curve.min()
print(f"wrote {path}: {len(y)} points, bands {D['bands']}, P = {P} h, amplitude {amp:.3f} mag, "
      f"rms residual {1e3 * np.std(r):.1f} mmag")
