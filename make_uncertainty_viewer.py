#!/usr/bin/env python
"""Mean convex shape coloured by posterior radius uncertainty (self-contained HTML, three.js).

  python make_uncertainty_viewer.py results/K03Q21N [--n 150]
      -> results/K03Q21N/shape_uncertainty_viewer.html

Steps
  1. Reconstruct the convex polyhedron of every posterior shape draw (spin-state posterior
     sample + Laplace draw of the shape coefficients) by solving the Minkowski problem.
  2. Register the draws: the rotational phase is a gauge freedom, so each body may be rotated
     arbitrarily about its spin axis (z). Radii on a latitude-longitude grid make a rotation
     about z a circular shift in longitude; each draw is rotated to best match the reference
     (first the max-likelihood shape, then the ensemble mean; two passes).
  3. Radius r(u) from the centroid along ~2500 directions u for every registered draw:
     mean, standard deviation and 16/50/84th percentiles per direction.
  4. Render the mean-radius surface coloured by sigma_r (viridis), with optional +-1 sigma
     envelopes, the spin axis and the maximum-inertia axis of the mean shape.

Radii are in units of the volume-equivalent sphere radius R_eq (all draws have volume 1).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcinv import minkowski as mk, sampler  # noqa: E402
from lcinv.convex import fibonacci_sphere  # noqa: E402
from make_viewer import principal_axes  # noqa: E402
from lcinv.plots import align_equatorial  # noqa: E402

R_EQ = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)  # radius of the unit-volume sphere


def radii(hull_eq, dirs):
    """Distance from the origin to the convex surface {x : n.x + d <= 0} along unit dirs."""
    n, d = hull_eq[:, :3], -hull_eq[:, 3]
    cosang = dirs @ n.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cosang > 1e-12, d[None, :] / cosang, np.inf)
    return t.min(axis=1)


def latlon_dirs(nlat=90, nlon=180):
    """Unit vectors of a latitude-longitude grid and their area weights cos(lat)."""
    lat = np.radians(np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat))
    lon = np.radians(np.arange(nlon) * 360.0 / nlon)
    L, P = np.meshgrid(lat, lon, indexing="ij")
    return np.stack([np.cos(L) * np.cos(P), np.cos(L) * np.sin(P), np.sin(L)], -1).reshape(-1, 3), \
        np.cos(L)


def rot_z(a):
    """Rotation matrix about z by angle `a` [rad]."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def best_shift(grid, ref, w):
    """Longitude shift (in grid cells) of `grid` that best matches `ref` (area-weighted)."""
    nlon = grid.shape[1]
    cost = [np.sum(w * (np.roll(grid, -k, axis=1) - ref) ** 2) for k in range(nlon)]
    return int(np.argmin(cost))



def _is_hybrid(outdir):
    import glob as _g
    return (not os.path.exists(os.path.join(outdir, "nested_results.pkl"))
            and bool(_g.glob(os.path.join(outdir, "posterior_spin_state_*.npz"))
                     or _g.glob(os.path.join(outdir, "posterior_mode*.npz"))))


def _hybrid_viewers(outdir, extra=()):
    """Hybrid results have no nested_results.pkl: build the per-spin-state viewers from the NUTS
    posteriors instead (same as make_hybrid_report.py)."""
    import sys as _s
    import make_hybrid_report
    print(f"{outdir} is a hybrid run: writing viewer_spin_state_*.html from the NUTS posteriors")
    _s.argv = ["make_hybrid_report.py", outdir, *extra]
    make_hybrid_report.main()


def main():
    """Command line of make_uncertainty_viewer.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir",
                   help="finished run_inversion.py folder")
    p.add_argument("--n", type=int, default=150, help="number of posterior shape draws")
    p.add_argument("--ndir", type=int, default=2562, help="surface directions for the mesh")
    a, extra = p.parse_known_args()
    if _is_hybrid(a.outdir):
        return _hybrid_viewers(a.outdir, extra)

    R = sampler.load(os.path.join(a.outdir, "nested_results.pkl"))
    summ = json.load(open(os.path.join(a.outdir, "summary.json")))
    ens = R["ensemble"]
    nf = R["args"]["nfacets"]
    lmax = R["args"]["lmax"]
    normals = fibonacci_sphere(nf)
    from lcinv.convex import real_sph_harm_basis
    Y, _ = real_sph_harm_basis(normals, lmax)
    w0 = 4 * np.pi / nf

    # 1. reconstruct ------------------------------------------------------------------
    hulls, thetas = [], []
    n = min(a.n, len(ens["c_draw"]))
    for k in range(n):
        A = w0 * np.exp(Y @ ens["c_draw"][k])
        v, _ = mk.reconstruct(normals, A)
        v = v - v.mean(axis=0)
        hulls.append(ConvexHull(v))
        thetas.append(ens["theta"][k])
        if (k + 1) % 25 == 0:
            print(f"reconstructed {k + 1}/{n}")
    # max-likelihood reference shape
    ml = np.array([l.split()[1:4] for l in open(os.path.join(a.outdir, "shapes", "shape_maxlike.obj"))
                   if l.startswith("v ")], float)
    ml -= ml.mean(axis=0)
    ml = align_equatorial(ml)  # reference: longest equatorial axis along x
    ref_hull = ConvexHull(ml)

    # 2. register about z -------------------------------------------------------------
    gdirs, coslat = latlon_dirs()
    nlat, nlon = coslat.shape
    grids = np.array([radii(h.equations, gdirs).reshape(nlat, nlon) for h in hulls])
    ref = radii(ref_hull.equations, gdirs).reshape(nlat, nlon)
    shifts = np.zeros(n, int)
    for _pass in range(2):
        shifts = np.array([best_shift(g, ref, coslat) for g in grids])
        aligned = np.array([np.roll(g, -s, axis=1) for g, s in zip(grids, shifts)])
        ref = aligned.mean(axis=0)
    angles = 2 * np.pi * shifts / nlon  # rotate body by -angle about z

    # 3. radii along mesh directions ---------------------------------------------------
    dirs = fibonacci_sphere(a.ndir)
    tri = ConvexHull(dirs).simplices
    # orient triangles outward
    for i, t in enumerate(tri):
        pnt = dirs[t]
        if np.dot(np.cross(pnt[1] - pnt[0], pnt[2] - pnt[0]), pnt.mean(0)) < 0:
            tri[i] = t[::-1]
    # r_k(u) of the rotated body = r_k(Rz(+angle) u)
    rr = np.array([radii(h.equations, dirs @ rot_z(ang).T) for h, ang in zip(hulls, angles)])
    rr /= R_EQ
    r_mean = rr.mean(axis=0)
    r_std = rr.std(axis=0)
    q16, q50, q84 = np.percentile(rr, [16, 50, 84], axis=0)
    mean_verts = dirs * r_mean[:, None]
    mom, iax = principal_axes(mean_verts, tri)
    iang = float(np.degrees(np.arccos(min(1.0, abs(iax[2])))))
    ext = mean_verts.max(0) - mean_verts.min(0)
    th = np.array(thetas)
    print(f"sigma_r: median {np.median(r_std) * 100:.1f}% of R_eq, range {r_std.min() * 100:.1f}-"
          f"{r_std.max() * 100:.1f}%; mean-shape extents {np.round(ext / ext[2], 3)}; "
          f"spin vs max-inertia axis {iang:.0f} deg")

    data = dict(
        dirs=np.round(dirs, 5).ravel().tolist(), tri=tri.ravel().tolist(),
        mean=np.round(r_mean, 5).tolist(), std=np.round(r_std, 5).tolist(),
        q16=np.round(q16, 5).tolist(), q84=np.round(q84, 5).tolist(),
        iax=np.round(iax, 4).tolist(), iang=round(iang, 1), mom=np.round(mom, 3).tolist(),
        ext=np.round(ext / ext[2], 3).tolist(), ndraw=n,
        fpro=float(np.mean(th[:, 1] > 0)), P=summ["posterior"]["P_h"],
        designation=summ["horizons_target"], npts=summ["n_points"], nnights=summ["n_nights"],
        bands=summ["bands"])
    vend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcinv", "vendor")
    html = TEMPLATE.replace("__THREEJS__", open(os.path.join(vend, "three.min.js")).read()) \
                   .replace("__ORBIT__", open(os.path.join(vend, "OrbitControls.js")).read()) \
                   .replace("__DATA__", json.dumps(data, separators=(",", ":"))) \
                   .replace("__TITLE__", f"{summ['horizons_target']} radius uncertainty")
    out = os.path.join(a.outdir, "shape_uncertainty_viewer.html")
    open(out, "w").write(html)
    np.savez(os.path.join(a.outdir, "radius_uncertainty.npz"), dirs=dirs, tri=tri, r=rr,
             mean=r_mean, std=r_std, q16=q16, q50=q50, q84=q84, angles=angles, theta=th)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f2ee;--panel:#ffffff;--ink:#1f2430;--mute:#6b7080;--line:#dedad2;--canvas:#e9e6e0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171c;--panel:#1d2027;--ink:#e8e6e1;--mute:#9aa0ad;--line:#2d313a;--canvas:#101216}}
:root[data-theme="dark"]{--bg:#15171c;--panel:#1d2027;--ink:#e8e6e1;--mute:#9aa0ad;--line:#2d313a;--canvas:#101216}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{display:grid;grid-template-columns:1fr 320px;height:100vh}
#view{position:relative;background:var(--canvas);min-height:360px}
#view canvas{display:block;width:100%;height:100%}
.hint{position:absolute;left:14px;bottom:12px;color:var(--mute);font-size:12px;pointer-events:none}
.legend{position:absolute;right:16px;top:16px;bottom:56px;width:64px;display:flex;flex-direction:column;align-items:flex-start;pointer-events:none}
.legend .lab{font-size:11px;color:var(--mute);margin-bottom:6px;width:120px;margin-left:-56px;text-align:right}
.legend .barwrap{flex:1;display:flex;gap:6px}
.legend .bar{width:14px;border-radius:3px;border:1px solid var(--line)}
.legend .ticks{display:flex;flex-direction:column;justify-content:space-between;font-size:11px;font-variant-numeric:tabular-nums}
aside{background:var(--panel);border-left:1px solid var(--line);padding:18px 18px 24px;overflow:auto}
h1{font-size:17px;margin:0 0 2px}
.sub{color:var(--mute);font-size:12px;margin-bottom:16px}
h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:18px 0 8px;font-weight:600}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0}
.row label{display:flex;align-items:center;gap:8px;cursor:pointer}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin-bottom:6px}
.seg button{flex:1;border:0;background:transparent;color:var(--ink);padding:6px 4px;font:inherit;font-size:12px;cursor:pointer}
.seg button.on{background:var(--ink);color:var(--panel)}
button.play{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:6px 12px;font:inherit;cursor:pointer}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0;font-size:13px}
dt{color:var(--mute)}dd{margin:0;font-variant-numeric:tabular-nums}
.note{color:var(--mute);font-size:12px;margin-top:14px}
.warn{color:#c4452f}
@media (max-width:760px){.wrap{grid-template-columns:1fr;grid-template-rows:62vh auto;height:auto}aside{border-left:0;border-top:1px solid var(--line);padding:16px}.legend{top:12px;bottom:44px}}
</style>
</head>
<body>
<div class="wrap">
  <div id="view">
    <div class="legend"><div class="lab" id="leglab"></div><div class="barwrap"><div class="bar" id="bar"></div><div class="ticks" id="ticks"></div></div></div>
    <div class="hint">drag to rotate · scroll / pinch to zoom · right-drag to pan · hover for values</div>
  </div>
  <aside>
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>

    <h2>Colour</h2>
    <div class="seg" id="cmode">
      <button data-m="std" class="on">σ<sub>r</sub> absolute</button>
      <button data-m="rel">σ<sub>r</sub> / r</button>
      <button data-m="iqr">68% width</button>
    </div>

    <h2>Display</h2>
    <div class="row"><label><input type="checkbox" id="env"> ±1σ envelopes (16th / 84th pct)</label></div>
    <div class="row"><label><input type="checkbox" id="axes" checked> Spin axis &amp; max-inertia axis</label></div>
    <div class="row"><label><input type="checkbox" id="edges"> Mesh edges</label></div>
    <div class="row"><button class="play" id="play">Pause rotation</button><span style="color:var(--mute);font-size:12px">1 rotation / 10 s</span></div>

    <h2>Point under cursor</h2>
    <dl id="hover"><dt>—</dt><dd>hover the surface</dd></dl>

    <h2>Mean shape</h2>
    <dl id="info"></dl>

    <div class="note">
      Mean surface of <span id="nd"></span> posterior shape draws (spin-state posterior samples, each with a
      Laplace draw of the shape coefficients). Every draw is rotated about the spin axis (vertical, z) to
      best match the ensemble mean before radii are compared, because the rotational phase is a gauge
      freedom. Radii are measured from the centroid in units of the volume-equivalent sphere radius
      R<sub>eq</sub>.<br><br>
      <span class="warn">Caveat:</span> these solutions spin far from their maximum-inertia axis
      (dashed); see the inversion notes before interpreting the shape physically.
    </div>
  </aside>
</div>
<script>/* three.js r128, MIT licence (c) 2010-2021 three.js authors */
__THREEJS__</script>
<script>__ORBIT__</script>
<script>
// null-safe lookup: optional elements may be absent in derived viewers
const $id = id => document.getElementById(id) || document.createElement('span');
const D = __DATA__;
const fmt = (x, d) => Number(x).toFixed(d);
const N = D.mean.length;
$id('title').textContent = D.designation;
$id('subtitle').textContent =
  `Posterior radius uncertainty · ${D.npts} Rubin points (${D.bands.join('/')}), ${D.nnights} nights`;
$id('nd').textContent = D.ndraw;
$id('info').innerHTML = [
  ['Draws', D.ndraw],
  ['Prograde draws', fmt(100 * D.fpro, 0) + ' %'],
  ['Period', `${fmt(D.P[1], 5)} h`],
  ['Extents x:y:z', D.ext.map(x => fmt(x, 2)).join(' : ')],
  ['Principal moments', D.mom.map(x => fmt(x, 2)).join(' : ')],
  ['Spin vs max-inertia axis', `<span class="${D.iang > 20 ? 'warn' : ''}">${fmt(D.iang, 0)}°</span>`],
  ['Median σ_r', fmt(100 * median(D.std), 1) + ' % R_eq'],
].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
function median(a) { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; }

// viridis
const VIR = ['#440154','#482878','#3e4989','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'];
const VIRC = VIR.map(h => new THREE.Color(h));
function viridis(t) {
  t = Math.min(1, Math.max(0, t)) * (VIRC.length - 1);
  const i = Math.min(VIRC.length - 2, Math.floor(t)), f = t - i;
  return VIRC[i].clone().lerp(VIRC[i + 1], f);
}
$id('bar').style.background = `linear-gradient(to top, ${VIR.join(',')})`;

// scalar fields
const FIELDS = {
  std: {v: D.std.map(x => 100 * x), lab: 'σ<sub>r</sub> [% R<sub>eq</sub>]', d: 1},
  rel: {v: D.std.map((x, i) => 100 * x / D.mean[i]), lab: 'σ<sub>r</sub> / r [%]', d: 1},
  iqr: {v: D.q84.map((x, i) => 100 * (x - D.q16[i])), lab: '84th − 16th pct [% R<sub>eq</sub>]', d: 1},
};

// ---------------------------------------------------------------- scene
const view = $id('view');
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
camera.position.set(4.2, 2.0, 4.2);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x555560, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.75);
sun.position.set(3, 2.5, 2);
scene.add(sun);

const spin = new THREE.Group();
const body = new THREE.Group();
body.rotation.x = -Math.PI / 2;  // body z -> world up
spin.add(body);
scene.add(spin);

function surface(radius) {
  const pos = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) pos[3 * i + k] = D.dirs[3 * i + k] * radius[i];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(D.tri);
  g.computeVertexNormals();
  return g;
}
const gMean = surface(D.mean);
gMean.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3 * N), 3));
const mesh = new THREE.Mesh(gMean, new THREE.MeshLambertMaterial({vertexColors: true}));
body.add(mesh);
const edges = new THREE.LineSegments(new THREE.WireframeGeometry(gMean),
  new THREE.LineBasicMaterial({color: 0x000000, transparent: true, opacity: 0.12}));
edges.visible = false;
body.add(edges);

const envGroup = new THREE.Group();
envGroup.add(new THREE.Mesh(surface(D.q84), new THREE.MeshBasicMaterial({color: 0x8a8f99, wireframe: true, transparent: true, opacity: 0.18})));
envGroup.add(new THREE.Mesh(surface(D.q16), new THREE.MeshBasicMaterial({color: 0x1f2430, wireframe: true, transparent: true, opacity: 0.18})));
envGroup.visible = false;
body.add(envGroup);

const axes = new THREE.Group();
const rmax = Math.max(...D.mean) * 1.35;
axes.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), rmax, 0x1f2430, 0.12, 0.07));
axes.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 0), rmax, 0x1f2430, 0.12, 0.07));
const ia = D.iax;
const iline = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-ia[0] * rmax, -ia[1] * rmax, -ia[2] * rmax), new THREE.Vector3(ia[0] * rmax, ia[1] * rmax, ia[2] * rmax)]),
  new THREE.LineDashedMaterial({color: 0xd99a1e, dashSize: 0.08, gapSize: 0.05}));
iline.computeLineDistances();
axes.add(iline);
body.add(axes);

// ---------------------------------------------------------------- colouring
let field = 'std';
function recolour() {
  const F = FIELDS[field], v = F.v;
  const lo = Math.min(...v), hi = Math.max(...v);
  const col = gMean.attributes.color;
  for (let i = 0; i < N; i++) { const c = viridis((v[i] - lo) / (hi - lo)); col.setXYZ(i, c.r, c.g, c.b); }
  col.needsUpdate = true;
  $id('leglab').innerHTML = F.lab;
  const ticks = [];
  for (let k = 4; k >= 0; k--) ticks.push(`<span>${fmt(lo + (hi - lo) * k / 4, F.d)}</span>`);
  $id('ticks').innerHTML = ticks.join('');
}
document.querySelectorAll('#cmode button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#cmode button').forEach(x => x.classList.toggle('on', x === b));
  field = b.dataset.m; recolour();
}));
$id('env').addEventListener('change', e => envGroup.visible = e.target.checked);
$id('axes').addEventListener('change', e => axes.visible = e.target.checked);
$id('edges').addEventListener('change', e => edges.visible = e.target.checked);
let playing = true;
const playBtn = $id('play');
playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause rotation' : 'Play rotation'; });

// ---------------------------------------------------------------- hover readout
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener('pointermove', ev => {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObject(mesh)[0];
  if (!hit) return;
  const f = hit.face, idx = [f.a, f.b, f.c];
  const local = mesh.worldToLocal(hit.point.clone());
  let best = idx[0], bd = 1e9;
  idx.forEach(j => { const d = local.distanceToSquared(new THREE.Vector3(D.dirs[3*j]*D.mean[j], D.dirs[3*j+1]*D.mean[j], D.dirs[3*j+2]*D.mean[j])); if (d < bd) { bd = d; best = j; } });
  const u = [D.dirs[3*best], D.dirs[3*best+1], D.dirs[3*best+2]];
  const lat = Math.asin(u[2]) * 180 / Math.PI, lon = ((Math.atan2(u[1], u[0]) * 180 / Math.PI) + 360) % 360;
  $id('hover').innerHTML = [
    ['Body lat, lon', `${fmt(lat, 0)}°, ${fmt(lon, 0)}°`],
    ['Mean r', `${fmt(D.mean[best], 3)} R_eq`],
    ['σ_r', `${fmt(100 * D.std[best], 1)} % R_eq (${fmt(100 * D.std[best] / D.mean[best], 1)} %)`],
    ['16–84th pct', `${fmt(D.q16[best], 3)} – ${fmt(D.q84[best], 3)}`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
});

function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize);
let last = performance.now();
function tick(now) {
  const dt = (now - last) / 1000; last = now;
  if (playing) spin.rotation.y += 2 * Math.PI * dt / 10;
  controls.update(); renderer.render(scene, camera); requestAnimationFrame(tick);
}
recolour(); resize(); requestAnimationFrame(tick);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
