#!/usr/bin/env python
"""Build a self-contained interactive 3D viewer (HTML + inlined three.js r128, works offline).

  python make_viewer.py results/K03Q21N            -> results/K03Q21N/shape_viewer.html

Reads shapes/*.obj, summary.json and nested_results.pkl. For display, every shape is
rotated about its spin axis so that its longest equatorial axis lies along x (the
rotational phase is a gauge freedom, so posterior draws are otherwise arbitrarily rotated
relative to each other). The OBJ files keep the physical orientation.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lcinv import sampler  # noqa: E402
from lcinv.hybrid import naming  # noqa: E402
from lcinv.plots import align_equatorial  # noqa: E402


def read_obj(path):
    """Vertices, zero-based faces and header comment lines of an OBJ file."""
    v, f, head = [], [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        elif line.startswith("#"):
            head.append(line[1:].strip())
    return np.array(v), np.array(f, int), " ".join(head)


def principal_axes(v, f):
    """Uniform-density inertia tensor of a closed triangulated polyhedron.
    Returns (moments normalised to max, unit vector of the maximum-inertia axis)."""
    c = v.mean(axis=0)
    I = np.zeros((3, 3))
    for a, b, d in f:
        p = np.array([v[a], v[b], v[d]]) - c
        vol = np.linalg.det(p) / 6.0
        s = p.sum(axis=0)
        C = vol / 20.0 * (np.outer(s, s) + p.T @ p)
        I += np.trace(C) * np.eye(3) - C
    w, V = np.linalg.eigh(I)
    ax = V[:, -1] * np.sign(V[2, -1] or 1.0)
    return w / w.max(), ax


def parse_spin(head):
    """Pole and period from an OBJ header written by lcinv (lambda=..., beta=..., P=...)."""
    g = lambda k: float(re.search(k + r"=([-\d.]+)", head).group(1))
    return dict(lam=g("lambda") % 360.0, bet=g("beta"), P=g("P"))



def _is_hybrid(outdir):
    return (not os.path.exists(os.path.join(outdir, "nested_results.pkl"))
            and naming.has_hybrid(outdir))


def _hybrid_viewers(outdir, extra=()):
    """Hybrid results have no nested_results.pkl: build the per-spin-state viewers from the NUTS
    posteriors instead (same as make_hybrid_report.py)."""
    import sys as _s
    import make_hybrid_report
    print(f"{outdir} is a hybrid run: writing viewer_spin_state_*.html from the NUTS posteriors")
    _s.argv = ["make_hybrid_report.py", outdir, *extra]
    make_hybrid_report.main()


def _q3(x):
    return [float(v) for v in np.percentile(x, [16, 50, 84])]


def _shape_rec(name, v, f, spin):
    v = v - v.mean(axis=0)
    ext = v.max(0) - v.min(0)
    mom, iax = principal_axes(v, f)
    return dict(name=name, v=np.round(v, 4).ravel().tolist(), f=f.ravel().tolist(),
                ext=np.round(ext, 4).tolist(), mom=np.round(mom, 3).tolist(), iax=np.round(iax, 4).tolist(),
                iang=round(float(np.degrees(np.arccos(min(1.0, abs(iax[2]))))), 1),
                **{x: round(float(spin[x]), 6 if x == "P" else 2) for x in spin})


def _write(outdir, name, shapes, meta, title):
    vend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcinv", "vendor")
    html = TEMPLATE.replace("__THREEJS__", open(os.path.join(vend, "three.min.js")).read()) \
                   .replace("__ORBIT__", open(os.path.join(vend, "OrbitControls.js")).read()) \
                   .replace("__SHAPES__", json.dumps(shapes, separators=(",", ":"))) \
                   .replace("__META__", json.dumps(meta)) \
                   .replace("__TITLE__", title)
    out = os.path.join(outdir, name)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB, {len(shapes)} shapes)")


def hybrid_shape_viewers(outdir, ndraw=30, seed=0):
    """shape_viewer_spin_state_N.html for a hybrid run: posterior-mean radius field as the main
    shape, NUTS draws as the transparent ensemble overlay. All draws share the gauge-fixed
    body frame, so one display rotation (from the mean shape) is applied to all of them."""
    from lcinv.hybrid.quadrature import make_grid
    st = json.load(open(os.path.join(outdir, "summary_hybrid.json")))
    s = st["settings"]
    grid = make_grid(s["lmax"], nlat=s["nlat"], nlon=2 * s["nlat"])
    f = np.asarray(grid.tri, int)
    gdir = s.get("global_dir")
    gs = json.load(open(os.path.join(gdir, "summary.json"))) if gdir and os.path.exists(
        os.path.join(gdir, "summary.json")) else {}
    rng = np.random.default_rng(seed)
    for mode, fp in naming.posterior_files(outdir):
        z = np.load(fp)
        r = z["r_mesh"]
        if r.shape[1] != len(grid.u_m):
            raise ValueError(f"{fp}: radius mesh has {r.shape[1]} vertices, grid {len(grid.u_m)}")
        mean_v = grid.u_m * r.mean(0)[:, None]
        xy = mean_v[:, :2] - mean_v[:, :2].mean(0)
        _, V = np.linalg.eigh(xy.T @ xy)
        ang = -np.arctan2(V[1, -1], V[0, -1])
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        lam_c = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(z["lam"])))))) % 360
        shapes = [_shape_rec("Posterior mean", mean_v @ R.T, f,
                             dict(lam=lam_c, bet=np.median(z["bet"]), P=np.median(z["P"])))]
        idx = np.sort(rng.choice(len(r), size=min(ndraw, len(r)), replace=False))
        exts = []
        for k, i in enumerate(idx):
            v = (grid.u_m * r[i][:, None]) @ R.T
            rec = _shape_rec(f"Posterior draw {k + 1}", v, f, dict(lam=z["lam"][i], bet=z["bet"][i], P=z["P"][i]))
            exts.append(rec["ext"])
            shapes.append(rec)
        e = np.array([np.sort(x)[::-1] for x in
                      (z["extents"] if "extents" in z.files else np.array(exts))])
        mode_info = naming.state_info(st, mode)
        meta = dict(designation=gs.get("horizons_target", s.get("designation", "")),
                    npts=int(z["loglik"].shape[1]), nnights=gs.get("n_nights", 0),
                    bands=gs.get("bands", []), alpha=gs.get("phase_angle_range", [0, 0]),
                    P=_q3(z["P"]), jit=_q3(z["jitter"]), fpro=float(np.mean(z["bet"] > 0)),
                    eab=_q3(e[:, 0] / e[:, 1]), ebc=_q3(e[:, 1] / e[:, 2]),
                    kind=f"Hybrid star-shaped model, spin state {mode}"
                         + (f" (weight {mode_info['weight_local_laplace']:.2f})"
                            if "weight_local_laplace" in mode_info else ""))
        if "c_L" in z.files:
            meta["cl"] = _q3(z["c_L"])
        _write(outdir, f"shape_viewer_{naming.tag(mode)}.html", shapes, meta,
               f"{meta['designation']} hybrid spin state {mode} shape model")


def _is_binary(outdir):
    sp = os.path.join(outdir, "summary.json")
    return os.path.exists(sp) and json.load(open(sp)).get("model") == "contact binary"


def main():
    """Command line: python make_viewer.py OUTDIR (global, binary or hybrid folder)."""
    outdir = sys.argv[1]
    if _is_binary(outdir):
        print(f"{outdir} is a contact-binary run: rebuilding its products with run_binary.py --replot")
        import subprocess
        return subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "run_binary.py"), "--replot", "--outdir", outdir])
    if _is_hybrid(outdir):
        hybrid_shape_viewers(outdir)
        return _hybrid_viewers(outdir, sys.argv[2:])
    summ = json.load(open(os.path.join(outdir, "summary.json")))
    ens = sampler.load(os.path.join(outdir, "nested_results.pkl"))["ensemble"]
    shapes = []
    files = [os.path.join(outdir, "shapes", "shape_maxlike.obj")] + sorted(
        glob.glob(os.path.join(outdir, "shapes", "shape_post_*.obj")))
    for k, fp in enumerate(files):
        v, f, head = read_obj(fp)
        v = align_equatorial(v)
        v -= v.mean(axis=0)
        ext = v.max(0) - v.min(0)
        spin = parse_spin(head)
        mom, iax = principal_axes(v, f)
        rec = dict(name="Max-likelihood" if k == 0 else f"Posterior draw {k}",
                   v=np.round(v, 4).ravel().tolist(), f=f.ravel().tolist(),
                   ext=np.round(ext, 4).tolist(), mom=np.round(mom, 3).tolist(),
                   iax=np.round(iax, 4).tolist(),
                   iang=round(float(np.degrees(np.arccos(min(1.0, abs(iax[2]))))), 1), **{x: round(spin[x], 6 if x == "P" else 2) for x in spin})
        if k > 0:
            j = k - 1
            rec["ab"] = round(float(ens["ab"][j]), 3)
            rec["bc"] = round(float(ens["bc"][j]), 3)
        shapes.append(rec)

    post = summ["posterior"]
    meta = dict(
        designation=summ["horizons_target"],
        npts=summ["n_points"], nnights=summ["n_nights"], bands=summ["bands"],
        P=post["P_h"], jit=post["jitter_mag"], fpro=post["frac_prograde"],
        ab=post["ab"], bc=post["bc"], eab=post["extent_ratios_a_b"], ebc=post["extent_ratios_b_c"],
        alpha=summ["phase_angle_range"])
    if post.get("c_lambert"):
        meta["cl"] = post["c_lambert"]
    vend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcinv", "vendor")
    three = open(os.path.join(vend, "three.min.js")).read()
    orbit = open(os.path.join(vend, "OrbitControls.js")).read()
    html = TEMPLATE.replace("__THREEJS__", three).replace("__ORBIT__", orbit) \
                   .replace("__SHAPES__", json.dumps(shapes, separators=(",", ":"))) \
                   .replace("__META__", json.dumps(meta)) \
                   .replace("__TITLE__", f"{summ['horizons_target']} shape model")
    out = os.path.join(outdir, "shape_viewer.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB, {len(shapes)} shapes)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f2ee;--panel:#ffffff;--ink:#1f2430;--mute:#6b7080;--line:#dedad2;--acc:#c4452f;--canvas:#e9e6e0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171c;--panel:#1d2027;--ink:#e8e6e1;--mute:#9aa0ad;--line:#2d313a;--acc:#e0644d;--canvas:#101216}}
:root[data-theme="dark"]{--bg:#15171c;--panel:#1d2027;--ink:#e8e6e1;--mute:#9aa0ad;--line:#2d313a;--acc:#e0644d;--canvas:#101216}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{display:grid;grid-template-columns:1fr 320px;height:100vh}
#view{position:relative;background:var(--canvas);min-height:360px}
#view canvas{display:block;width:100%;height:100%}
.hint{position:absolute;left:14px;bottom:12px;color:var(--mute);font-size:12px;pointer-events:none}
.badge{position:absolute;left:14px;top:12px;font-size:12px;color:var(--mute)}
.badge b{color:var(--ink);font-weight:600}
aside{background:var(--panel);border-left:1px solid var(--line);padding:18px 18px 24px;overflow:auto}
h1{font-size:17px;margin:0 0 2px}
.sub{color:var(--mute);font-size:12px;margin-bottom:16px}
h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin:18px 0 8px;font-weight:600}
select,input[type=range]{width:100%}
select{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0}
.row label{display:flex;align-items:center;gap:8px;cursor:pointer}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{flex:1;border:0;background:transparent;color:var(--ink);padding:6px 4px;font:inherit;font-size:12px;cursor:pointer}
.seg button.on{background:var(--ink);color:var(--panel)}
button.play{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:6px 12px;font:inherit;cursor:pointer}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0;font-size:13px}
dt{color:var(--mute)}dd{margin:0;font-variant-numeric:tabular-nums}
.note{color:var(--mute);font-size:12px;margin-top:14px}
.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
@media (max-width:760px){.wrap{grid-template-columns:1fr;grid-template-rows:60vh auto;height:auto}aside{border-left:0;border-top:1px solid var(--line);padding:16px}}
</style>
</head>
<body>
<div class="wrap">
  <div id="view"><div class="badge" id="badge"></div><div class="hint">drag to rotate · scroll / pinch to zoom · right-drag to pan</div></div>
  <aside>
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>

    <h2>Shape</h2>
    <select id="pick"></select>
    <div class="row"><label><input type="checkbox" id="ens"> Overlay all posterior draws</label></div>

    <h2>Display</h2>
    <div class="seg" id="mode"><button data-m="smooth" class="on">Shaded</button><button data-m="flat">Facets</button><button data-m="wire">Wireframe</button></div>
    <div class="row"><label><input type="checkbox" id="axes" checked> Spin axis &amp; body axes</label></div>
    <div class="row"><button class="play" id="play">Pause rotation</button><span id="speedlab" style="color:var(--mute);font-size:12px"></span></div>
    <input type="range" id="speed" min="2" max="30" value="8" aria-label="seconds per rotation">

    <h2>Selected spin state</h2>
    <dl id="info"></dl>

    <h2>Posterior summary</h2>
    <dl id="post"></dl>

    <div class="note">
      <span class="key" style="background:var(--acc)"></span>selected shape &nbsp;
      <span class="key" style="background:var(--mute)"></span>posterior draws &nbsp;
      <span class="key" style="background:#d99a1e"></span>max-inertia axis<br>
      <span id="caveat"></span><br>
      Body frame: z (vertical) is the spin axis. Volume normalised to 1. Shapes are shown rotated
      about the spin axis so the longest equatorial axis lies along x (convex runs: each shape separately,
      since the rotational phase is a gauge freedom; hybrid runs: one common rotation, because all draws
      share the gauge-fixed body frame). Physical orientations (phase 0 at t<sub>ref</sub>) are in the OBJ files.
    </div>
  </aside>
</div>

<script>/* three.js r128, MIT licence (c) 2010-2021 three.js authors */
__THREEJS__</script>
<script>__ORBIT__</script>
<script>
// null-safe lookup: optional elements may be absent in derived viewers
const $id = id => document.getElementById(id) || document.createElement('span');
const SHAPES = __SHAPES__;
const META = __META__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (x, d) => Number(x).toFixed(d);
const q = (a, d) => `${fmt(a[1], d)} <span style="color:var(--mute)">(${fmt(a[0], d)}–${fmt(a[2], d)})</span>`;

$id('title').textContent = META.designation;
$id('subtitle').textContent = (META.kind ? META.kind + ' · ' : '') +
  `${META.kind ? 'S' : 'Convex s'}hape from ${META.npts} Rubin points (${META.bands.join('/')}), ${META.nnights} nights, phase ${fmt(META.alpha[1],1)}°→${fmt(META.alpha[0],1)}°`;
$id('post').innerHTML = [
  ['Period', q(META.P, 5) + ' h'],
  ['Prograde', fmt(100*META.fpro, 0) + ' %'],
  ['Jitter', q(META.jit.map(x => x*1000), 1) + ' mmag'],
  META.ab ? ['a/b (proj.)', q(META.ab, 2)] : null,
  META.bc ? ['b/c (proj.)', q(META.bc, 2)] : null,
  ['a/b (extent)', q(META.eab, 2)],
  ['b/c (extent)', q(META.ebc, 2)],
  META.cl ? ['c<sub>L</sub>', q(META.cl, 3)] : null,
  ...(META.extra || []),
].filter(Boolean).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

{ // inertia caveat, computed from the shapes in this file
  const ang = SHAPES.map(s => s.iang).sort((a, b) => a - b);
  const med = ang[Math.floor(ang.length / 2)];
  $id('caveat').innerHTML = med > 30
    ? `<b style="color:var(--ink)">Caveat:</b> in most of these shapes the spin axis (vertical) is far from the axis of maximum inertia (dashed; median angle ${fmt(med,0)}°), so they would not be stable, relaxed rotators. Treat the pole solution with caution.<br>`
    : `Spin axis vs maximum-inertia axis: median ${fmt(med,0)}° over the shapes shown.<br>`;
}
// ---------------------------------------------------------------- three.js scene
const view = $id('view');
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
{ // frame the largest shape: default distance for bodies of radius ~1, further out for longer ones
  const R = Math.max(1, ...SHAPES.map(s => 0.5 * Math.max(...s.ext)));
  camera.position.set(2.6 * R, 1.3 * R, 2.6 * R);
}
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x444450, 0.45));
const sun = new THREE.DirectionalLight(0xffffff, 0.95);
sun.position.set(3, 2.5, 2);
scene.add(sun);

// body group: body z (spin) -> world y (up)
const spinGroup = new THREE.Group();
const body = new THREE.Group();
body.rotation.x = -Math.PI / 2;
spinGroup.add(body);
scene.add(spinGroup);

function geometryOf(s) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(s.v, 3));
  g.setIndex(s.f);
  g.computeVertexNormals();
  return g;
}
const geoms = SHAPES.map(geometryOf);

const accent = () => new THREE.Color(css('--acc') || '#c4452f');
const matSmooth = new THREE.MeshStandardMaterial({color: 0xb8b2a7, roughness: 0.85, metalness: 0.0});
const matFlat = new THREE.MeshStandardMaterial({color: 0xb8b2a7, roughness: 0.9, flatShading: true});
const matWire = new THREE.MeshBasicMaterial({color: accent(), wireframe: true});
let mode = 'smooth';
const mainMesh = new THREE.Mesh(geoms[0], matSmooth);
body.add(mainMesh);
const edgeLines = new THREE.LineSegments(new THREE.EdgesGeometry(geoms[0], 25),
  new THREE.LineBasicMaterial({color: accent(), transparent: true, opacity: 0.35}));
body.add(edgeLines);

// ensemble overlay: faint silhouette shells of all posterior draws
const ensGroup = new THREE.Group();
SHAPES.slice(1).forEach((s, i) => {
  const m = new THREE.Mesh(geoms[i + 1], new THREE.MeshBasicMaterial({
    color: 0x8a8f99, wireframe: true, transparent: true, opacity: 0.07}));
  ensGroup.add(m);
});
ensGroup.visible = false;
body.add(ensGroup);

// axes: spin axis arrow (z) and body x, y
const axesGroup = new THREE.Group();
const mkArrow = (dir, len, col) => new THREE.ArrowHelper(new THREE.Vector3(...dir), new THREE.Vector3(0,0,0), len, col, 0.08, 0.05);
axesGroup.add(mkArrow([0,0,1], 1.05, 0x1f2430));
axesGroup.add(mkArrow([0,0,-1], 1.05, 0x1f2430));
axesGroup.add(mkArrow([1,0,0], 0.95, 0xc4452f));
axesGroup.add(mkArrow([0,1,0], 0.85, 0x2a9d8f));
body.add(axesGroup);

// maximum-inertia axis of the selected shape (dashed): a relaxed rotator spins about it
const inertiaLine = new THREE.Line(new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({color: 0xd99a1e, dashSize: 0.06, gapSize: 0.04}));
body.add(inertiaLine);
function setInertia(s) {
  const a = s.iax, L = 1.1;
  inertiaLine.geometry.dispose();
  inertiaLine.geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-a[0]*L, -a[1]*L, -a[2]*L), new THREE.Vector3(a[0]*L, a[1]*L, a[2]*L)]);
  inertiaLine.computeLineDistances();
}

// ---------------------------------------------------------------- UI
const pick = $id('pick');
SHAPES.forEach((s, i) => {
  const o = document.createElement('option');
  o.value = i;
  o.textContent = i === 0 ? `${s.name}  (λ ${fmt(s.lam,0)}°, β ${fmt(s.bet,0)}°)` : `${s.name}  (λ ${fmt(s.lam,0)}°, β ${fmt(s.bet,0)}°)`;
  pick.appendChild(o);
});
function showInfo(i) {
  const s = SHAPES[i];
  const e = s.ext;
  const rows = [
    ['Pole λ, β', `${fmt(s.lam,1)}°, ${fmt(s.bet,1)}°`],
    ['Sense', s.bet > 0 ? 'prograde' : 'retrograde'],
    ['Sidereal P', `${fmt(s.P,5)} h`],
    ['Extents x:y:z', `${fmt(e[0]/e[2],2)} : ${fmt(e[1]/e[2],2)} : 1`],
    ['Principal moments', s.mom.map(m => fmt(m,2)).join(' : ')],
    ['Spin vs max-inertia axis', `<span style="color:${s.iang > 20 ? 'var(--acc)' : 'inherit'}">${fmt(s.iang,0)}°</span>`],
  ];
  if (s.ab !== undefined) rows.push(['a/b, b/c (proj.)', `${fmt(s.ab,2)}, ${fmt(s.bc,2)}`]);
  $id('info').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $id('badge').innerHTML = `<b>${s.name}</b> · P = ${fmt(s.P,4)} h`;
}
function setShape(i) {
  mainMesh.geometry = geoms[i];
  edgeLines.geometry.dispose();
  edgeLines.geometry = new THREE.EdgesGeometry(geoms[i], 25);
  showInfo(i);
  setInertia(SHAPES[i]);
}
pick.addEventListener('change', e => setShape(+e.target.value));
$id('ens').addEventListener('change', e => { ensGroup.visible = e.target.checked; });
$id('axes').addEventListener('change', e => { axesGroup.visible = e.target.checked; inertiaLine.visible = e.target.checked; });
document.querySelectorAll('#mode button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('#mode button').forEach(x => x.classList.toggle('on', x === b));
  mode = b.dataset.m;
  mainMesh.material = mode === 'smooth' ? matSmooth : mode === 'flat' ? matFlat : matWire;
  edgeLines.visible = mode !== 'wire';
}));
let playing = true;
const playBtn = $id('play');
playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause rotation' : 'Play rotation'; });
const speed = $id('speed');
const speedLab = $id('speedlab');
const updSpeed = () => {
  const real = SHAPES[+pick.value].P * 3600;
  speedLab.textContent = `1 rotation / ${speed.value} s (×${Math.round(real / speed.value)} real time)`;
};
speed.addEventListener('input', updSpeed);
pick.addEventListener('change', updSpeed);

// ---------------------------------------------------------------- loop
function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
let last = performance.now();
function tick(now) {
  const dt = (now - last) / 1000; last = now;
  if (playing) spinGroup.rotation.y += 2 * Math.PI * dt / (+speed.value);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
setShape(0); updSpeed(); resize(); requestAnimationFrame(tick);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
