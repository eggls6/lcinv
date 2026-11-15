#!/usr/bin/env python
"""Export posterior shapes as Wavefront .obj files (global or hybrid results).

    python export_obj.py results/K03Q21N                 # global (convex) run
    python export_obj.py results/K03Q21N_hybrid_quat     # hybrid run, every spin state
    python export_obj.py results/X --n 50 --diameter-km 1.2 --frame principal

Global run (nested_results.pkl), written to <outdir>/obj/:
    shape_maxlike.obj   maximum-likelihood convex shape
    shape_mean.obj      Minkowski mean: the convex body whose facet areas are the posterior
                        mean areas (always a valid convex shape)
    draw_###.obj        posterior draws (spin sample + Laplace draw of the shape)

Hybrid run (posterior_spin_state_*.npz), written to <outdir>/obj/spin_state_<k>/:
    shape_mean.obj, shape_median.obj   mean / median radius at each mesh direction
    shape_p16.obj, shape_p84.obj       16th / 84th percentile radius surfaces (+-1 sigma envelopes)
    draw_###.obj                       NUTS draws

Frames: --frame body (default) keeps the body frame of the inversion: z = spin axis, and the
rotational phase is zero at t_ref (given in the file header), so draws are mutually aligned.
--frame principal rotates each shape to its own principal axes of inertia (x: smallest
moment = longest axis, z: largest moment), which suits 3D printing or comparison with other
models but discards the spin orientation.

Units: by default every shape is scaled to a volume-equivalent radius R_eq = 1 (the volume of
a unit sphere). Lightcurves alone do not fix the size: --diameter-km D scales to a
volume-equivalent diameter of D km (e.g. from H and an assumed albedo); --scale s sets R_eq = s.
Each export replaces the .obj files of the previous one in the same folder.
Every file is centred on its centre of mass (uniform density) and has outward-facing triangles.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from lcinv import minkowski as mk


# ----------------------------------------------------------------------------------- geometry
def mass_properties(v, f):
    """Volume, centroid and inertia tensor (about the centroid, unit density) of a closed,
    outward-oriented triangle mesh, by signed tetrahedra from the origin."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    dV = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    V = dV.sum()
    cen = (dV[:, None] * (a + b + c)).sum(0) / (4.0 * V)
    # second moments of each tetrahedron (0, a, b, c): (dV/20) (sum_i x_i x_i^T + s s^T)
    s = a + b + c
    S = (dV[:, None, None] / 20.0) * (np.einsum("ij,ik->ijk", a, a) + np.einsum("ij,ik->ijk", b, b)
                                      + np.einsum("ij,ik->ijk", c, c) + np.einsum("ij,ik->ijk", s, s))
    M = S.sum(0) - V * np.outer(cen, cen)
    I = np.trace(M) * np.eye(3) - M
    return V, cen, I


def orient_outward(v, f):
    """Faces of a closed mesh reordered so that every normal points away from the centroid."""
    f = f.copy()
    p = v[f]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = np.einsum("ij,ij->i", n, p.mean(1) - v.mean(0)) < 0
    f[flip] = f[flip][:, ::-1]
    return f


def normalise(v, f, frame, scale):
    """Centre on the centre of mass, scale to R_eq = scale, optionally rotate to principal axes."""
    f = orient_outward(v, f)
    V, cen, I = mass_properties(v, f)
    v = v - cen
    v = v * (scale / (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0))
    info = ""
    if frame == "principal":
        _, _, I = mass_properties(v, f)
        w, U = np.linalg.eigh(I)             # ascending moments: x smallest (long axis)
        if np.linalg.det(U) < 0:
            U[:, 2] *= -1
        v = v @ U
        info = "frame: principal axes (x smallest moment / longest, z largest moment)"
    return v, f, info


def fresh_dir(dest):
    """Create dest and remove .obj files of an earlier export, so frames/units never mix."""
    os.makedirs(dest, exist_ok=True)
    for fp in glob.glob(os.path.join(dest, "*.obj")):
        os.remove(fp)


def header(lines, frame_info, unit):
    """OBJ header lines with frame and units."""
    out = list(lines)
    out.append(frame_info or "frame: body (z = spin axis; rotational phase 0 at t_ref)")
    out.append(f"units: {unit}; origin: centre of mass (uniform density)")
    return "\n".join(out)


def write(path, v, f, lines, frame, scale, unit):
    """Normalise a mesh to the requested frame and scale and write it with its header."""
    v, f, info = normalise(v, f, frame, scale)
    mk.write_obj(path, v, f, header(lines, info, unit))


# ----------------------------------------------------------------------------------- global
def export_global(outdir, a, scale, unit):
    """Export the shapes of a global (convex) run to <outdir>/obj."""
    from lcinv import sampler
    from lcinv.convex import fibonacci_sphere, real_sph_harm_basis
    R = sampler.load(os.path.join(outdir, "nested_results.pkl"))
    summ = json.load(open(os.path.join(outdir, "summary.json")))
    ens, args = R["ensemble"], R["args"]
    normals = fibonacci_sphere(args["nfacets"])
    Y, _ = real_sph_harm_basis(normals, args["lmax"])
    w0 = 4 * np.pi / args["nfacets"]
    target = summ.get("horizons_target", args.get("designation", ""))
    t_ref = summ["max_likelihood"]["t_ref_jd"]
    dest = os.path.join(outdir, "obj")
    fresh_dir(dest)

    def body(A):
        v, f = mk.reconstruct(normals, A)
        return v, f

    def spin(th):
        return f"lambda={th[0] % 360:.2f} beta={th[1]:.2f} P={th[2]:.6f} h, t_ref JD {t_ref:.6f}"

    th = summ["max_likelihood"]["theta"]
    i_ml = int(np.argmax(ens["logL"]))
    ml_file = os.path.join(outdir, "shapes", "shape_maxlike.obj")
    if os.path.exists(ml_file):
        v = np.array([l.split()[1:4] for l in open(ml_file) if l.startswith("v ")], float)
        f = np.array([l.split()[1:4] for l in open(ml_file) if l.startswith("f ")], int) - 1
    else:
        v, f = body(w0 * np.exp(Y @ ens["c_hat"][i_ml]))
        th = ens["theta"][i_ml]
    write(os.path.join(dest, "shape_maxlike.obj"), v, f,
          [f"{target}: maximum-likelihood convex shape", spin(th)], a.frame, scale, unit)

    A = np.mean([w0 * np.exp(Y @ c) for c in ens["c_draw"]], 0)
    v, f = body(A)
    write(os.path.join(dest, "shape_mean.obj"), v, f,
          [f"{target}: Minkowski mean of {len(ens['c_draw'])} posterior draws (mean facet areas)",
           "spin: draws span both mirror poles; use the draws for spin-specific shapes"],
          a.frame, scale, unit)

    n = min(a.n, len(ens["c_draw"]))
    for k in range(n):
        v, f = body(w0 * np.exp(Y @ ens["c_draw"][k]))
        write(os.path.join(dest, f"draw_{k:03d}.obj"), v, f,
              [f"{target}: posterior draw {k}", spin(ens["theta"][k])], a.frame, scale, unit)
    print(f"wrote shape_maxlike, shape_mean and {n} draws -> {dest}")


# ----------------------------------------------------------------------------------- hybrid
def export_hybrid(outdir, a, scale, unit):
    """Export the shapes of every spin state of a hybrid run to <outdir>/obj/spin_state_<k>."""
    from lcinv.hybrid.quadrature import make_grid
    s = json.load(open(os.path.join(outdir, "summary_hybrid.json")))
    g = s["grid"]
    grid = make_grid(g["lmax"], lmin=g.get("lmin", 2), nlat=g["nlat"], nlon=g["nlon"])
    u, tri = np.asarray(grid.u_m), np.asarray(grid.tri)
    rng = np.random.default_rng(a.seed)
    from lcinv.hybrid import naming
    for k, fpath in naming.posterior_files(outdir):
        z = np.load(fpath)
        name = naming.name(k)
        meta = naming.state_info(s, k)
        r = z["r_mesh"]
        if r.shape[1] != len(u):
            raise ValueError(f"{fpath}: {r.shape[1]} radii but the grid has {len(u)} directions")
        dest = os.path.join(outdir, "obj", naming.tag(k))
        fresh_dir(dest)
        pole = meta.get("pole", {}).get("mean_pole")
        P = meta.get("P_h", [np.nan] * 3)[1]
        wgt = meta.get("weight_local_laplace", np.nan)
        base = [f"{s.get('designation', '')} hybrid {name}: weight {wgt:.3f}",
                f"mean pole lambda={pole[0]:.2f} beta={pole[1]:.2f}, median P={P:.6f} h" if pole else ""]
        for tag, rr in (("mean", r.mean(0)), ("median", np.median(r, 0)),
                        ("p16", np.percentile(r, 16, 0)), ("p84", np.percentile(r, 84, 0))):
            write(os.path.join(dest, f"shape_{tag}.obj"), rr[:, None] * u, tri,
                  base + [f"{tag} radius over {len(r)} NUTS draws at each mesh direction"],
                  a.frame, scale, unit)
        idx = rng.choice(len(r), size=min(a.n, len(r)), replace=False)
        for j, i in enumerate(sorted(idx)):
            spin = f"draw {i}: lambda={z['lam'][i] % 360:.2f} beta={z['bet'][i]:.2f} P={z['P'][i]:.6f} h"
            write(os.path.join(dest, f"draw_{j:03d}.obj"), r[i][:, None] * u, tri,
                  base + [spin], a.frame, scale, unit)
        print(f"{name}: wrote mean, median, p16, p84 and {len(idx)} draws -> {dest}")


def main():
    """Command line of export_obj.py."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="results folder of run_inversion.py or run_hybrid.py")
    p.add_argument("--n", type=int, default=20, help="number of posterior draws to export")
    p.add_argument("--frame", choices=["body", "principal"], default="body",
                   help="body: z = spin axis, phase 0 at t_ref (draws aligned); principal: each shape "
                        "in its own principal-axis frame")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--diameter-km", type=float, default=None, help="volume-equivalent diameter [km]")
    g.add_argument("--scale", type=float, default=1.0, help="volume-equivalent radius in output units")
    p.add_argument("--seed", type=int, default=0, help="seed for choosing the hybrid draws")
    p.add_argument("--H", type=float, default=None,
                   help="absolute magnitude: scale to the CNEOS size estimate D = 10^(3.1236 - 0.5 log10 p - 0.2 H) km")
    p.add_argument("--albedo", type=float, default=0.15, help="geometric albedo for --H / --size-from-H")
    p.add_argument("--size-from-H", action="store_true",
                   help="look H up in the JPL Small-Body Database and scale to the CNEOS size estimate")
    a = p.parse_args()
    if not a.diameter_km and (a.H is not None or a.size_from_H):
        from lcinv import size
        desig = None
        for fn in ("summary.json", "summary_hybrid.json"):
            fp = os.path.join(a.outdir, fn)
            if os.path.exists(fp):
                desig = json.load(open(fp)).get("designation")
                break
        e = size.estimate_for(desig, H=a.H, albedo=a.albedo, lookup=a.size_from_H)
        a.diameter_km = e["diameter_km"]
        print(f"size: H = {e['H']:.2f} ({e['H_source']}), albedo {e['albedo']:g} -> D = {e['diameter_km']:.3f} km "
              f"(CNEOS; {e['diameter_min_km']:.3f}-{e['diameter_max_km']:.3f} km for albedo 0.30-0.05)")
    if a.diameter_km:
        scale, unit = a.diameter_km / 2.0, "km"
    else:
        scale, unit = a.scale, ("R_eq (volume-equivalent radius)" if a.scale == 1.0 else f"{a.scale} x R_eq")
    sp = os.path.join(a.outdir, "summary.json")
    if os.path.exists(sp) and json.load(open(sp)).get("model") == "contact binary":
        import subprocess
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_binary.py"),
               "--replot", "--outdir", a.outdir, "--n-obj", str(a.n)]
        if a.diameter_km:
            cmd += ["--diameter-km", str(a.diameter_km)]
        print("contact-binary run: OBJ files are written to shapes/ by run_binary.py --replot")
        raise SystemExit(subprocess.call(cmd))
    if os.path.exists(os.path.join(a.outdir, "nested_results.pkl")):
        export_global(a.outdir, a, scale, unit)
    elif glob.glob(os.path.join(a.outdir, "posterior_spin_state_*.npz")) or \
            glob.glob(os.path.join(a.outdir, "posterior_mode*.npz")):
        export_hybrid(a.outdir, a, scale, unit)
    else:
        raise SystemExit(f"{a.outdir}: no nested_results.pkl or posterior_spin_state_*.npz found")


if __name__ == "__main__":
    main()
