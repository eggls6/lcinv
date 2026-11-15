"""Figures, HTML viewers and summaries for the hybrid inversion.

Every figure is produced inside its own guard: a failure in one (missing optional
package, empty group, absent log-likelihood) is reported and the rest continue.
"""
import json
import os
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lcinv import setup_data
from lcinv.plots import _style, BAND_COL, INK, MUTE

MODE_COL = ["#c4452f", "#2a6fb0", "#2a9d8f", "#8a5a2b"]


def guarded(fn):
    """Decorator: a figure function that fails prints a message instead of failing the finished
    run.
    """
    def wrap(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # keep the report going
            print(f"[report] {fn.__name__} skipped: {e}")
            traceback.print_exc(limit=1)
            return None
    return wrap


def _moll(lam_deg):
    l = np.radians(np.asarray(lam_deg))
    return np.where(l > np.pi, l - 2 * np.pi, l)


@guarded
def pole_map(path, mode_runs, global_post=None, los=None):
    """Pole posterior of every spin state over the global posterior, ``fig_pole_hybrid.png``."""
    _style()
    fig = plt.figure(figsize=(9.5, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    if global_post is not None:
        ax.scatter(_moll(global_post[:, 0]), np.radians(global_post[:, 1]), s=2, color=MUTE,
                   alpha=0.25, lw=0, label="global layer (convex, nested)")
    for i, mr in enumerate(mode_runs):
        d = mr["der"]
        ax.scatter(_moll(d["lam"]), np.radians(d["bet"]), s=3, color=MODE_COL[i % 4], alpha=0.45, lw=0,
                   label=f"{mr['name']}: NUTS, weight {mr['weight']:.2f}")
    if los is not None:
        for (l, b), lab in zip(los, ("LOS", "anti-LOS")):
            ax.plot(_moll(l), np.radians(b), "x", color=INK, ms=7, mew=1.5)
            ax.annotate(lab, (_moll(l), np.radians(b)), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.grid(True, color=MUTE, lw=0.3, alpha=0.6)
    ax.set_xticklabels([f"{d}°" for d in (210, 240, 270, 300, 330, 0, 30, 60, 90, 120, 150)], fontsize=7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=7, frameon=False, markerscale=4)
    ax.set_title("Pole posterior: hybrid (local NUTS per spin state) over global layer", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@guarded
def traces(path, mode_runs):
    """NUTS trace plots of every spin state, ``fig_traces.png``."""
    _style()
    keys = [("lam", "λ [deg]"), ("bet", "β [deg]"), ("P", "P [h]"), ("jitter", "σ_jit [mag]"),
            ("inertia_angle", "spin–I_max angle [deg]")]
    fig, axs = plt.subplots(len(keys), len(mode_runs), figsize=(5.2 * len(mode_runs), 1.6 * len(keys)), squeeze=False)
    for j, mr in enumerate(mode_runs):
        nc, nd = mr["res"]["x"].shape[:2]
        for i, (k, lab) in enumerate(keys):
            a = mr["der"][k].reshape(nc, nd)
            for c in range(nc):
                axs[i, j].plot(a[c], lw=0.5, alpha=0.8)
            axs[i, j].set_ylabel(lab, fontsize=7)
        axs[0, j].set_title(mr["name"], fontsize=9)
        axs[-1, j].set_xlabel("draw")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@guarded
def diagnostics(path, mode_runs, prior_ln_sigma=None):
    """Hyperparameter and inertia-angle posteriors, ``fig_diagnostics.png``."""
    _style()
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.8))
    for i, mr in enumerate(mode_runs):
        d, col = mr["der"], MODE_COL[i % 4]
        axs[0].hist(d["inertia_angle"], bins=30, color=col, alpha=0.6, label=mr["name"])
        axs[1].hist(1000 * d["jitter"], bins=30, color=col, alpha=0.6)
        axs[2].hist(d["sigma_f"], bins=30, color=col, alpha=0.6)
        axs[3].hist(d["ell"], bins=30, color=col, alpha=0.6)
    axs[0].set_xlabel("spin axis vs max-inertia axis [deg]")
    axs[1].set_xlabel("jitter [mmag]")
    axs[2].set_xlabel("σ_f (sd of log r)")
    axs[3].set_xlabel("ℓ (angular scale)")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@guarded
def radius_maps(path, mode_runs, grid):
    """Posterior mean radius and its standard deviation on the body surface,
    ``fig_radius_maps.png``.
    """
    _style()
    nq = grid.nlat * grid.nlon
    fig, axs = plt.subplots(len(mode_runs), 2, figsize=(10, 2.8 * len(mode_runs)), squeeze=False)
    lat = 90 - np.degrees(np.arccos(grid.u_q[:, 2].reshape(grid.nlat, grid.nlon)[:, 0]))
    for i, mr in enumerate(mode_runs):
        r = mr["der"]["r_mesh"][:, :nq]
        mean = r.mean(0).reshape(grid.nlat, grid.nlon)
        sd = r.std(0).reshape(grid.nlat, grid.nlon)
        ext = [0, 360, lat.min(), lat.max()]
        im0 = axs[i, 0].imshow(mean, origin="lower", extent=ext, aspect="auto", cmap="cividis")
        im1 = axs[i, 1].imshow(100 * sd, origin="lower", extent=ext, aspect="auto", cmap="viridis")
        fig.colorbar(im0, ax=axs[i, 0], label="mean r / R_eq")
        fig.colorbar(im1, ax=axs[i, 1], label="σ_r [% R_eq]")
        axs[i, 0].set_ylabel(f"{mr['name']}\nbody latitude [deg]")
    for a in axs[-1]:
        a.set_xlabel("body longitude [deg]")
    axs[0, 0].set_title("mean radius", fontsize=9)
    axs[0, 1].set_title("radius uncertainty", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@guarded
def lightcurves(path, D, mode_runs, n_curves=25):
    """Data with the main spin state's posterior-median offsets removed, and posterior-predictive
    curves (r band) of every spin state."""
    _style()
    main = max(mode_runs, key=lambda m: m["weight"])
    M = main["model"]
    nb = M.X.shape[1] - 1
    eta_ref = np.median(main["der"]["eta"], axis=0)
    X = np.asarray(M.X)
    y = np.asarray(M.y) - X @ eta_ref
    night = D["night"]
    nn = night.max() + 1
    fig, axs = plt.subplots(1, nn, figsize=(2.6 * nn, 3.3), sharey=True)
    jd = D["df"]["epoch"].values
    r_idx = D["bands"].index("r") if "r" in D["bands"] else 0
    rng = np.random.default_rng(3)
    for k in range(nn):
        ax = axs[k]
        m = night == k
        t0 = np.floor(D["t"][m].min() - 0.5) + 0.5
        jd_d = setup_data.dense_epochs(jd[m].min() - 0.01, jd[m].max() + 0.01, M.P0, n_min=250)
        g = setup_data.plot_geometry(D, jd_d)
        texp = float(np.median(D["exptime"])) if np.any(np.asarray(D.get("exptime", [0])) > 0) else None
        Xd = np.c_[np.eye(nb)[np.full(len(jd_d), r_idx)], g["alpha"]]
        for i, mr in enumerate(mode_runs):
            xs = mr["res"]["x"].reshape(-1, mr["model"].ndim)
            pick = rng.choice(len(xs), size=min(n_curves, len(xs)), replace=False)
            for j in pick:
                L = np.asarray(mr["model"].brightness(xs[j], t=g["t_emit"], e_sun=g["e_sun"], e_obs=g["e_obs"],
                                                      exptime=texp))
                curve = -2.5 * np.log10(L) + Xd @ mr["der"]["eta"][j] - Xd @ eta_ref
                ax.plot((g["t_emit"] - t0) * 24, curve, color=MODE_COL[i % 4], lw=0.5, alpha=0.25)
        for bi, bl in enumerate(D["bands"]):
            mb = m & (D["band_idx"] == bi)
            ax.errorbar((D["t"][mb] - t0) * 24, y[mb], yerr=D["err"][mb], fmt="o", ms=2.5,
                        color=BAND_COL.get(bl, INK), lw=0.6, mew=0, label=bl)
        ax.set_xlabel(f"hours after JD {t0:.1f}")
        ax.set_title(f"night {k + 1}", fontsize=8)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("reduced mag − offsets − phase term")
    axs[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@guarded
def html_viewer(path, mr, grid, meta):
    """Radius-uncertainty viewer for one spin state, reusing the three.js template."""
    import make_uncertainty_viewer as muv
    from make_viewer import principal_axes
    r = mr["der"]["r_mesh"]
    mean = r.mean(0)
    verts = grid.u_m * mean[:, None]
    verts = verts - verts.mean(0)
    mom, iax = principal_axes(verts, grid.tri)
    iang = float(np.degrees(np.arccos(min(1.0, abs(iax[2])))))
    ext = verts.max(0) - verts.min(0)
    data = dict(dirs=np.round(grid.u_m, 5).ravel().tolist(), tri=grid.tri.ravel().tolist(),
                mean=np.round(mean, 5).tolist(), std=np.round(r.std(0), 5).tolist(),
                q16=np.round(np.percentile(r, 16, 0), 5).tolist(),
                q84=np.round(np.percentile(r, 84, 0), 5).tolist(),
                iax=np.round(iax, 4).tolist(), iang=round(iang, 1), mom=np.round(mom, 3).tolist(),
                ext=np.round(ext / ext[2], 3).tolist(), ndraw=int(len(r)),
                fpro=float(np.mean(mr["der"]["bet"] > 0)),
                P=[float(q) for q in np.percentile(mr["der"]["P"], [16, 50, 84])], **meta)
    vend = os.path.join(os.path.dirname(os.path.abspath(muv.__file__)), "lcinv", "vendor")
    html = muv.TEMPLATE.replace("__THREEJS__", open(os.path.join(vend, "three.min.js")).read()) \
        .replace("__ORBIT__", open(os.path.join(vend, "OrbitControls.js")).read()) \
        .replace("__DATA__", json.dumps(data, separators=(",", ":"))) \
        .replace("__TITLE__", f"{meta['designation']} {mr['name']} radius uncertainty")
    note_old = html[html.index('<div class="note">'):html.index("</aside>")]
    lam, bet = np.median(mr["der"]["lam"]), np.median(mr["der"]["bet"])
    note_new = (f'<div class="note">Hybrid model, {mr["name"]} (pole ≈ λ {lam:.0f}°, β {bet:.0f}°; '
                f'weight {mr["weight"]:.2f}). Mean radius field of <span id="nd">{len(r)}</span> NUTS draws of the joint '
                'spin + shape posterior (star-shaped log-radius field with a Matérn-like harmonic prior, '
                'principal-axis rotation prior). The rotational phase is gauge-fixed at t<sub>ref</sub>, so '
                'all draws share one body frame and no registration is needed. Radii are in units of each '
                'draw\'s volume-equivalent radius R<sub>eq</sub>; the vertical axis is the spin axis and the '
                'dashed line the maximum-inertia axis of the mean shape.</div>\n  ')
    html = html.replace(note_old, note_new)
    html = html.replace("Posterior radius uncertainty ·", f"Hybrid {mr['name']} ·")
    with open(path, "w") as f:
        f.write(html)
    return path


def write_summary(path, summary):
    """Write ``summary_hybrid.json`` (NumPy values converted to plain JSON)."""
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=conv)
