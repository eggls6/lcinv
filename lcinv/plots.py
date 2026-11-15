"""Diagnostic figures of the scripts (Matplotlib, written as PNG).

`folded` is the phase-folded lightcurve figure shared by all models; `lcinv.folding` wraps it
for the convex and hybrid layers.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

INK = "#1f2430"
ACC = "#c4452f"
MUTE = "#8a8f99"
BAND_COL = {"g": "#2a9d8f", "r": "#c4452f", "i": "#6c4ba3", "z": "#444444", "u": "#3a6fb0", "y": "#7a5c2e"}


def _style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160})


def periodogram(path, P, chi2, nu, ref, minima):
    """Periodogram figure, ``fig_periodogram.png``.

    Shows chi2/nu over the searched range with the six lowest minima labelled, and a zoom on the
    best minimum with its threshold interval.

    Parameters
    ----------
    path : str
    P, chi2 : ndarray
        Periodogram (`lcinv.period.periodogram`).
    nu : int
        Degrees of freedom.
    ref : dict
        Fine scan (`lcinv.period.refine_period`).
    minima : list
        Labelled minima (`lcinv.period.local_minima`).
    """
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    a.plot(P, chi2 / nu, color=INK, lw=0.7)
    a.set_xscale("log")
    a.set_xlabel("rotation period [h]")
    a.set_ylabel(r"$\chi^2/\nu$ (6-harmonic multiband Fourier)")
    for p, c in minima[:4]:
        a.annotate(f"{p:.4f} h", (p, c / nu), xytext=(0, -14), textcoords="offset points",
                   ha="center", fontsize=7, color=MUTE)
    a.axvline(ref["P"], color=ACC, lw=0.8, ls=":")
    b.plot(ref["scan_P"], ref["scan_chi2"] - ref["chi2"], color=INK, lw=0.8)
    b.axhspan(0, ref["chi2"] * np.sqrt(2 / ref["nu"]), color=ACC, alpha=0.12, lw=0)
    b.axvline(ref["P"], color=ACC, lw=0.8)
    b.set_xlim(ref["P"] - 4 * (ref["P_hi"] - ref["P"]), ref["P"] + 4 * (ref["P_hi"] - ref["P"]))
    b.set_ylim(0, 4 * ref["chi2"] * np.sqrt(2 / ref["nu"]))
    b.set_xlabel("period [h]")
    b.set_ylabel(r"$\Delta\chi^2$")
    b.set_title(f"P = {ref['P']:.5f} h; shaded: $\\chi^2<\\chi^2_{{min}}(1+\\sqrt{{2/\\nu}})$", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pole_map(path, samples, los_lonlat, weights=None, map_theta=None):
    """Posterior of the pole on a Mollweide map, with marginals of P and jitter."""
    _style()
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_axes([0.03, 0.12, 0.62, 0.8], projection="mollweide")
    lam = np.radians(samples[:, 0])
    lam = np.where(lam > np.pi, lam - 2 * np.pi, lam)
    bet = np.radians(samples[:, 1])
    hb = ax.hexbin(lam, bet, gridsize=36, cmap="Reds", mincnt=1, linewidths=0.1)
    ax.grid(True, color=MUTE, lw=0.3, alpha=0.6)
    ax.set_xticklabels([f"{d}°" for d in (210, 240, 270, 300, 330, 0, 30, 60, 90, 120, 150)], fontsize=7)
    for (l, b), lab in zip(los_lonlat, ("line of sight", "anti-LOS")):
        l = np.radians(l)
        l = l - 2 * np.pi if l > np.pi else l
        ax.plot(l, np.radians(b), marker="x", color=INK, ms=7, mew=1.5)
        ax.annotate(lab, (l, np.radians(b)), xytext=(5, 5), textcoords="offset points", fontsize=7)
    if map_theta is not None:
        l = np.radians(map_theta[0])
        l = l - 2 * np.pi if l > np.pi else l
        ax.plot(l, np.radians(map_theta[1]), marker="*", color="#f2b134", ms=11, mec=INK, mew=0.6)
    ax.set_title(r"pole posterior (ecliptic J2000, $\lambda,\beta$); $\star$ = max-likelihood sample", fontsize=9)
    cb = fig.colorbar(hb, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08)
    cb.set_label("posterior samples per cell", fontsize=8)
    a2 = fig.add_axes([0.72, 0.58, 0.25, 0.32])
    a2.hist(samples[:, 2], bins=40, color=INK, alpha=0.85)
    a2.set_xlabel("sidereal period [h]")
    a2.ticklabel_format(useOffset=False)
    a3 = fig.add_axes([0.72, 0.13, 0.25, 0.32])
    a3.hist(np.exp(samples[:, 3]) * 1000, bins=40, color=INK, alpha=0.85)
    a3.set_xlabel(r"jitter $\sigma_{\rm jit}$ [mmag]")
    fig.savefig(path)
    plt.close(fig)


def lightcurves(path, D, nights_curves, map_beta, X):
    """Per-night data (band offsets + phase term removed with the MAP nuisance values)
    with posterior predictive model curves."""
    _style()
    night = D["night"]
    nn = night.max() + 1
    fig, axs = plt.subplots(1, nn, figsize=(2.6 * nn, 3.3), sharey=True)
    y = D["mag"] - X @ map_beta
    for k in range(nn):
        ax = axs[k]
        m = night == k
        t0 = np.floor(D["t"][m].min() - 0.5) + 0.5
        for tc, curves in nights_curves[k]:
            for cv in curves:
                ax.plot((tc - t0) * 24, cv, color=MUTE, lw=0.5, alpha=0.35)
        for bi, bl in enumerate(D["bands"]):
            mb = m & (D["band_idx"] == bi)
            ax.errorbar((D["t"][mb] - t0) * 24, y[mb], yerr=D["err"][mb], fmt="o", ms=2.5,
                        color=BAND_COL.get(bl, INK), label=bl, lw=0.6, mew=0)
        ax.set_xlabel(f"hours after JD {t0:.1f}")
        ax.set_title(f"night {k + 1}", fontsize=8)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("reduced mag − band offset − phase term")
    axs[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _silhouette(ax, verts, i, j, **kw):
    p = verts[:, [i, j]]
    h = ConvexHull(p)
    poly = np.r_[p[h.vertices], p[h.vertices[:1]]]
    ax.plot(poly[:, 0], poly[:, 1], **kw)


def align_equatorial(verts):
    """Rotate about z so the longest equatorial axis is along x (for display only)."""
    xy = verts[:, :2] - verts[:, :2].mean(axis=0)
    w, V = np.linalg.eigh(xy.T @ xy)
    ang = np.arctan2(V[1, -1], V[0, -1])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return verts @ R.T


def shapes(path, map_verts, draw_verts, ab, bc):
    """Shape figure, ``fig_shapes.png``.

    The maximum-likelihood shape and posterior draws in three projections, aligned on the longest
    equatorial axis, with histograms of the projected-area axis ratios.
    """
    _style()
    map_verts = align_equatorial(map_verts)
    draw_verts = [align_equatorial(v) for v in draw_verts]
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8, 0.8]))
    views = [(0, 1, "pole-on (x–y)"), (0, 2, "equator, along y (x–z)"), (1, 2, "equator, along x (y–z)")]
    draw_verts = [v for v in draw_verts if np.all(np.isfinite(v))]
    lim = 1.15 * np.abs(np.concatenate([map_verts] + list(draw_verts))).max()
    for ax, (i, j, title) in zip(axs[:3], views):
        for v in draw_verts:
            _silhouette(ax, v, i, j, color=MUTE, lw=0.5, alpha=0.5)
        _silhouette(ax, map_verts, i, j, color=ACC, lw=1.6)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    ab, bc = np.asarray(ab, float), np.asarray(bc, float)
    ab, bc = ab[np.isfinite(ab)], bc[np.isfinite(bc)]
    axs[3].hist(ab, bins=30, color=INK, alpha=0.85)
    axs[3].set_xlabel("a/b (projected-area, ellipsoid-equivalent)")
    axs[4].hist(bc, bins=30, color=INK, alpha=0.85)
    axs[4].set_xlabel("b/c (projected-area, ellipsoid-equivalent)")
    fig.suptitle("convex shape: red = max-likelihood spin state, grey = posterior draws "
                 "(spin state + Laplace shape draws), aligned on the longest equatorial axis; z = spin axis",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------- folded lightcurves
NIGHT_COL = ["#c4452f", "#2a6fb0", "#2a9d8f", "#8a5a2b", "#6c4ba3", "#d99a1e", "#444444", "#b0487a"]
BAND_MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]


def ref_band(bands):
    """Index of the r band (also 'Lr', 'X05:r', 'R', ...), else 0."""
    for i, b in enumerate(bands):
        if str(b).split(":")[-1].lower().lstrip("l") == "r":
            return i
    return 0


def reference_geometry(D):
    """Geometry of the data epoch closest to the median time: the folded model curves are
    computed at this fixed Sun/observer direction and phase angle."""
    i = int(np.argmin(np.abs(D["t"] - np.median(D["t"]))))
    return i, D["t"][i], D["e_sun"][i], D["e_obs"][i], float(D["alpha"][i])


def phase_grid(t_near, t_ref, P_h, n=400):
    """Epochs (days) covering the rotation that starts at a phase-0 time (t_ref + k P) close to
    t_near, and their phase in rotations. Returns (t_grid, s_grid, t0)."""
    Pd = P_h / 24.0
    t0 = t_ref + Pd * np.floor((t_near - t_ref) / Pd)
    s = np.linspace(0.0, 1.0, n, endpoint=False)
    return t0 + s * Pd, s, t0


def folded(path, D, P_h, t_c, resid, s_grid, curves, title, curve_labels=None, curve_cols=None,
           alpha_ref=None, n_rot=2):
    """Phase-folded lightcurve over n_rot rotations (default 0-4 pi).

    The data are shown as  model_ref(phase) + residual,  where model_ref is the model at the
    fixed reference geometry (Sun/observer directions of the median epoch) and the residual is
    data minus the model at each point's own geometry, offsets and phase term. Geometry
    changes between nights are therefore removed and the scatter about the curves is the
    actual misfit.

    resid: (N,) residuals [mag] of the reference fit; curves: list of (n_grid,) or a dict
    {label: list of (n_grid,)} of model curves at the reference geometry (the first curve of
    the first group is the reference fit and defines model_ref)."""
    _style()
    groups = curves if isinstance(curves, dict) else {"model": curves}
    labels = list(groups)
    cols = curve_cols or [MUTE] * len(labels)
    ref = np.asarray(groups[labels[0]][0])
    ph = ((D["t"] - t_c) / (P_h / 24.0)) % 1.0
    y = np.interp(ph, s_grid, ref, period=1.0) + resid
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.0, 4.9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    fig.subplots_adjust(left=0.1, right=0.8, top=0.88, bottom=0.11)
    xg = np.concatenate([s_grid + k for k in range(n_rot)] + [[n_rot]])
    for lab, col in zip(labels, cols):
        for j, cv in enumerate(groups[lab]):
            cv = np.asarray(cv)
            ax.plot(2 * xg, np.r_[np.tile(cv, n_rot), cv[0]], color=col, lw=0.6, alpha=0.3,
                    label=(lab if len(labels) > 1 and j == 0 else None))
    ax.plot(2 * xg, np.r_[np.tile(ref, n_rot), ref[0]], color=INK, lw=1.1, label="reference fit")
    nights = D["night"]
    for k in range(nights.max() + 1):
        for b in range(len(D["bands"])):
            m = (nights == k) & (D["band_idx"] == b)
            if not m.any():
                continue
            for r in range(n_rot):
                kw = dict(fmt=BAND_MARK[b % len(BAND_MARK)], ms=3, lw=0.5, mew=0, alpha=0.85,
                          color=NIGHT_COL[k % len(NIGHT_COL)])
                ax.errorbar(2 * (ph[m] + r), y[m], yerr=D["err"][m], **kw)
                axr.errorbar(2 * (ph[m] + r), 1000 * resid[m], yerr=1000 * D["err"][m], **kw)
    # legends: nights (colour) and bands (marker)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=4, color=NIGHT_COL[k % len(NIGHT_COL)], label=f"night {k + 1}")
         for k in range(nights.max() + 1)]
    h += [Line2D([], [], ls="", marker=BAND_MARK[b % len(BAND_MARK)], ms=4, color=INK, label=str(bl))
          for b, bl in enumerate(D["bands"])]
    leg1 = ax.legend(handles=h, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    if len(labels) > 1:
        ax.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    ax.invert_yaxis()
    ax.set_ylabel("reduced mag (offsets, phase term\nand geometry removed)", fontsize=8)
    rms = 1000 * np.sqrt(np.mean(resid ** 2))
    sub = f"P = {P_h:.7f} h ({P_h * 3600:.2f} s), phase 0 at JD {t_c:.5f} (light-time corrected)"
    if alpha_ref is not None:
        sub += f", curves at phase angle {alpha_ref:.1f}°"
    ax.set_title(f"{title}\n{sub}", fontsize=8.5)
    axr.axhline(0, color=INK, lw=0.6)
    axr.set_ylabel(f"O − C [mmag]\nrms {rms:.1f}", fontsize=8)
    axr.invert_yaxis()
    axr.set_xlim(0, 2 * n_rot)
    ticks = np.arange(0, 2 * n_rot + 0.01, 0.5)
    axr.set_xticks(ticks)
    axr.set_xticklabels(["0" if t == 0 else ("π" if t == 1 else f"{t:g}π") for t in ticks])
    axr.set_xlabel("rotational phase")
    fig.savefig(path, dpi=150)
    plt.close(fig)
