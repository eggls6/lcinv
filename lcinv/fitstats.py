"""Goodness of fit of a best-fit model: reduced chi^2 with an honest parameter count.

    chi2_nominal   = sum r_i^2 / e_i^2                 (quoted errors only)
    chi2_jitter    = sum r_i^2 / (e_i^2 + s_jit^2)     (errors inflated by the fitted jitter)
    dof            = N - p_eff
    chi2_red       = chi2 / dof

p_eff, the effective number of parameters, counts how many directions in parameter space the
data actually constrain. For a parameter block with Gaussian prior precision Lambda and
likelihood curvature H_lik at the optimum,

    p_eff = tr( H_lik (H_lik + Lambda)^-1 ),

which is the number of parameters for a weak prior and 0 for a prior that dominates (the
"effective degrees of freedom" of a regularised fit). The spin parameters (pole, period,
and c_L or tau0 when fitted) and the linear nuisance (band offsets, phase slope) are counted in
full; the jitter is a noise parameter and is not counted.

Reading: chi2_red(nominal) >> 1 means the quoted errors are too small or the model misses
signal (the jitter then absorbs it); chi2_red(jitter) ~ 1 by construction. Compare models by
the residual rms / jitter and by ln Z, not by chi2_red alone: a more flexible model always
lowers chi2.
"""
import numpy as np


def quality(resid, err, jitter, p_eff, p_nominal, n_spin, n_linear, note=None):
    """Goodness-of-fit summary of a best-fit model.

    Parameters
    ----------
    resid : array_like
        Residuals [mag].
    err : array_like
        Quoted uncertainties [mag].
    jitter : float
        Fitted jitter [mag].
    p_eff : float
        Effective number of parameters.
    p_nominal : int
        Nominal number of parameters.
    n_spin, n_linear : int
        Spin parameters and linear nuisance parameters included in `p_eff`.
    note : str, optional

    Returns
    -------
    dict
        ``n_points``, ``p_eff``, ``p_nominal``, ``n_spin``, ``n_linear``, ``dof``,
        ``chi2_nominal``, ``chi2_red_nominal``, ``chi2_jitter``, ``chi2_red_jitter``, ``rms_mag``,
        ``median_err_mag``, ``jitter_mag`` (and ``note``).
    """
    r = np.asarray(resid, float)
    e2 = np.asarray(err, float) ** 2
    n = len(r)
    dof = max(n - p_eff, 1.0)
    c_nom = float(np.sum(r ** 2 / e2))
    c_jit = float(np.sum(r ** 2 / (e2 + jitter ** 2)))
    out = dict(n_points=int(n), p_eff=float(p_eff), p_nominal=int(p_nominal), n_spin=int(n_spin),
               n_linear=int(n_linear), dof=float(dof),
               chi2_nominal=c_nom, chi2_red_nominal=c_nom / dof,
               chi2_jitter=c_jit, chi2_red_jitter=c_jit / dof,
               rms_mag=float(np.sqrt(np.mean(r ** 2))), median_err_mag=float(np.median(np.sqrt(e2))),
               jitter_mag=float(jitter))
    if note:
        out["note"] = note
    return out


def effective_block(H_lik, H_total):
    """Effective number of parameters of a block, ``tr(H_lik H_total^-1)``.

    Evaluated in the eigenbasis of `H_total`: direction i adds ``(v_i^T H_lik v_i) / lambda_i``,
    clipped to [0, 1]. A direction with non-positive total curvature (not exactly at the optimum)
    counts 1 if the data curve it, else 0.

    Parameters
    ----------
    H_lik : ndarray
        Hessian of the negative log-likelihood alone.
    H_total : ndarray
        Hessian of likelihood plus prior and penalties.

    Returns
    -------
    float
    """
    Ht = 0.5 * (H_total + H_total.T)
    Hl = 0.5 * (H_lik + H_lik.T)
    ev, V = np.linalg.eigh(Ht)
    hl = np.einsum("ji,jk,ki->i", V, Hl, V)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(ev > 0, hl / ev, np.where(hl > 0, 1.0, 0.0))
    return float(np.sum(np.clip(frac, 0.0, 1.0)))


def format_line(name, q):
    """One-line text summary of a `quality` dict, as printed by the scripts."""
    return (f"{name}: N = {q['n_points']}, p_eff = {q['p_eff']:.1f} (nominal {q['p_nominal']}), "
            f"chi2_red = {q['chi2_red_nominal']:.2f} (quoted errors) / {q['chi2_red_jitter']:.2f} (with jitter "
            f"{1000 * q['jitter_mag']:.1f} mmag), rms {1000 * q['rms_mag']:.1f} mmag")


def data_fingerprint(D):
    """Short hash of the fitted data (epochs, magnitudes, errors, bands).

    Evidences are only comparable between runs with the same fingerprint; `compare_models.py`
    excludes runs whose fingerprint differs.

    Parameters
    ----------
    D : dict
        Dictionary from `lcinv.setup_data.prepare` (after any clipping).

    Returns
    -------
    str
        12 hexadecimal characters.
    """
    import hashlib
    h = hashlib.sha1()
    for k in ("t", "mag", "err"):
        h.update(np.round(np.asarray(D[k], float), 7).tobytes())
    h.update(np.asarray(D["band_idx"], np.int64).tobytes())
    return h.hexdigest()[:12]
