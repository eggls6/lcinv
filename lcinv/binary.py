"""Contact-binary lightcurve model: two triaxial ellipsoid lobes in contact.

Body frame: the lobes lie along the body x axis with their long axes aligned, the spin axis
is body z (the maximum-inertia axis of this configuration when c <= b <= 1).

    lobe 1: semi-axes (1, b, c)        centre (-x1, 0, 0)
    lobe 2: semi-axes q (1, b, c)      centre (+x2, 0, 0)
    centre distance D = s (1 + q); x1 = D q^3 / (1 + q^3), x2 = D / (1 + q^3), so the
    (uniform-density) centre of mass is at the origin. s = 1 is touching, s < 1 overlapping.

Parameters theta (nested sampling, prior in BinaryPrior):
    [lambda, beta, P, ln_jitter, phi0, q, b, c/b, s] (+ ln c_L with fit_cl)
phi0 is needed here (the convex model absorbs it into the shape orientation).

Brightness. Each lobe is a convex triangulated ellipsoid, so a facet can only be hidden or
shadowed by the *other* lobe. For a facet centre p and a direction d (Sun or observer) the
ray p + t d (t >= 0) is tested analytically against the other ellipsoid: in that lobe's
scaled coordinates p' = (p - c_B) / A_B, d' = d / A_B the closest approach is
    m = |p' + t* d'|^2 - 1,  t* = max(0, -p'.d' / d'.d'),
and the facet counts as visible with weight sigmoid(m / soft). Facets inside the other
lobe (the neck) are removed with weight sigmoid((|p'|^2 - 1) / soft). The soft edges make
the likelihood a smooth function of the parameters. Scattering: Lommel-Seeliger + Lambert,
S = mu mu0 [1/(mu + mu0) + c_L], as in the convex and hybrid models.

Linear nuisance (band offsets + phase-angle slope) is profiled exactly as in the convex
model, so ln Z of this model and of run_inversion.py are directly comparable (same data,
same period / jitter priors, same treatment of the offsets).
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import ConvexHull

from lcinv import smear
from lcinv.convex import to_body

NAMES = ["lambda", "beta", "P", "ln_jitter", "phi0", "q", "b", "c_over_b", "s"]
CL_NAME = "ln_c_L"


def sphere_mesh(n):
    """Quasi-uniform unit vectors (Fibonacci) and an outward-oriented triangulation.

    Returns
    -------
    u : ndarray
        (n, 3) vertices.
    tri : ndarray of int
        Triangles.
    """
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    ph = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    u = np.c_[r * np.cos(ph), r * np.sin(ph), z]
    tri = ConvexHull(u).simplices.copy()
    a, b, c = u[tri[:, 0]], u[tri[:, 1]], u[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return u, tri


def geometry(q, b, c, s):
    """Lobe centres and semi-axes for shape parameters ``(q, b, c, s)``.

    Works on NumPy and JAX arrays.

    Returns
    -------
    centres : array
        (2, 3); the centre of mass is at the origin.
    semi_axes : array
        (2, 3): ``(1, b, c)`` and ``q (1, b, c)``.
    """
    xp = jnp if isinstance(q, jnp.ndarray) else np
    D = s * (1 + q)
    x1 = D * q ** 3 / (1 + q ** 3)
    x2 = D / (1 + q ** 3)
    z = 0.0 * q
    cen = xp.stack([xp.stack([-x1, z, z]), xp.stack([x2, z, z])])
    ax = xp.stack([xp.stack([1.0 + z, b, c]), q * xp.stack([1.0 + z, b, c])])
    return cen, ax


class ContactBinary:
    """Contact-binary model: two ellipsoid lobes with mutual occlusion and shadowing.

    Same interface as `lcinv.convex.ConvexInversion` (``loglike``, ``loglike_jit``,
    ``predict``, ``fit_quality``, ``init_kwargs``), so `lcinv.sampler` samples it unchanged.

    Parameters
    ----------
    t, mag, err, band_idx, alpha, e_sun, e_obs
        Data as for `lcinv.convex.ConvexInversion`.
    t_ref : float, optional
        Phase reference [JD]; default the mean epoch.
    nfacets : int, optional
        Facets per lobe.
    c_lambert : float, optional
        Fixed Lambert weight.
    fit_cl : bool, optional
        Sample ln c_L as an extra parameter.
    soft : float, optional
        Width of the smooth visibility edges (in the other lobe's scaled coordinates).
    exptime, smear_nodes, smear_P_h
        Exposure smearing, as for the convex model.
    cohesion : dict, optional
        Cohesion-limit penalty (`lcinv.cohesion.spec_from_args`).

    Notes
    -----
    ``theta = (lambda, beta, P, ln sigma_jit, phi0, q, b, c/b, s[, ln c_L])``. Unlike the
    convex model the rotational phase ``phi0`` [deg] is sampled, because the two-lobe shape fixes
    its own orientation. Shape: mass ratio ``q`` (size ratio of the lobes), axis ratios ``b`` and
    ``c/b`` of each lobe, separation ``s`` (1 = touching, < 1 overlapping).
    """

    def __init__(self, t, mag, err, band_idx, alpha, e_sun, e_obs, t_ref=None, nfacets=800,
                 c_lambert=0.1, fit_cl=False, soft=0.02, exptime=None, smear_nodes=None, smear_P_h=None,
                 cohesion=None):
        self.init_kwargs = dict(t=np.asarray(t), mag=np.asarray(mag), err=np.asarray(err),
                                band_idx=np.asarray(band_idx), alpha=np.asarray(alpha),
                                e_sun=np.asarray(e_sun), e_obs=np.asarray(e_obs), t_ref=t_ref,
                                nfacets=nfacets, c_lambert=c_lambert, fit_cl=fit_cl, soft=soft,
                                exptime=None if exptime is None else np.asarray(exptime),
                                smear_nodes=smear_nodes, smear_P_h=smear_P_h, cohesion=cohesion,
                                model_class="lcinv.binary.ContactBinary")
        self.cohesion = cohesion
        self.t = np.asarray(t, float)
        self.t_ref = float(np.mean(t) if t_ref is None else t_ref)
        self.mag = jnp.asarray(mag, float)
        self.err2 = jnp.asarray(err, float) ** 2
        self.nb = int(np.max(band_idx)) + 1
        self.X = jnp.asarray(np.c_[np.eye(self.nb)[band_idx], alpha], float)
        self.c_lambert = float(c_lambert)
        self.fit_cl = bool(fit_cl)
        self.ntheta = len(NAMES) + (1 if fit_cl else 0)
        self.soft = float(soft)
        u, tri = sphere_mesh(int(nfacets) // 2 + 2)
        self.u_np, self.tri_np = u, tri
        self.u = jnp.asarray(u)
        self.tri = jnp.asarray(tri)
        # exposure smearing (same scheme as the convex model)
        n = len(self.t)
        T = np.zeros(n) if exptime is None else np.nan_to_num(np.asarray(exptime, float))
        if smear_nodes is not None:
            self.Q = int(smear_nodes) if np.any(T > 0) else 1
        else:
            self.Q = smear.choose_nodes(T, smear_P_h, k_max=12) if (smear_P_h and np.any(T > 0)) else 1
        tq, wq = smear.node_times(self.t, T, self.Q)
        self.t_nodes = jnp.asarray(tq)
        self.w_nodes = jnp.asarray(wq)
        self.es_nodes = jnp.repeat(jnp.asarray(e_sun, float), self.Q, axis=0)
        self.eo_nodes = jnp.repeat(jnp.asarray(e_obs, float), self.Q, axis=0)
        self.loglike_jit = jax.jit(self._loglike_full)
        self._bright_jit = jax.jit(self._brightness, static_argnums=(5,))

    # ------------------------------------------------------------------ geometry
    def shape(self, theta):
        """Lobe centres and semi-axes for a parameter vector (see `geometry`)."""
        q, b, cb, s = theta[5], theta[6], theta[7], theta[8]
        return geometry(q, b, b * cb, s)

    def c_l(self, theta):
        """Lambert weight (fitted or fixed)."""
        return jnp.exp(theta[9]) if self.fit_cl else self.c_lambert

    def lobe_facets(self, cen, ax, k):
        """Facet centres, outward normals and areas of lobe `k` (0 or 1)."""
        v = self.u * ax[k] + cen[k]
        a, b_, c = v[self.tri[:, 0]], v[self.tri[:, 1]], v[self.tri[:, 2]]
        cr = jnp.cross(b_ - a, c - a)
        area = 0.5 * jnp.linalg.norm(cr, axis=1)
        nrm = cr / (2 * area[:, None])
        return (a + b_ + c) / 3, nrm, area

    def _unoccluded(self, p, d, cB, aB):
        """(N,F) soft weight that rays p + t d miss ellipsoid (cB, aB). p (F,3), d (N,3)."""
        ps = (p - cB) / aB                     # (F,3)
        ds = d / aB                            # (N,3)
        pp = jnp.sum(ps * ps, 1)[None, :]
        pd = ds @ ps.T                         # (N,F)
        dd = jnp.sum(ds * ds, 1)[:, None]
        t = jnp.maximum(0.0, -pd / dd)
        m = pp + 2 * t * pd + t * t * dd - 1.0
        return jax.nn.sigmoid(m / self.soft)

    def _brightness(self, theta, t, es, eo, w, n):
        lam, bet, P, phi0 = theta[0], theta[1], theta[2], theta[4]
        cen, ax = self.shape(theta)
        es_b = to_body(es, lam, bet, P, jnp.deg2rad(phi0), t, self.t_ref)
        eo_b = to_body(eo, lam, bet, P, jnp.deg2rad(phi0), t, self.t_ref)
        cl = self.c_l(theta)
        L = 0.0
        for k in (0, 1):
            o = 1 - k
            p, nrm, area = self.lobe_facets(cen, ax, k)
            outside = jax.nn.sigmoid((jnp.sum(((p - cen[o]) / ax[o]) ** 2, 1) - 1.0) / self.soft)
            mu0 = jnp.maximum(es_b @ nrm.T, 0.0)
            mu = jnp.maximum(eo_b @ nrm.T, 0.0)
            vis = self._unoccluded(p, es_b, cen[o], ax[o]) * self._unoccluded(p, eo_b, cen[o], ax[o])
            S = mu * mu0 * (1.0 / (mu + mu0 + 1e-12) + cl) * vis
            L = L + S @ (area * outside)
        return L.reshape(n, -1) @ w

    # ------------------------------------------------------------------ likelihood
    def _loglike_full(self, theta):
        sig2 = self.err2 + jnp.exp(2 * theta[3])
        L = self._brightness(theta, self.t_nodes, self.es_nodes, self.eo_nodes, self.w_nodes, len(self.t))
        m0 = -2.5 * jnp.log10(jnp.maximum(L, 1e-30))
        W = 1.0 / sig2
        XtW = self.X.T * W
        beta = jnp.linalg.solve(XtW @ self.X, XtW @ (self.mag - m0))
        model = m0 + self.X @ beta
        r = self.mag - model
        chi2 = jnp.sum(r * r / sig2)
        logL = -0.5 * jnp.sum(r * r / sig2 + jnp.log(2 * jnp.pi * sig2))
        if self.cohesion is not None:
            from lcinv import cohesion as _co
            pen = _co.penalty_jax(self.required_cohesion_jax(theta), self.cohesion["sigma_max"],
                                  self.cohesion["softness"])
            logL = logL - pen
        return dict(logL=logL, chi2=chi2, model=model, beta=beta)

    def required_cohesion_jax(self, theta, spec=None):
        """Cohesion (Pa) needed by the equivalent ellipsoid of the two-lobe body (moments of
        inertia of both lobes, overlap ignored), Sanchez & Scheeres (2014) eq. 23. The neck of a
        real contact binary can need more; this is the bulk (centre) criterion."""
        from lcinv import cohesion as _co
        spec = spec or self.cohesion
        cen, ax = self.shape(theta)
        m = ax[:, 0] * ax[:, 1] * ax[:, 2]
        x = cen[:, 0]
        Ixx = jnp.sum(m * (ax[:, 1] ** 2 + ax[:, 2] ** 2) / 5)
        Iyy = jnp.sum(m * ((ax[:, 0] ** 2 + ax[:, 2] ** 2) / 5 + x ** 2))
        Izz = jnp.sum(m * ((ax[:, 0] ** 2 + ax[:, 1] ** 2) / 5 + x ** 2))
        A, B, C = _co.ellipsoid_from_moments_jax(Ixx, Iyy, Izz, spec["diameter_km"])
        return jnp.clip(_co.required_cohesion_jax(A, B, C, theta[2], spec["density"], spec["friction_deg"]), 0.0, None)

    def fit_quality(self, theta):
        """Reduced chi^2 at a parameter vector (all shape and spin parameters counted)."""
        from lcinv import fitstats
        th = jnp.asarray(theta, float)
        out = self.loglike_jit(th)
        r = np.asarray(self.mag - out["model"])
        n_spin = 4 + (1 if self.fit_cl else 0)     # pole, period, phi0 (+ c_L)
        n_lin = self.nb + 1
        p = n_spin + n_lin + 4                     # + q, b, c/b, s
        return fitstats.quality(r, np.sqrt(np.asarray(self.err2)), float(np.exp(theta[3])), p, p, n_spin, n_lin,
                                note="parameter count: all lobe and spin parameters (no shrinkage prior)")

    def loglike(self, theta):
        """Log-likelihood of a parameter vector (offsets and phase slope profiled); -1e300 if not
        finite.
        """
        v = float(self.loglike_jit(jnp.asarray(theta, float))["logL"])
        return v if np.isfinite(v) else -1e300

    def warmup_theta(self):
        """A valid parameter vector for compiling the likelihood."""
        th = [0.0, 0.0, 3.0, -4.0, 0.0, 0.8, 0.8, 0.9, 0.9]
        return np.array(th + ([np.log(0.1)] if self.fit_cl else []))

    def predict(self, theta, t, e_sun, e_obs, band_idx, alpha, beta, exptime=None):
        """Model reduced magnitudes at arbitrary epochs and geometries (for plotting).

        Parameters as `lcinv.convex.ConvexInversion.predict`, without the shape coefficients (the shape
        is part of `theta`).
        """
        t = np.asarray(t, float)
        n = len(t)
        T = np.zeros(n) if exptime is None else np.broadcast_to(np.asarray(exptime, float), (n,))
        Q = self.Q if exptime is not None else 1
        tq, wq = smear.node_times(t, T, Q)
        L = self._bright_jit(jnp.asarray(theta, float), jnp.asarray(tq),
                             jnp.repeat(jnp.asarray(e_sun, float), Q, axis=0),
                             jnp.repeat(jnp.asarray(e_obs, float), Q, axis=0), jnp.asarray(wq), n)
        X = np.c_[np.eye(self.nb)[band_idx], alpha]
        return np.asarray(-2.5 * jnp.log10(L)) + X @ np.asarray(beta)

    # ------------------------------------------------------------------ shape export
    def union_mesh(self, theta, n=4000):
        """Watertight star-shaped mesh of the two-lobe union (the union of two convex sets
        that both contain a common point x0 is star-shaped about x0). Vertices are centred
        on the centre of mass and scaled to unit volume-equivalent radius."""
        cen, ax = (np.asarray(a) for a in self.shape(np.asarray(theta, float)))
        lo = cen[1, 0] - ax[1, 0]                  # left end of lobe 2
        hi = cen[0, 0] + ax[0, 0]                  # right end of lobe 1
        x0 = np.array([0.5 * (lo + hi), 0.0, 0.0])  # inside both lobes when s < 1
        u, tri = sphere_mesh(n)
        r = np.zeros(len(u))
        for k in (0, 1):
            ps, ds = (x0 - cen[k]) / ax[k], u / ax[k]
            A = np.sum(ds * ds, 1)
            B = 2 * ds @ ps
            C = ps @ ps - 1
            disc = np.clip(B * B - 4 * A * C, 0, None)
            r = np.maximum(r, (-B + np.sqrt(disc)) / (2 * A))
        v = x0 + u * r[:, None]
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        vol = np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
        v = v / (3 * abs(vol) / (4 * np.pi)) ** (1 / 3)
        return v, tri


class BinaryPrior:
    """Prior transform of the contact-binary model.

    Pole uniform on the sphere, period as in `lcinv.sampler.PriorTransform`, ln jitter uniform,
    ``phi0`` uniform in [0, 360) deg, shape parameters uniform in the given ranges.

    Parameters
    ----------
    period : tuple
        As for `lcinv.sampler.PriorTransform`.
    ln_jit_range : (float, float), optional
    q_range, b_range, cb_range, s_range : (float, float), optional
        Ranges of q, b, c/b and s (``--q-range`` etc. of ``run_binary.py``).
    ln_cl_range : (float, float), optional
        Range of ln c_L, or None.
    """

    def __init__(self, period, ln_jit_range=(np.log(1e-3), np.log(0.1)), q_range=(0.3, 1.0),
                 b_range=(0.5, 1.0), cb_range=(0.5, 1.0), s_range=(0.70, 0.98), ln_cl_range=None):
        from lcinv.sampler import PriorTransform
        self._pt = PriorTransform(period, ln_jit_range=ln_jit_range)
        self.jit = ln_jit_range
        self.ranges = [q_range, b_range, cb_range, s_range]
        self.cl_range = ln_cl_range
        self.ndim = len(NAMES) + (1 if ln_cl_range is not None else 0)
        self.periodic = [0, 4]                     # lambda and phi0 wrap around

    def __call__(self, u):
        """`theta` at unit-cube point `u`."""
        lo, hi = self.jit
        th = [360.0 * u[0], np.degrees(np.arcsin(2 * u[1] - 1)), self._pt.P(u[2]),
              lo + (hi - lo) * u[3], 360.0 * u[4]]
        th += [a + (b - a) * x for (a, b), x in zip(self.ranges, u[5:9])]
        if self.cl_range is not None:
            a, b = self.cl_range
            th.append(a + (b - a) * u[9])
        return np.array(th)

    def describe(self):
        """The priors as a dict, for ``summary.json``."""
        d = dict(pole="uniform on sphere", phi0="uniform [0, 360) deg",
                 ln_jitter=f"uniform in [{self.jit[0]:.3f}, {self.jit[1]:.3f}]",
                 q=list(self.ranges[0]), b=list(self.ranges[1]), c_over_b=list(self.ranges[2]),
                 s=list(self.ranges[3]))
        if self.cl_range is not None:
            d["ln_c_L"] = list(self.cl_range)
        return d


def param_names(fit_cl=False):
    """Names of the sampled parameters, in order."""
    return NAMES + ([CL_NAME] if fit_cl else [])
