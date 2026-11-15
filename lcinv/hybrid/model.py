"""Local (per-mode) joint model of spin and radius-field shape (JAX log density).

Parameter vector x (unconstrained, length 6 + k):
    x[0:2]  d1, d2   gnomonic tangent-plane coordinates of the pole around the mode centre p0
    x[2]    zP       standardised period offset,  P = P0 + sP * zP
    x[3]    ln sigma_jit
    x[4]    ln sigma_f     (shape hyperparameter: pointwise sd of log r)
    x[5]    ln ell         (shape hyperparameter: angular correlation scale, in degree units)
    x[6]    ln c_L         (only with fit_cl=True: Lambert weight of the scattering law)
    x[NSPIN:] z_lm         (non-centred shape coefficients; NSPIN = 6, or 7 with fit_cl)

Body frame: by default built with quaternions by parallel transport from the mode's reference
pole (lcinv.hybrid.rotations.ParallelFrame), so pole changes never twist the frame about the
spin axis; frame="euler" keeps the legacy Rz(phi) Ry(90 - beta) Rz(lambda) construction,
whose twist couples to lambda near the ecliptic poles.

Pole: p = normalize(p0 + d1 e1 + d2 e2). The gnomonic chart is singularity-free around p0
(no coordinate pole at |beta| = 90 deg) and covers the hemisphere centred on p0, so a chain
cannot wander into the mirror mode. A uniform prior on the sphere becomes
    ln pi(d) = -ln(4 pi) - 3/2 ln(1 + d1^2 + d2^2).

Rotational phase: gauge-fixed, phi0 = 0 at t_ref (rotations of the body about the spin axis
are absorbed by the m != 0 coefficients; the prior is rotation invariant).

Likelihood: Gaussian with sigma_i^2 = e_i^2 + sigma_jit^2; band offsets and the linear phase
slope eta are marginalised analytically under a flat prior:
    ln p(y|x) = -1/2 r^T W r - 1/2 sum ln(2 pi sigma_i^2) - 1/2 ln det(X^T W X) + p/2 ln 2 pi,
with r the residual at the weighted least-squares eta.

Limb/terminator: mu -> T softplus(mu / T) with T = 0.005 keeps the likelihood smooth for NUTS.

Exposure smearing (lcinv.smear): with data["exptime"] (s) the facet sum is evaluated at Q
Gauss-Legendre nodes within each exposure and the flux is averaged before converting to
magnitudes. Facet normals and areas depend only on the shape, so the cost grows by Q only in
the scattering step.
"""
import numpy as np
import jax
import jax.numpy as jnp

from lcinv import smear
from lcinv.convex import to_body
from lcinv.hybrid.rotations import ParallelFrame

jax.config.update("jax_enable_x64", True)


def tangent_basis(p0):
    """Orthonormal tangent vectors ``(p0, e1, e2)`` at pole `p0` for the gnomonic pole chart."""
    p0 = np.asarray(p0, float) / np.linalg.norm(p0)
    a = np.array([0.0, 0.0, 1.0]) if abs(p0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, p0)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(p0, e1)
    return p0, e1, e2


def lonlat_to_vec(lam_deg, bet_deg):
    """Ecliptic unit vector of longitude and latitude in degrees."""
    l, b = np.radians(lam_deg), np.radians(bet_deg)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


class LocalModel:
    """Joint log density of spin and radius-field shape within one spin state.

    The parameter vector ``x`` is described in the module docstring; `logdensity` and `grad`
    are jitted and used by the MAP search, the Laplace evidence and NUTS.

    Parameters
    ----------
    data : dict
        Dictionary from `lcinv.setup_data.prepare`.
    field : RadiusField
        Shape model and its priors.
    pole0 : array_like
        Centre of the spin state (ecliptic unit vector); the pole chart is centred here.
    P0, sP : float
        Centre and width of the Gaussian period prior [h].
    t_ref : float, optional
        Phase reference [JD]; default the mean epoch.
    c_lambert : float, optional
        Fixed Lambert weight.
    softplus_T : float, optional
        Smoothing of the limb and terminator, ``mu -> T softplus(mu / T)``.
    ln_jit_prior : (float, float), optional
        Mean and sd of the normal prior on ln sigma_jit.
    frame : {'parallel', 'euler'}, optional
        Body frame: quaternion parallel transport (default; no twist coupling to the pole) or
        the legacy Euler construction.
    smear_nodes : int, optional
        Quadrature nodes per exposure; default chosen automatically.
    fit_cl : bool, optional
        Sample ln c_L (``x[6]``).
    ln_cl_prior : (float, float), optional
        Mean and sd of the normal prior on ln c_L.
    cohesion : dict, optional
        Cohesion-limit penalty.

    Attributes
    ----------
    ndim : int
        Length of ``x``: ``NSPIN + k``.
    NSPIN : int
        Number of non-shape parameters (6, or 7 with `fit_cl`).
    k : int
        Number of shape coefficients.
    logdensity, grad : callable
        Jitted log posterior density and its gradient.
    """
    NSPIN = 6

    def __init__(self, data, field, pole0, P0, sP, t_ref=None, c_lambert=0.1, softplus_T=0.005,
                 ln_jit_prior=(np.log(0.015), 1.0), frame="parallel", smear_nodes=None,
                 fit_cl=False, ln_cl_prior=(np.log(0.3), 2.0), cohesion=None):
        self.F = field
        self.cohesion = cohesion     # None or dict (lcinv.cohesion.spec_from_args)
        self.k = field.k
        self.fit_cl = bool(fit_cl)
        self.NSPIN = LocalModel.NSPIN + (1 if self.fit_cl else 0)
        self.cl_mu, self.cl_sd = ln_cl_prior
        self.ndim = self.NSPIN + self.k
        self.t = jnp.asarray(data["t"], float)
        self.t_ref = float(np.mean(data["t"]) if t_ref is None else t_ref)
        self.y = jnp.asarray(data["mag"], float)
        self.e2 = jnp.asarray(data["err"], float) ** 2
        nb = int(np.max(data["band_idx"])) + 1
        self.X = jnp.asarray(np.c_[np.eye(nb)[data["band_idx"]], data["alpha"]], float)
        self.p = self.X.shape[1]
        self.e_sun = jnp.asarray(data["e_sun"], float)
        self.e_obs = jnp.asarray(data["e_obs"], float)
        n = len(np.asarray(data["t"]))
        ex = data.get("exptime") if hasattr(data, "get") else None
        self.exptime = np.zeros(n) if ex is None else np.nan_to_num(np.asarray(ex, float))
        self.Q = (int(smear_nodes) if smear_nodes is not None else
                  smear.choose_nodes(self.exptime, float(P0), smear.k_max_for_lmax(field.g.lmax)))
        tq, wq = smear.node_times(np.asarray(data["t"], float), self.exptime, self.Q)
        self.t_nodes, self.w_nodes = jnp.asarray(tq), jnp.asarray(wq)
        self.e_sun_nodes = jnp.repeat(self.e_sun, self.Q, axis=0)
        self.e_obs_nodes = jnp.repeat(self.e_obs, self.Q, axis=0)
        self.p0, self.e1, self.e2v = [jnp.asarray(v) for v in tangent_basis(pole0)]
        if frame not in ("parallel", "euler"):
            raise ValueError(frame)
        self.frame = frame
        self.pframe = ParallelFrame(np.asarray(self.p0))
        self.P0, self.sP = float(P0), float(sP)
        self.cL = c_lambert
        self.T = softplus_T
        self.lj_mu, self.lj_sd = ln_jit_prior
        self.logdensity = jax.jit(self._logdensity)
        self.grad = jax.jit(jax.grad(self._logdensity))

    # ------------------------------------------------------------------ unpacking
    def pole_vector(self, x):
        """Pole unit vector at chart coordinates ``x[0:2]``."""
        pv = self.p0 + x[0] * self.e1 + x[1] * self.e2v
        return pv / jnp.linalg.norm(pv)

    def body_vectors(self, x, e, t):
        """Ecliptic unit vectors -> body frame. 'parallel': quaternion parallel-transport frame
        (default, no twist coupling to the pole); 'euler': legacy Rz(phi) Ry(90-beta) Rz(lambda)."""
        P = self.P0 + self.sP * x[2]
        if self.frame == "parallel":
            return self.pframe.to_body(e, self.pole_vector(x), P, t, self.t_ref)
        lam, bet, _ = self.spin(x)
        return to_body(e, lam, bet, P, 0.0, t, self.t_ref)

    def spin(self, x):
        """``(lambda [deg], beta [deg], P [h])`` of a parameter vector."""
        pv = self.pole_vector(x)
        lam = jnp.degrees(jnp.arctan2(pv[1], pv[0])) % 360.0
        bet = jnp.degrees(jnp.arcsin(jnp.clip(pv[2], -1, 1)))
        P = self.P0 + self.sP * x[2]
        return lam, bet, P

    def c_l(self, x):
        """Lambert weight: fitted (x[6]) or the fixed c_lambert."""
        return jnp.exp(x[6]) if self.fit_cl else self.cL

    def coeffs(self, x):
        """Radius-field coefficients ``a_lm`` from the non-centred ``z_lm`` and the
        hyperparameters.
        """
        return self.F.coeffs(x[self.NSPIN:], x[4], x[5])

    # ------------------------------------------------------------------ forward model
    def brightness(self, x, t=None, e_sun=None, e_obs=None, exptime=None):
        """Disk-integrated brightness. Without t: at the data epochs, averaged over each
        exposure. With t: instantaneous, or exposure-averaged if exptime (s) is given."""
        if t is None:
            n, tq, w = self.t.shape[0], self.t_nodes, self.w_nodes
            esq, eoq = self.e_sun_nodes, self.e_obs_nodes
        else:
            t = np.asarray(t, float)
            n = len(t)
            Q = 1 if exptime is None else self.Q
            T = np.zeros(n) if exptime is None else np.broadcast_to(np.asarray(exptime, float), (n,))
            tq_np, w_np = smear.node_times(t, T, Q)
            tq, w = jnp.asarray(tq_np), jnp.asarray(w_np)
            esq = jnp.repeat(jnp.asarray(e_sun, float), Q, axis=0)
            eoq = jnp.repeat(jnp.asarray(e_obs, float), Q, axis=0)
        a = self.coeffs(x)
        _, area, nrm = self.F.mesh(a)
        es = self.body_vectors(x, esq, tq)
        eo = self.body_vectors(x, eoq, tq)
        sp = lambda m: self.T * jax.nn.softplus(m / self.T)
        mu0 = sp(es @ nrm.T)
        mu = sp(eo @ nrm.T)
        S = mu * mu0 * (1.0 / (mu + mu0) + self.c_l(x))
        L = S @ area
        return L.reshape(n, -1) @ w

    def _gls(self, ystar, sig2):
        W = 1.0 / sig2
        XtW = self.X.T * W
        A = XtW @ self.X
        eta = jnp.linalg.solve(A, XtW @ ystar)
        return eta, A

    def loglike_terms(self, x):
        """Pointwise log-likelihood (at the profiled eta) and the marginalisation correction."""
        sig2 = self.e2 + jnp.exp(2 * x[3])
        L = self.brightness(x)
        ystar = self.y + 2.5 * jnp.log10(L)
        eta, A = self._gls(ystar, sig2)
        r = ystar - self.X @ eta
        point = -0.5 * (r ** 2 / sig2 + jnp.log(2 * jnp.pi * sig2))
        corr = -0.5 * jnp.linalg.slogdet(A)[1] + 0.5 * self.p * jnp.log(2 * jnp.pi)
        return point, corr, eta, r

    def logprior(self, x):
        """Log prior: uniform pole (through the chart Jacobian), normal period offset, jitter,
        hyperparameter, c_L and standard-normal shape priors.
        """
        rho2 = x[0] ** 2 + x[1] ** 2
        lp = -jnp.log(4 * jnp.pi) - 1.5 * jnp.log1p(rho2)                       # uniform pole
        lp += -0.5 * x[2] ** 2 - 0.5 * jnp.log(2 * jnp.pi)                        # period
        lp += -0.5 * ((x[3] - self.lj_mu) / self.lj_sd) ** 2 - jnp.log(self.lj_sd * jnp.sqrt(2 * jnp.pi))
        lp += self.F.hyper_logprior(x[4], x[5])
        if self.fit_cl:
            lp += -0.5 * ((x[6] - self.cl_mu) / self.cl_sd) ** 2 - jnp.log(self.cl_sd * jnp.sqrt(2 * jnp.pi))
        z = x[self.NSPIN:]
        lp += -0.5 * jnp.sum(z ** 2) - 0.5 * self.k * jnp.log(2 * jnp.pi)
        return lp

    def physical_logprior(self, x):
        """Physical log prior: principal-axis rotation, weak convexity and, if set, the cohesion
        limit.
        """
        a = self.coeffs(x)
        lp = self.F.inertia_logprior(a) + self.F.convexity_logprior(a)
        if self.cohesion is not None:
            from lcinv import cohesion as _co
            lp = lp - _co.penalty_jax(self.required_cohesion_jax(x), self.cohesion["sigma_max"],
                                      self.cohesion["softness"])
        return lp

    def required_cohesion_jax(self, x, spec=None):
        """Cohesion (Pa) needed by the equivalent ellipsoid of the shape (principal moments of
        inertia), Sanchez & Scheeres (2014) eq. 23."""
        from lcinv import cohesion as _co
        spec = spec or self.cohesion
        _, _, I = self.F.mass_properties(self.coeffs(x))
        w = jnp.linalg.eigvalsh(0.5 * (I + I.T))
        A, B, C = _co.ellipsoid_from_moments_jax(w[0], w[1], w[2], spec["diameter_km"])
        P = self.P0 + self.sP * x[2]
        return jnp.clip(_co.required_cohesion_jax(A, B, C, P, spec["density"], spec["friction_deg"]), 0.0, None)

    def fit_quality(self, x):
        """Reduced chi^2 at x (normally the MAP) with p_eff = tr(H_lik H_post^-1) over all
        sampled parameters (minus the jitter, a noise parameter) plus the analytically
        marginalised band offsets and phase slope (lcinv.fitstats)."""
        from lcinv import fitstats
        x = jnp.asarray(x, float)
        point, corr, eta, r = self.loglike_terms(x)

        def ll(xx):
            pt, cr, _, _ = self.loglike_terms(xx)
            return jnp.sum(pt) + cr

        H_lik = -np.asarray(jax.hessian(ll)(x))
        H_tot = -np.asarray(jax.hessian(self._logdensity)(x))
        p_all = fitstats.effective_block(H_lik, H_tot)
        n_spin = 3 + (1 if self.fit_cl else 0)
        p_eff = max(p_all - 1.0, 0.0) + self.p
        q = fitstats.quality(np.asarray(r), np.sqrt(np.asarray(self.e2)), float(np.exp(x[3])), p_eff,
                             n_spin + self.p + self.k + 2, n_spin, self.p)
        q["k_shape"] = int(self.k)
        q["lmax"] = int(self.F.g.lmax)
        q["offset_marginalisation_term"] = float(corr)
        return q

    def _logdensity(self, x):
        point, corr, _, _ = self.loglike_terms(x)
        return jnp.sum(point) + corr + self.logprior(x) + self.physical_logprior(x)

    # ------------------------------------------------------------------ initial points
    def initial_points(self, n_starts=4, ln_sigma=np.log(0.15), ln_ell=np.log(3.0), amp=0.25, c_l0=None):
        """Starting points with the ellipsoid-like l = 2, m = +-2 elongation at different angles."""
        tau = np.asarray(self.F.tau(ln_sigma, ln_ell))
        l, m = self.F.g.l, self.F.g.m
        i22 = int(np.where((l == 2) & (m == 2))[0][0])
        i2m2 = int(np.where((l == 2) & (m == -2))[0][0])
        xs = []
        for psi in np.linspace(0, np.pi, n_starts, endpoint=False):
            x = np.zeros(self.ndim)
            x[3], x[4], x[5] = np.log(0.015), ln_sigma, ln_ell
            if self.fit_cl:
                x[6] = np.log(c_l0) if c_l0 else self.cl_mu
            x[self.NSPIN + i22] = -amp * np.cos(2 * psi) / tau[i22]
            x[self.NSPIN + i2m2] = -amp * np.sin(2 * psi) / tau[i2m2]
            xs.append(x)
        return np.array(xs)
