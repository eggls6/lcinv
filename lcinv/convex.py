"""JAX convex-shape forward model and likelihood.

Shape
-----
The convex body is represented by its Gaussian image (Kaasalainen & Torppa 2001):
a fixed set of N outward facet normals n_i (Fibonacci sphere) with areas
    A_i = w_i exp( sum_{l=1..lmax, m} c_lm Y_lm(n_i) ),
which are positive by construction. Closure (sum A_i n_i = 0), required for the areas to
belong to a closed convex body, is enforced by a penalty (the "dark facet" term).

Brightness
----------
Lommel-Seeliger + Lambert scattering, S = mu mu0 [1/(mu + mu0) + c_L], summed over facets
that are both lit and visible (exact for a convex body: no shadowing).
Exposure smearing (lcinv.smear): with exposure times given, S is averaged over Q
Gauss-Legendre nodes in time within each exposure. Brightness is linear in the facet areas,
so only the scattering matrix changes (S_bar = sum_q w_q S(t_iq)); the shape optimisation and
the Laplace step are unchanged.
Reduced magnitude model:  m_i = -2.5 log10 L_i + o_band(i) + s alpha_i
The band offsets o_b and the linear phase slope s are profiled analytically
(weighted least squares) inside the likelihood.

Spin state
----------
theta = (lambda, beta, P, ln sigma_jit[, ln c_L]): ecliptic J2000 pole (deg), sidereal
period (h), extra white noise (mag) added in quadrature to the quoted errors, and optionally
(fit_cl=True) the log of the Lambert weight c_L of the scattering law. S is linear in c_L,
S = S_LS + c_L S_L, so fitting it costs one extra matrix per likelihood call; the shape
optimisation is unchanged.
Body frame: r_body = Rz(phi(t)) Ry(90 deg - beta) Rz(lambda) r_ecl (passive rotations),
phi(t) = phi0 + 2 pi (t - t_ref) / P.
The rotational phase phi0 is a gauge freedom for a free shape: rotating the body about the
spin axis by delta is identical to shifting phi0 by delta, and the shape prior is rotation
invariant. phi0 is therefore fixed to 0 at t_ref and the shape orientation absorbs it.

Likelihood used for nested sampling
-----------------------------------
For each theta the shape coefficients c are optimised (L-BFGS, jitted) and marginalised
with a Laplace approximation:
    ln L(theta) = -f(c_hat) + k/2 ln(2 pi) - 1/2 ln det H(c_hat),
    f(c) = -ln p(data | theta, c) - ln p(c) + penalties,
with a Gaussian prior on c (scale tau_l = tau0 / l) and optional priors on the
axis ratios a/b and b/c derived from projected areas of the shape.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.special import sph_harm_y

from lcinv import smear

jax.config.update("jax_enable_x64", True)

LN10_04 = 0.4 * np.log(10.0)


# ----------------------------------------------------------------------------- geometry helpers
def fibonacci_sphere(n):
    """``n`` nearly uniformly spaced unit vectors (Fibonacci lattice): the facet normals of the
    Gaussian image, shape (n, 3).
    """
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    rho = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[rho * np.cos(phi), rho * np.sin(phi), z]


def real_sph_harm_basis(normals, lmax):
    """Real orthonormal spherical harmonics at given directions.

    Parameters
    ----------
    normals : ndarray
        Unit vectors, shape (N, 3).
    lmax : int
        Maximum degree; degrees 1 to `lmax` are included (l = 0 is the overall scale, fixed by
        the volume normalisation).

    Returns
    -------
    Y : ndarray
        Shape (N, (lmax + 1)^2 - 1).
    ls : ndarray of int
        Degree l of each column.
    """
    theta = np.arccos(np.clip(normals[:, 2], -1, 1))
    phi = np.arctan2(normals[:, 1], normals[:, 0])
    cols, ls = [], []
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m > 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.real)
            elif m < 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.imag)
            else:
                cols.append(Y.real)
            ls.append(l)
    return np.stack(cols, axis=1), np.array(ls)


def _rz(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def to_body(e, lam_deg, bet_deg, P_h, phi0, t, t_ref):
    """Rotate ecliptic unit vectors into the rotating body frame.

    ``r_body = Rz(phi(t)) Ry(90 deg - beta) Rz(lambda) r_ecl`` (passive rotations), with
    ``phi(t) = phi0 + 2 pi (t - t_ref) / P``.

    Parameters
    ----------
    e : array_like
        Ecliptic unit vectors, shape (N, 3).
    lam_deg, bet_deg : float
        Pole longitude and latitude [deg].
    P_h : float
        Sidereal period [h].
    phi0 : float
        Rotational phase at `t_ref` [rad].
    t : array_like
        Times [JD].
    t_ref : float
        Phase reference [JD].

    Returns
    -------
    jax.Array
        Body-frame vectors, shape (N, 3); z is the spin axis.
    """
    R = _ry(jnp.deg2rad(90.0 - bet_deg)) @ _rz(jnp.deg2rad(lam_deg))
    v = e @ R.T
    ph = phi0 + 2.0 * jnp.pi * (t - t_ref) * 24.0 / P_h
    c, s = jnp.cos(ph), jnp.sin(ph)
    return jnp.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)


# ----------------------------------------------------------------------------- model class
class ConvexInversion:
    """Convex-shape model and Laplace-marginalised likelihood of the global layer.

    Bundles the data, the Gaussian-image shape basis, the priors and the jitted likelihood.
    ``loglike(theta)`` is what `lcinv.sampler.run_nested` samples.

    Parameters
    ----------
    t : array_like
        Emission times [JD] (``D["t"]`` of `lcinv.setup_data.prepare`).
    mag : array_like
        Reduced magnitudes.
    err : array_like
        Quoted uncertainties [mag].
    band_idx : array_like of int
        Photometric group of each point; each group has its own offset.
    alpha : array_like
        Phase angles [deg].
    e_sun, e_obs : array_like
        Unit vectors from the asteroid to the Sun and to the observer (ecliptic J2000), (N, 3).
    lmax : int, optional
        Spherical-harmonic degree of the log-area expansion (``--lmax``).
    nfacets : int, optional
        Number of Gaussian-image facets (``--nfacets``).
    c_lambert : float, optional
        Lambert weight c_L of the scattering law, when not fitted (``--c-lambert``).
    tau0 : float, optional
        Prior scale of the shape coefficients, ``tau_l = tau0 / l`` (``--tau0``).
    closure_weight : float, optional
        Weight of the closure ("dark facet") penalty.
    ab_prior, bc_prior : tuple or None, optional
        ``("lognormal", value, sigma_ln)`` or ``("lower", value, sigma_ln)`` priors on the
        projected-area axis ratios a/b and b/c (``--ab-prior``, ``--bc-prior``).
    n_inner : int, optional
        L-BFGS iterations of the full shape optimisation per likelihood call (``--n-inner``).
    t_ref : float, optional
        Phase reference [JD]; default the mean epoch. The phase is gauge-fixed to 0 here.
    n_starts : int, optional
        Starts of the coarse l <= 2 stage, with the elongation at different angles.
    n_stage1 : int, optional
        L-BFGS iterations of the coarse stage.
    exptime : array_like, optional
        Exposure times [s] for smearing (0: instantaneous).
    smear_nodes : int, optional
        Gauss-Legendre nodes per exposure; default chosen from `smear_P_h`, the exposure times
        and `lmax` (`lcinv.smear.choose_nodes`).
    smear_P_h : float, optional
        Period [h] used to choose the nodes.
    fit_cl : bool, optional
        Sample ln c_L as the fifth spin parameter.
    cohesion : dict, optional
        Cohesion-limit penalty (`lcinv.cohesion.spec_from_args`).
    fit_tau : bool, optional
        Sample ln tau0 as the last spin parameter.

    Attributes
    ----------
    ntheta : int
        Length of `theta`: 4, plus 1 with `fit_cl`, plus 1 with `fit_tau`.
    k : int
        Number of shape coefficients, ``(lmax + 1)^2 - 1``.
    nb : int
        Number of photometric groups.
    Q : int
        Quadrature nodes per exposure.
    t_ref : float
        Phase reference [JD].
    normals_np : ndarray
        Facet normals, (nfacets, 3).
    loglike_jit : callable
        Jitted ``theta -> dict`` with ``logL``, the optimal coefficients ``c``, the Hessian
        ``H``, ``grad_norm``, ``chi2``, ``model``, the profiled offsets and phase slope ``beta``
        and ``min_eig``.

    Notes
    -----
    ``theta = (lambda, beta, P, ln sigma_jit[, ln c_L][, ln tau0])`` in degrees, hours and
    natural logs of magnitudes. See the module docstring for the model.

    Examples
    --------
    >>> M = ConvexInversion(D["t"], D["mag"], D["err"], D["band_idx"], D["alpha"],
    ...                     D["e_sun"], D["e_obs"])
    >>> M.loglike([30.0, 82.5, 3.24836, -4.07])   # ln L at one spin state
    """

    def __init__(self, t, mag, err, band_idx, alpha, e_sun, e_obs, lmax=4, nfacets=400,
                 c_lambert=0.1, tau0=0.6, closure_weight=2000.0, ab_prior=None, bc_prior=None,
                 n_inner=80, t_ref=None, n_starts=3, n_stage1=40, exptime=None, smear_nodes=None,
                 smear_P_h=None, fit_cl=False, cohesion=None, fit_tau=False):
        self.init_kwargs = dict(t=np.asarray(t), mag=np.asarray(mag), err=np.asarray(err),
                                band_idx=np.asarray(band_idx), alpha=np.asarray(alpha),
                                e_sun=np.asarray(e_sun), e_obs=np.asarray(e_obs), lmax=lmax,
                                nfacets=nfacets, c_lambert=c_lambert, tau0=tau0,
                                closure_weight=closure_weight, ab_prior=ab_prior,
                                bc_prior=bc_prior, n_inner=n_inner, t_ref=t_ref,
                                n_starts=n_starts, n_stage1=n_stage1, exptime=exptime,
                                smear_nodes=smear_nodes, smear_P_h=smear_P_h, fit_cl=fit_cl,
                                cohesion=cohesion, fit_tau=fit_tau)
        self.cohesion = cohesion   # None or dict (lcinv.cohesion.spec_from_args)
        self.fit_cl = bool(fit_cl)
        self.fit_tau = bool(fit_tau)
        self.i_tau = 4 + (1 if self.fit_cl else 0)
        self.ntheta = 4 + int(self.fit_cl) + int(self.fit_tau)
        self.t = jnp.asarray(t, float)
        self.t_ref = float(np.mean(t) if t_ref is None else t_ref)
        self.mag = jnp.asarray(mag, float)
        self.err2 = jnp.asarray(err, float) ** 2
        self.nb = int(np.max(band_idx)) + 1
        self.X = jnp.asarray(np.c_[np.eye(self.nb)[band_idx], alpha], float)  # linear nuisance
        self.e_sun = jnp.asarray(e_sun, float)
        self.e_obs = jnp.asarray(e_obs, float)
        # exposure smearing: Q Gauss-Legendre nodes per exposure (Q = 1: instantaneous)
        n = len(np.asarray(t))
        self.exptime = (np.zeros(n) if exptime is None
                        else np.broadcast_to(np.nan_to_num(np.asarray(exptime, float)), (n,)).copy())
        if smear_nodes is not None:
            self.Q = int(smear_nodes)
        elif smear_P_h is not None:
            self.Q = smear.choose_nodes(self.exptime, smear_P_h, smear.k_max_for_lmax(lmax))
        else:
            self.Q = 1
        tq, wq = smear.node_times(np.asarray(t, float), self.exptime, self.Q)
        self.t_nodes = jnp.asarray(tq)
        self.w_nodes = jnp.asarray(wq)
        self.e_sun_nodes = jnp.repeat(self.e_sun, self.Q, axis=0)
        self.e_obs_nodes = jnp.repeat(self.e_obs, self.Q, axis=0)
        self.normals_np = fibonacci_sphere(nfacets)
        self.normals = jnp.asarray(self.normals_np)
        Y, ls = real_sph_harm_basis(self.normals_np, lmax)
        self.Y = jnp.asarray(Y)
        self.ls = ls
        self.k = Y.shape[1]
        self.tau = jnp.asarray(tau0 / ls, float)
        self.tau_rel = jnp.asarray(1.0 / ls, float)
        self.w0 = 4 * np.pi / nfacets
        self.c_lambert = c_lambert
        self.closure_weight = closure_weight
        self.ab_prior = ab_prior
        self.bc_prior = bc_prior
        self.n_inner = n_inner
        # equatorial directions for projected-area axis ratios
        a = np.linspace(0, np.pi, 36, endpoint=False)
        self.u_eq = jnp.asarray(np.c_[np.cos(a), np.sin(a), np.zeros_like(a)])
        self._opt = optax.lbfgs()
        # coarse-to-fine optimisation: l<=2 mask and starting elongation angles
        self.mask2 = jnp.asarray((ls <= 2).astype(float))
        self.n_stage1 = n_stage1
        i22 = int(np.where((ls == 2))[0][-1])   # m = +2  (~ x^2 - y^2)
        i2m2 = int(np.where((ls == 2))[0][0])   # m = -2  (~ x y)
        st = np.zeros((n_starts, self.k))
        for j, ang in enumerate(np.linspace(0, np.pi, n_starts, endpoint=False)):
            st[j, i22] = -0.3 * np.cos(2 * ang)
            st[j, i2m2] = -0.3 * np.sin(2 * ang)
        self.starts = jnp.asarray(st)
        self.loglike_jit = jax.jit(self._loglike_full)

    # ------------------------------------------------------------------ pieces
    def areas(self, c):
        """Facet areas ``A_i = w0 exp(Y c)`` of shape coefficients `c`."""
        return self.w0 * jnp.exp(self.Y @ c)

    def _scatter(self, es, eo, c_l):
        mu0 = jnp.clip(es @ self.normals.T, 0.0, None)
        mu = jnp.clip(eo @ self.normals.T, 0.0, None)
        return mu * mu0 * (1.0 / (mu + mu0 + 1e-12) + c_l)

    def c_l(self, theta):
        """Lambert weight for a spin-state vector (fitted, or the fixed `c_lambert`)."""
        return jnp.exp(theta[4]) if self.fit_cl else self.c_lambert

    def tau_of(self, theta):
        """Prior scales ``tau_l = tau0 / l`` of the shape coefficients; with `fit_tau`, tau0 is
        taken from `theta` (as ln tau0).
        """
        return jnp.exp(theta[self.i_tau]) * self.tau_rel if self.fit_tau else self.tau

    def _smeared(self, lam, bet, P, phi0, t_nodes, es_nodes, eo_nodes, w, n, c_l=None):
        es = to_body(es_nodes, lam, bet, P, phi0, t_nodes, self.t_ref)
        eo = to_body(eo_nodes, lam, bet, P, phi0, t_nodes, self.t_ref)
        S = self._scatter(es, eo, self.c_lambert if c_l is None else c_l)
        Q = w.shape[0]
        return jnp.einsum("iqf,q->if", S.reshape(n, Q, -1), w)

    def scattering_matrix(self, lam, bet, P, phi0, c_l=None):
        """Brightness of each unit-area facet at each epoch.

        Parameters
        ----------
        lam, bet : float
            Pole [deg].
        P : float
            Period [h].
        phi0 : float
            Rotational phase at `t_ref` [rad] (0 in the likelihood).
        c_l : float, optional
            Lambert weight; default the fixed value.

        Returns
        -------
        jax.Array
            ``S[i, j]``, shape (N, nfacets): exposure-averaged ``mu mu0 [1/(mu + mu0) + c_L]`` of facet
            j at epoch i, zero where the facet is unlit or invisible. Brightness is ``S @ areas(c)``.
        """
        return self._smeared(lam, bet, P, phi0, self.t_nodes, self.e_sun_nodes, self.e_obs_nodes,
                             self.w_nodes, self.t.shape[0], c_l)

    def projected_areas(self, A):
        """Projected areas of a shape along 36 equatorial directions and along the spin axis."""
        Peq = 0.5 * jnp.abs(self.normals @ self.u_eq.T).T @ A
        Pz = 0.5 * jnp.abs(self.normals[:, 2]) @ A
        return Peq, Pz

    def axis_ratios(self, A, k=60.0):
        """Ellipsoid-equivalent axis ratios from projected areas.

        ``a/b`` is the ratio of the largest to the smallest equatorial projected area and
        ``b/c = P_z / max(P_eq)``; both are exact for a triaxial ellipsoid spinning about its short
        axis. Max and min are smoothed (log-sum-exp with sharpness `k`) so the priors stay
        differentiable.

        Returns
        -------
        ab, bc : jax.Array
        """
        Peq, Pz = self.projected_areas(A)
        lp = jnp.log(Peq)
        lmax_ = jax.scipy.special.logsumexp(k * lp) / k
        lmin_ = -jax.scipy.special.logsumexp(-k * lp) / k
        return jnp.exp(lmax_ - lmin_), jnp.exp(jnp.log(Pz) - lmax_)

    @staticmethod
    def _ratio_penalty(ratio, spec):
        if spec is None:
            return 0.0
        kind, val, sig = spec
        d = (jnp.log(ratio) - np.log(val)) / sig
        if kind == "lognormal":
            return 0.5 * d * d
        if kind == "lower":
            return 0.5 * jnp.clip(-d, 0.0, None) ** 2
        raise ValueError(kind)

    def model_mag(self, S, c, sig2):
        """Model reduced magnitudes with the linear nuisance profiled.

        Band offsets and the phase slope are solved by weighted least squares.

        Parameters
        ----------
        S : jax.Array
            Scattering matrix.
        c : jax.Array
            Shape coefficients.
        sig2 : jax.Array
            Total variances (quoted plus jitter) [mag^2].

        Returns
        -------
        model : jax.Array
        beta : jax.Array
            Offsets per group, then the phase slope [mag/deg].
        """
        L = S @ self.areas(c)
        y = self.mag + 2.5 * jnp.log10(L)
        W = 1.0 / sig2
        XtW = self.X.T * W
        beta = jnp.linalg.solve(XtW @ self.X, XtW @ y)
        return -2.5 * jnp.log10(L) + self.X @ beta, beta

    def nll(self, c, S, sig2):
        """Gaussian negative log-likelihood of the data for shape `c`, scattering matrix `S` and
        variances `sig2`.
        """
        model, _ = self.model_mag(S, c, sig2)
        r = self.mag - model
        return 0.5 * jnp.sum(r * r / sig2 + jnp.log(2 * jnp.pi * sig2))

    def objective(self, c, S, sig2, P=None, tau=None):
        """Negative log posterior of the shape coefficients at a fixed spin state.

        Data term, Gaussian prior on `c`, closure penalty, axis-ratio priors and, if set, the
        cohesion penalty.

        Parameters
        ----------
        c, S, sig2
            As in `model_mag`.
        P : float, optional
            Period [h]; needed only for the cohesion limit.
        tau : jax.Array, optional
            Prior scales; default the fixed ``tau0 / l``.

        Returns
        -------
        jax.Array
            Scalar.
        """
        nll = self.nll(c, S, sig2)
        tau = self.tau if tau is None else tau
        prior = 0.5 * jnp.sum((c / tau) ** 2) + jnp.sum(jnp.log(tau * jnp.sqrt(2 * jnp.pi)))
        A = self.areas(c)
        closure = jnp.sum((A @ self.normals) ** 2) / jnp.sum(A) ** 2
        pen = self.closure_weight * closure
        if self.ab_prior is not None or self.bc_prior is not None:
            ab, bc = self.axis_ratios(A)
            pen = pen + self._ratio_penalty(ab, self.ab_prior) + self._ratio_penalty(bc, self.bc_prior)
        if self.cohesion is not None and P is not None:
            from lcinv import cohesion as _co
            ab, bc = self.axis_ratios(A)
            pen = pen + _co.penalty_from_ratios_jax(ab, jnp.maximum(bc, 1.0), P, self.cohesion)[0]
        return nll + prior + pen

    def _run_lbfgs(self, f, c0, n):
        vg = optax.value_and_grad_from_state(f)
        opt = self._opt

        def step(carry, _):
            c, st = carry
            v, g = vg(c, state=st)
            upd, st = opt.update(g, st, c, value=v, grad=g, value_fn=f)
            return (optax.apply_updates(c, upd), st), v

        (c, _), _ = jax.lax.scan(step, (c0, opt.init(c0)), None, length=n)
        return c, f(c)

    def fit_shape(self, S, sig2, P=None, tau=None):
        """Coarse-to-fine optimisation of the shape at a fixed spin state.

        Stage 1 optimises only the l <= 2 (ellipsoid-like) coefficients from `n_starts` starts with
        the equatorial elongation at different angles (vectorised); stage 2 optimises all
        coefficients from the best stage-1 solution. This makes the optimum, and so the marginal
        likelihood, a smooth function of the spin state.

        Returns
        -------
        jax.Array
            Optimal coefficients.
        """
        f = lambda c: self.objective(c, S, sig2, P, tau)
        f1 = lambda c: self.objective(c * self.mask2, S, sig2, P, tau)
        cs, vs = jax.vmap(lambda c0: self._run_lbfgs(f1, c0, self.n_stage1))(self.starts)
        c1 = cs[jnp.argmin(vs)] * self.mask2
        c, _ = self._run_lbfgs(f, c1, self.n_inner)
        return c

    def _loglike_full(self, theta):
        lam, bet, P, lnsj = theta[0], theta[1], theta[2], theta[3]
        sig2 = self.err2 + jnp.exp(2 * lnsj)
        S = self.scattering_matrix(lam, bet, P, 0.0, self.c_l(theta))  # phi0 = 0: gauge fixed
        tau = self.tau_of(theta)
        c = self.fit_shape(S, sig2, P, tau)
        f = lambda cc: self.objective(cc, S, sig2, P, tau)
        fval, g = jax.value_and_grad(f)(c)
        H = jax.hessian(f)(c)
        ev = jnp.linalg.eigvalsh(0.5 * (H + H.T))
        # curvature floor = prior precision of the loosest coefficient: where the shape
        # optimisation has not converged (non-positive curvature) a direction then counts as
        # prior-only instead of inflating ln L by -1/2 ln(1e-10) per direction
        logdet = jnp.sum(jnp.log(jnp.clip(ev, 1.0 / jnp.max(tau) ** 2, None)))
        logL = -fval + 0.5 * self.k * jnp.log(2 * jnp.pi) - 0.5 * logdet
        model, beta = self.model_mag(S, c, sig2)
        chi2 = jnp.sum((self.mag - model) ** 2 / sig2)
        return dict(logL=logL, c=c, H=H, grad_norm=jnp.linalg.norm(g), chi2=chi2,
                    model=model, beta=beta, min_eig=ev[0])

    # ------------------------------------------------------------------ public helpers
    def fit_quality(self, theta):
        """Goodness of fit at a spin state (`lcinv.fitstats`).

        The shape coefficients count through ``tr(H_lik H^-1)``: the curvature of the data term
        relative to that of data, prior and penalties at the optimal shape.

        Parameters
        ----------
        theta : array_like

        Returns
        -------
        dict
            As `lcinv.fitstats.quality`, plus ``p_eff_shape``, ``k_shape`` and ``lmax``.
        """
        from lcinv import fitstats
        th = jnp.asarray(theta, float)
        out = self.loglike_jit(th)
        lam, bet, P, lnsj = th[0], th[1], th[2], th[3]
        sig2 = self.err2 + jnp.exp(2 * lnsj)
        S = self.scattering_matrix(lam, bet, P, 0.0, self.c_l(th))
        H_lik = np.asarray(jax.hessian(lambda cc: self.nll(cc, S, sig2))(out["c"]))
        p_shape = fitstats.effective_block(H_lik, np.asarray(out["H"]))
        n_spin = 3 + (1 if self.fit_cl else 0)
        n_lin = self.nb + 1
        r = np.asarray(self.mag - out["model"])
        if self.fit_tau:
            n_spin += 1                          # the inferred prior scale
        q = fitstats.quality(r, np.sqrt(np.asarray(self.err2)), float(np.exp(lnsj)), n_spin + n_lin + p_shape,
                             n_spin + n_lin + self.k, n_spin, n_lin)
        q["p_eff_shape"] = p_shape
        q["k_shape"] = int(self.k)
        q["lmax"] = int(self.ls.max())
        return q

    def axis_ratios_of(self, c):
        """Projected-area axis ratios ``(a/b, b/c)`` of shape coefficients `c`, as floats."""
        ab, bc = self.axis_ratios(self.areas(jnp.asarray(c)))
        return float(ab), float(bc)

    def loglike(self, theta):
        """Laplace-marginalised log-likelihood of a spin state.

        Parameters
        ----------
        theta : array_like
            ``(lambda, beta, P, ln sigma_jit[, ln c_L][, ln tau0])``.

        Returns
        -------
        float
            ``ln L``; -1e300 if not finite.
        """
        out = self.loglike_jit(jnp.asarray(theta, float))
        v = float(out["logL"])
        return v if np.isfinite(v) else -1e300

    def predict(self, theta, t, e_sun, e_obs, band_idx, alpha, c, beta, exptime=None, Q=None):
        """Model reduced magnitudes at arbitrary epochs and geometries, for plotting.

        Parameters
        ----------
        theta : array_like
            Spin state.
        t : array_like
            Emission times [JD].
        e_sun, e_obs : array_like
            Unit vectors, (N, 3).
        band_idx : array_like of int
            Group whose offset is applied.
        alpha : array_like
            Phase angles [deg].
        c : array_like
            Shape coefficients.
        beta : array_like
            Offsets and phase slope (``loglike_jit(theta)["beta"]``).
        exptime : float or array_like, optional
            Average over exposures of this length [s]; default instantaneous.
        Q : int, optional
            Nodes per exposure; default the model's.

        Returns
        -------
        ndarray
        """
        lam, bet, P = theta[:3]
        t = np.asarray(t, float)
        n = len(t)
        T = np.zeros(n) if exptime is None else np.broadcast_to(np.asarray(exptime, float), (n,))
        Q = (self.Q if Q is None else int(Q)) if exptime is not None else 1
        tq, wq = smear.node_times(t, T, Q)
        S = self._smeared(lam, bet, P, 0.0, jnp.asarray(tq), jnp.repeat(jnp.asarray(e_sun), Q, axis=0),
                          jnp.repeat(jnp.asarray(e_obs), Q, axis=0), jnp.asarray(wq), n,
                          self.c_l(jnp.asarray(theta, float)) if self.fit_cl else None)
        L = S @ self.areas(jnp.asarray(c))
        X = np.c_[np.eye(self.nb)[band_idx], alpha]
        return np.asarray(-2.5 * jnp.log10(L) + X @ jnp.asarray(beta))
