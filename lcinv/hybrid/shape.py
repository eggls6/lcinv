"""Star-shaped log-radius field and its physical properties (JAX).

    r(u) = exp( sum_{l=lmin..lmax, m} a_lm Y_lm(u) ),   a_lm = tau_l(sigma_f, ell) z_lm,  z ~ N(0, 1)

Non-centred parametrisation of a truncated, isotropic Gaussian-process-like prior with a
Matern-like angular power spectrum

    tau_l^2 = sigma_f^2 * 4 pi s_l / sum_l' (2 l' + 1) s_l',   s_l = (1 + l (l + 1) / ell^2)^(-nu)

normalised so that sigma_f is the pointwise standard deviation of log r. l = 0 (scale) is
degenerate with the photometric offsets and l = 1 (approximately a translation) is weakly
identified, so lmin = 2 by default. Hyperpriors: ln sigma_f, ln ell ~ Normal.

Volume, centroid and inertia of the uniform-density body follow from radial integrals
evaluated with the grid quadrature:
    V = 1/3 int r^3 dOmega,  c = (1/4 int r^4 u dOmega) / V,
    M = 1/5 int r^5 u u^T dOmega - V c c^T,  I = tr(M) 1 - M.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class RadiusField:
    """Star-shaped log-radius field with a Matern-like prior and its mass properties.

    Parameters
    ----------
    grid : SphereGrid
        Quadrature grid and basis (`lcinv.hybrid.quadrature.make_grid`).
    nu : float, optional
        Spectral index of the prior power spectrum (``--nu``).
    ln_sigma_prior : (float, float), optional
        Mean and sd of the normal prior on ln sigma_f, the pointwise sd of log r (``--sigma-f``).
    ln_ell_prior : (float, float), optional
        Mean and sd of the normal prior on ln ell, the angular correlation scale (``--ell``).
    inertia_offdiag : float, optional
        Scale of the soft constraint on the products of inertia I_xz, I_yz (``--inertia-offdiag``).
    inertia_order : float, optional
        Scale of the soft constraint ``I_zz >= max eig(I_xy)`` (``--inertia-order``).
    convexity_scale : float or None, optional
        Scale of the weak concavity penalty; None switches it off.
    """
    def __init__(self, grid, nu=2.0, ln_sigma_prior=(np.log(0.15), 0.5),
                 ln_ell_prior=(np.log(3.0), 0.5), inertia_offdiag=0.01, inertia_order=0.01,
                 convexity_scale=0.02):
        self.g = grid
        self.nu = nu
        self.lsig_mu, self.lsig_sd = ln_sigma_prior
        self.lell_mu, self.lell_sd = ln_ell_prior
        self.s_off = inertia_offdiag
        self.s_ord = inertia_order
        self.s_cvx = convexity_scale
        self.l = jnp.asarray(grid.l, float)
        self.ul = np.unique(grid.l)
        self.Yq = jnp.asarray(grid.Y_q)
        self.Ym = jnp.asarray(grid.Y_m)
        self.uq = jnp.asarray(grid.u_q)
        self.um = jnp.asarray(grid.u_m)
        self.wq = jnp.asarray(grid.w_q)
        self.tri = jnp.asarray(grid.tri)
        self.nbr = jnp.asarray(grid.nbr)
        self.k = grid.k

    # ------------------------------------------------------------------ prior spectrum
    def tau(self, ln_sigma, ln_ell):
        """Prior scales ``tau_l(sigma_f, ell)`` of the coefficients."""
        ell = jnp.exp(ln_ell)
        s = (1.0 + self.l * (self.l + 1.0) / ell ** 2) ** (-self.nu)
        ul = jnp.asarray(self.ul, float)
        su = (1.0 + ul * (ul + 1.0) / ell ** 2) ** (-self.nu)
        norm = jnp.sum((2 * ul + 1) * su) / (4 * jnp.pi)
        return jnp.exp(ln_sigma) * jnp.sqrt(s / norm)

    def coeffs(self, z, ln_sigma, ln_ell):
        """Coefficients ``a = tau(ln_sigma, ln_ell) * z``."""
        return self.tau(ln_sigma, ln_ell) * z

    def hyper_logprior(self, ln_sigma, ln_ell):
        """Log prior of the hyperparameters ln sigma_f and ln ell."""
        n = lambda x, mu, sd: -0.5 * ((x - mu) / sd) ** 2 - jnp.log(sd * jnp.sqrt(2 * jnp.pi))
        return n(ln_sigma, self.lsig_mu, self.lsig_sd) + n(ln_ell, self.lell_mu, self.lell_sd)

    # ------------------------------------------------------------------ geometry
    def radius_mesh(self, a):
        """Radii at the mesh vertices."""
        return jnp.exp(self.Ym @ a)

    def radius_quad(self, a):
        """Radii at the quadrature points."""
        return jnp.exp(self.Yq @ a)

    def mesh(self, a):
        """Vertices, facet areas and unit normals of the triangulated surface."""
        v = self.radius_mesh(a)[:, None] * self.um
        p = v[self.tri]
        cr = jnp.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        nn = jnp.linalg.norm(cr, axis=1)
        return v, 0.5 * nn, cr / nn[:, None]

    def mass_properties(self, a):
        """Volume, centroid and inertia tensor of the uniform-density body (radial integrals on the grid).

        Returns
        -------
        V : jax.Array
        c : jax.Array
            Centroid, (3,).
        I : jax.Array
            Inertia tensor about the centroid, (3, 3).
        """
        r = self.radius_quad(a)
        V = jnp.sum(self.wq * r ** 3) / 3.0
        c = jnp.sum((self.wq * r ** 4)[:, None] * self.uq, axis=0) / 4.0 / V
        M = jnp.einsum("q,qi,qj->ij", self.wq * r ** 5, self.uq, self.uq) / 5.0 - V * jnp.outer(c, c)
        I = jnp.trace(M) * jnp.eye(3) - M
        return V, c, I

    def inertia_logprior(self, a):
        """Soft principal-axis rotation about z: small I_xz, I_yz and I_zz >= lambda_max(I_xy)."""
        _, _, I = self.mass_properties(a)
        izz = I[2, 2]
        lp = -0.5 * ((I[0, 2] / izz) / self.s_off) ** 2 - 0.5 * ((I[1, 2] / izz) / self.s_off) ** 2
        lam_xy = 0.5 * (I[0, 0] + I[1, 1]) + jnp.sqrt(0.25 * (I[0, 0] - I[1, 1]) ** 2 + I[0, 1] ** 2)
        d = (lam_xy - izz) / izz
        lp = lp - 0.5 * (self.s_ord * jax.nn.softplus(d / self.s_ord)) ** 2 / self.s_ord ** 2
        return lp

    def convexity_logprior(self, a):
        """Weak penalty on local concavity: a vertex lying inside the mean of its ring
        neighbours (along the radial direction) indicates a dent."""
        if self.s_cvx is None:
            return 0.0
        r = self.radius_mesh(a)
        v = r[:, None] * self.um
        nq = self.nbr.shape[0]
        mean_nb = jnp.mean(v[self.nbr], axis=1)
        out = jnp.sum((v[:nq] - mean_nb) * self.um[:nq], axis=1) / jnp.mean(r)
        return -0.5 * jnp.sum(jax.nn.relu(-out) ** 2) / self.s_cvx ** 2

    # ------------------------------------------------------------------ diagnostics
    def inertia_diagnostics(self, a):
        """Inertia diagnostics of a shape.

        Returns
        -------
        dict
            ``volume``, ``centroid``, principal ``moments`` (normalised to the largest) and
            ``angle_deg``, the angle between the spin axis z and the maximum-inertia axis.
        """
        V, c, I = self.mass_properties(a)
        w, U = jnp.linalg.eigh(I)
        ang = jnp.degrees(jnp.arccos(jnp.clip(jnp.abs(U[2, -1]), 0.0, 1.0)))
        return dict(volume=V, centroid=c, moments=w / w[-1], angle_deg=ang)
