"""Cohesion needed to hold a spinning rubble pile together (Sanchez & Scheeres 2014).

Sanchez, P., & Scheeres, D. J. 2014, Meteoritics & Planetary Science, 49, 788
("The strength of regolith and rubble pile asteroids"). For a uniformly rotating triaxial
ellipsoid with semi-axes alpha >= beta >= gamma, spinning about gamma, the Drucker-Prager
criterion at the centre (their eq. 20 with E = 1, i.e. eq. 23) gives

    omega^2 <= omega_alpha^2 + 2 / (rho alpha^2) (3 - sin phi) / (3 + sin phi) sigma_Y,

so the minimum cohesive (tensile yield) strength is

    sigma_Y >= rho alpha^2 (omega^2 - omega_alpha^2) (3 + sin phi) / (2 (3 - sin phi)),

with the self-gravity term (their eq. 16)

    omega_alpha^2 = 2 pi G rho alpha beta gamma int_0^inf dv / ((alpha^2 + v) sqrt(
                    (alpha^2 + v)(beta^2 + v)(gamma^2 + v)))
                  = (4/3) pi G rho alpha beta gamma R_D(beta^2, gamma^2, alpha^2)

(R_D = Carlson's symmetric elliptic integral). phi -> 90 deg gives their conservative eq. 24,
sigma_Y >= rho alpha^2 (omega^2 - omega_alpha^2), which is the default here (the smallest
required strength, so the limit rejects as few shapes as possible). Their estimate for
rubble piles held by van der Waals forces between fine regolith is ~25 Pa (sigma_Y =
1.56e-4 / r_p Pa, r_p = mean grain radius in m).

Use in the inversion: with --cohesion-max S (Pa), shapes and spin states that need more than
S are penalised, ln g = -1/2 (max(0, ln(sigma_req / S)) / softness)^2. The penalty acts as an
extra (soft) constraint "the body holds together with cohesion <= S", so it enters the
evidence of every model the same way and model comparison stays consistent.
Needs the volume-equivalent diameter (--diameter-km) and the bulk density (--density).
"""
import numpy as np

G = 6.674e-11


# ----------------------------------------------------------------------------- numpy versions
def omega_alpha2(alpha, beta, gamma, rho):
    """Self-gravity term omega_alpha^2 (rad^2/s^2) for semi-axes in m and density in kg/m^3."""
    from scipy.special import elliprd
    a, b, c = (np.asarray(x, float) for x in (alpha, beta, gamma))
    return 4.0 / 3.0 * np.pi * G * rho * a * b * c * elliprd(b * b, c * c, a * a)


def friction_factor(phi_deg):
    """Friction term ``(3 + sin phi) / (2 (3 - sin phi))`` of the Drucker-Prager criterion for
    internal friction angle `phi_deg`.
    """
    s = np.sin(np.radians(phi_deg))
    return (3.0 + s) / (2.0 * (3.0 - s))


def required_cohesion(alpha, beta, gamma, P_h, rho, phi_deg=90.0, clip=True):
    """Minimum cohesive strength (Pa) for semi-axes alpha >= beta >= gamma (m), spin period
    P_h (h) about gamma and bulk density rho (kg/m^3). Zero below the gravitational spin limit
    unless clip=False."""
    w2 = (2 * np.pi / (np.asarray(P_h, float) * 3600.0)) ** 2
    s = rho * np.asarray(alpha, float) ** 2 * (w2 - omega_alpha2(alpha, beta, gamma, rho)) * friction_factor(phi_deg)
    return np.clip(s, 0.0, None) if clip else s


def semi_axes_from_ratios(D_km, ab, bc):
    """Semi-axes (m) of the ellipsoid with volume-equivalent diameter D and axis ratios a/b,
    b/c: alpha beta gamma = (D/2)^3."""
    R = 0.5 * np.asarray(D_km, float) * 1e3
    b = R * (np.asarray(bc, float) / np.asarray(ab, float)) ** (1.0 / 3.0)
    return ab * b, b, b / bc


def semi_axes_from_moments(moments, D_km):
    """Semi-axes (m) of the uniform ellipsoid with the given principal moments of inertia
    (any normalisation, ascending or not), scaled to volume-equivalent diameter D."""
    I = np.sort(np.asarray(moments, float))           # I1 <= I2 <= I3  (I1 about the long axis)
    a2 = I[1] + I[2] - I[0]
    b2 = I[0] + I[2] - I[1]
    c2 = I[0] + I[1] - I[2]
    ax = np.sqrt(np.clip([a2, b2, c2], 1e-12, None))
    ax = ax * (0.5 * D_km * 1e3) / np.prod(ax) ** (1.0 / 3.0)
    return tuple(np.sort(ax)[::-1])


def penalty(sigma_req, sigma_max, softness=0.1):
    """-ln g: zero up to sigma_max, then quadratic in ln(sigma_req / sigma_max)."""
    x = np.log(np.maximum(sigma_req, 1e-30) / sigma_max)
    return 0.5 * (np.maximum(x, 0.0) / softness) ** 2


# ----------------------------------------------------------------------------- JAX versions
def _rd_jax(x, y, z, n_iter=12):
    """Carlson R_D(x, y, z) by the duplication theorem (fixed iterations; differentiable)."""
    import jax.numpy as jnp
    s = 0.0
    fac = 1.0
    for _ in range(n_iter):
        sx, sy, sz = jnp.sqrt(x), jnp.sqrt(y), jnp.sqrt(z)
        lam = sx * sy + sx * sz + sy * sz
        s = s + fac / (sz * (z + lam))
        fac = fac / 4.0
        x, y, z = (x + lam) / 4.0, (y + lam) / 4.0, (z + lam) / 4.0
    mu = (x + y + 3.0 * z) / 5.0
    X, Y, Z = (mu - x) / mu, (mu - y) / mu, (mu - z) / mu
    ea = X * Y
    eb = Z * Z
    ec = ea - eb
    ed = ea - 6.0 * eb
    ee = ed + ec + ec
    series = 1.0 + ed * (-3.0 / 14.0 + 9.0 / 88.0 * ed - 4.5 / 26.0 * Z * ee) \
        + Z * (ee / 6.0 + Z * (-9.0 / 22.0 * ec + 3.0 / 26.0 * Z * ea))
    return 3.0 * s + fac * series / (mu * jnp.sqrt(mu))


def required_cohesion_jax(alpha, beta, gamma, P_h, rho, phi_deg=90.0):
    """Differentiable required cohesion (Pa), not clipped (negative below the gravity limit)."""
    import jax.numpy as jnp
    w2 = (2 * jnp.pi / (P_h * 3600.0)) ** 2
    # scale lengths to km inside R_D for conditioning (R_D is homogeneous of degree -3/2)
    k = 1e-3
    rd = _rd_jax((beta * k) ** 2, (gamma * k) ** 2, (alpha * k) ** 2) * k ** 3
    wa2 = 4.0 / 3.0 * jnp.pi * G * rho * alpha * beta * gamma * rd
    return rho * alpha ** 2 * (w2 - wa2) * friction_factor(phi_deg)


def penalty_jax(sigma_req, sigma_max, softness=0.1):
    """Differentiable `penalty`: ``0.5 (max(0, ln(sigma_req / sigma_max)) / softness)^2``."""
    import jax.numpy as jnp
    x = jnp.log(jnp.clip(sigma_req, 1e-30, None) / sigma_max)
    return 0.5 * (jnp.clip(x, 0.0, None) / softness) ** 2


def penalty_from_ratios_jax(ab, bc, P_h, spec):
    """Penalty for an ellipsoid given axis ratios (convex model)."""
    import jax.numpy as jnp
    R = 0.5 * spec["diameter_km"] * 1e3
    b = R * (bc / ab) ** (1.0 / 3.0)
    s = required_cohesion_jax(ab * b, b, b / bc, P_h, spec["density"], spec["friction_deg"])
    return penalty_jax(s, spec["sigma_max"], spec["softness"]), jnp.clip(s, 0.0, None)


# ----------------------------------------------------------------------------- CLI helpers
def add_cli_arguments(p, diameter=True):
    """Add the cohesion and size options to an argument parser.

    ``--cohesion-max``, ``--diameter-km`` (if `diameter`), ``--H``, ``--albedo``,
    ``--size-from-H``, ``--density``, ``--friction-angle`` and ``--cohesion-softness``.
    """
    g = p.add_argument_group("cohesion limit (Sanchez & Scheeres 2014)")
    g.add_argument("--cohesion-max", type=float, default=None, metavar="PA",
                   help="penalise shapes/spins that need more cohesive strength than this (Pa); "
                        "~25 Pa is the Sanchez & Scheeres (2014) estimate for fine-regolith rubble piles. "
                        "Needs --diameter-km")
    if diameter:
        g.add_argument("--diameter-km", type=float, default=None,
                       help="volume-equivalent diameter (km); also enables the required-cohesion report")
    g.add_argument("--H", type=float, default=None,
                   help="absolute magnitude: diameter from the CNEOS size estimator if --diameter-km is not given")
    g.add_argument("--albedo", type=float, default=0.15, help="geometric albedo for the H-based diameter")
    g.add_argument("--size-from-H", action="store_true",
                   help="diameter from H looked up in the JPL Small-Body Database (CNEOS formula)")
    g.add_argument("--density", type=float, default=1700.0, help="bulk density (kg/m^3)")
    g.add_argument("--friction-angle", type=float, default=90.0,
                   help="internal friction angle (deg); 90 = conservative limit (their eq. 24)")
    g.add_argument("--cohesion-softness", type=float, default=0.1,
                   help="width of the soft limit in ln(sigma_req / cohesion_max)")


def resolve_diameter(a):
    """Volume-equivalent diameter (km): --diameter-km, else CNEOS estimate from --H, else from H
    looked up in JPL SBDB (--size-from-H, using the designation). Stores it in a."""
    get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    D = get("diameter_km")
    if D:
        return float(D)
    if get("H") is None and not get("size_from_H"):
        return None
    from lcinv import size
    e = size.estimate_for(get("designation"), H=get("H"), albedo=get("albedo", 0.15) or 0.15,
                          lookup=bool(get("size_from_H")))
    print(f"size: H = {e['H']:.2f} ({e['H_source']}), albedo {e['albedo']:g} -> D = {e['diameter_km']:.3f} km "
          f"(CNEOS; {e['diameter_min_km']:.3f}-{e['diameter_max_km']:.3f} km for albedo 0.30-0.05)")
    if isinstance(a, dict):
        a["diameter_km"] = e["diameter_km"]
    else:
        a.diameter_km = e["diameter_km"]
    return e["diameter_km"]


def spec_from_args(a):
    """dict for the models, or None. Raises if --cohesion-max is given without a diameter."""
    smax = getattr(a, "cohesion_max", None) if not isinstance(a, dict) else a.get("cohesion_max")
    get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    D = resolve_diameter(a) if smax is not None else get("diameter_km")
    if smax is None:
        return None
    if not D or not np.isfinite(D):
        raise SystemExit("--cohesion-max needs a size: --diameter-km, --H or --size-from-H")
    return dict(sigma_max=float(smax), diameter_km=float(D), density=float(get("density", 1700.0)),
                friction_deg=float(get("friction_angle", 90.0)), softness=float(get("cohesion_softness", 0.1)))


def report_spec(a):
    """Settings for the required-cohesion report (diameter known), even without a limit."""
    get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    D = resolve_diameter(a)
    if not D or not np.isfinite(D):
        return None
    return dict(diameter_km=float(D), density=float(get("density", 1700.0)),
                friction_deg=float(get("friction_angle", 90.0)), sigma_max=get("cohesion_max"))


def summarise(sig, spec):
    """Posterior summary of required cohesion draws (Pa)."""
    sig = np.asarray(sig, float)
    sig = sig[np.isfinite(sig)]
    if len(sig) == 0:
        return dict(required_cohesion_Pa=None, note="no finite shape draws")
    out = dict(required_cohesion_Pa=[float(v) for v in np.percentile(sig, [16, 50, 84])],
               frac_needs_cohesion=float(np.mean(sig > 0)),
               density=spec["density"], diameter_km=spec["diameter_km"], friction_deg=spec["friction_deg"],
               reference="Sanchez & Scheeres 2014, MAPS 49, 788 (eq. 23/24)")
    if spec.get("sigma_max"):
        out.update(cohesion_max_Pa=float(spec["sigma_max"]),
                   frac_within_limit=float(np.mean(sig <= spec["sigma_max"])))
    return out


def ellipsoid_from_moments_jax(Ixx, Iyy, Izz, D_km):
    """Equivalent-ellipsoid semi-axes (m) for principal moments about x <= y <= z axes,
    scaled to volume-equivalent diameter D (JAX)."""
    import jax.numpy as jnp
    a2 = jnp.clip(Iyy + Izz - Ixx, 1e-12, None)
    b2 = jnp.clip(Ixx + Izz - Iyy, 1e-12, None)
    c2 = jnp.clip(Ixx + Iyy - Izz, 1e-12, None)
    a, b, c = jnp.sqrt(a2), jnp.sqrt(b2), jnp.sqrt(c2)
    k = 0.5 * D_km * 1e3 / (a * b * c) ** (1.0 / 3.0)
    return a * k, b * k, c * k
