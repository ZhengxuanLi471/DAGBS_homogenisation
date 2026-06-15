# Per-geometry Maxwell-time calibration for the DAGBS homogeniser.
#
# The Maxwell time is the ratio of the steady-state (Coble) viscosity to the
# unrelaxed shear modulus,
#
#     tau_M = eta_ss / G_U ,
#
# both of which are EMERGENT properties of the homogenised RVE and therefore
# depend on the grain geometry. They are read off the two asymptotes of the
# computed complex modulus C*(omega) under simple shear:
#
#     G_U   = lim_{omega->inf} C'   -- unrelaxed shear modulus.  Measured as the
#             diffusion-off limit (diff_coeff=0): the GB surface-diffusion
#             self-term vanishes, t_n enforces [u_n]=0 (normal-locked, free
#             tangential slide), giving an omega-independent elastic problem.
#
#     eta_ss = lim_{omega->0} C''/omega -- steady-state Coble viscosity.  Measured
#             at the uncalibrated reference diff_coeff=1 and a single low omega
#             (the C''/omega plateau is flat there).
#
# tau_M is then folded into the bilinear form by running the production sweep
# with diff_coeff = tau_M: this rescales code-omega to omega*tau_M, putting the
# fluid<->solid crossover (C' ~ C'', Q^-1 ~ 1) at omega = 1. tau_M is NEVER
# hardcoded -- call measure_tau_M once per geometry.
#
# Usage:
#   from calibrate_tau import measure_tau_M
#   tau_M, info = measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
#                               junction_incidence, num_grains, nu=NU, mu=MU)
#   # then sweep with solve_rve(..., diff_coeff=tau_M) and loss prefactor tau_M.
#
# This module builds NO geometry of its own -- the caller (real_im_energy.py,
# real_im_hex_energy.py, ...) constructs the mesh/spaces and passes them in, so
# the same calibration runs for any RVE.

from main import solve_rve
from physics import *
from ngsolve import *
import numpy as np

MACRO_SCALE = 1e-3                       # Gamma scaling in _setup_material_properties
MODULUS_SCALE = 2.0 / (MACRO_SCALE ** 2)  # recover C* from the energy density
SHEAR = ((0, 1), (0, 0))                 # simple-shear macro strain


def storage_modulus(gfu, mesh, nu, mu, num_grains):
    """C' = MODULUS_SCALE * (1/2)int sigma:eps / |Omega|, summed grain by grain
    (never integrating across a GB)."""
    uR, uI = gfu.components[0], gfu.components[1]
    lam = 2 * mu * nu / (1 - 2 * nu)
    density = InnerProduct(stress(uR, lam, mu), strain(uR)) + \
              InnerProduct(stress(uI, lam, mu), strain(uI))
    area = float(Integrate(1, mesh, VOL))
    bulk = 0.0
    for g in range(1, num_grains + 1):
        bulk += float(Integrate(density, mesh, VOL,
                                definedon=mesh.Materials(f"region_{g}")))
    return MODULUS_SCALE * 0.5 * bulk / area


def diffusional_loss(gfu, mesh, contact_pairs, gb_normal_indices, omega, diff_coeff):
    """C'' = MODULUS_SCALE * (1/2)(C_d/omega) int_{core u slide} |d_s t_n|^2 / |Omega|.
    The C_d prefactor must match the diff_coeff used in the form."""
    area = float(Integrate(1, mesh, VOL))
    gb_energy = 0.0
    for (a, b), (_, right_name) in contact_pairs.items():
        region = mesh.Boundaries(f"core_{right_name}.*|slide_{right_name}")
        idx_re, idx_im = gb_normal_indices[(a, b)]
        gtr = Grad(gfu.components[idx_re]).Trace()
        gti = Grad(gfu.components[idx_im]).Trace()
        flux_sq = InnerProduct(gtr, gtr) + InnerProduct(gti, gti)
        gb_energy += float(Integrate(flux_sq, mesh, BND, definedon=region))
    return MODULUS_SCALE * 0.5 * diff_coeff * gb_energy / area / omega


def fit_andrade_time(x, Gc, G0, alpha, window=(300.0, 5000.0)):
    """Andrade time tau_A/tau_M from the high-omega complex modulus.

    Fits the high-frequency Andrade form
        G* = G0 / (1 + Gamma(1+alpha) (i*omega*tau_A)^-alpha)
    (the 1/(i*omega*tau_M) viscous term is negligible at high omega) to the FEM
    complex modulus Gc = G1 + i*G2, over x = omega*tau_M in `window`. Returns the
    single least-squares best-fit tau_A/tau_M. Fitting the full Andrade form (not
    just the leading Q^-1 ~ (omega tau_A)^-alpha asymptote) removes the ~10% bias
    of the leading-order estimate. alpha is fixed by the triple-junction angle.
    """
    from scipy.special import gamma as _gamma
    from scipy.optimize import minimize_scalar
    Ga = _gamma(1.0 + alpha)
    x = np.asarray(x, dtype=float)
    Gc = np.asarray(Gc, dtype=complex)
    m = (x >= window[0]) & (x <= window[1])
    xm, Gm = x[m], Gc[m]
    def resid(r):
        model = G0 / (1.0 + Ga * (1j * xm * r) ** (-alpha))
        return float(np.sum(np.abs(model - Gm) ** 2))
    return float(minimize_scalar(resid, bounds=(0.1, 8.0), method="bounded").x)


# Low-omega endmember for the eta_ss = C''/omega measurement. The creep plateau
# is extremely wide: eta_ss = C''/omega is flat for omega*tau_M << 1, i.e. (at the
# uncalibrated diff_coeff=1) for code-omega << 1/tau_M ~ 3e4. So we measure at the
# WELL-CONDITIONED code-omega = 1, where the diffusion form-coefficient
# (diff_coeff/omega) is O(1). Do NOT push omega very low: at e.g. ln omega = -10
# the coefficient 1/omega ~ e^10 is huge, the saddle system becomes stiff, and the
# bulk-only multigrid CG FAILS to converge for stiff (near-incompressible) nu --
# giving garbage eta_ss/tau_M. At code-omega = 1, plain CG converges and eta_ss
# comes out ~2.418e-5 for ALL nu (it is the geometry-only Coble viscosity,
# nu-independent, as theory requires). The high-omega endmember (G_U) needs no
# large omega -- diff_coeff=0 gives the normal-locked limit directly.
LOW_OMEGA = 1.0


def measure_tau_M(spaces, mesh, contact_pairs, outer_contact_pairs,
                  junction_incidence, num_grains, nu, mu,
                  low_omega=LOW_OMEGA, ref_omega=1.0, solver='cg', rtol=1e-8):
    """Measure tau_M = eta_ss / G_U for one geometry (simple shear).

    Two solves only:
      * G_U   from diff_coeff=0 (normal-locked, omega-independent), and
      * eta_ss = C''/omega at diff_coeff=1 and a single low omega (default
        ln omega = -20, deep in the creep plateau).
    Returns (tau_M, info) where info carries eta_ss, G_U and the implied
    crossover ln(1/tau_M). Pass tau_M as diff_coeff to the production sweep.
    """
    gb_normal_indices = spaces[4]

    print("=" * 70, flush=True)
    print(f"[calibrate_tau] measure_tau_M: {num_grains} grains, nu={nu}, mu={mu}, "
          f"solver='{solver}', rtol={rtol:g}", flush=True)
    print("=" * 70, flush=True)

    # G_U: omega-independent, diffusion-off (normal-locked) elastic limit.
    print(f"[calibrate_tau] solve 1/2: G_U  (diff_coeff=0, omega={ref_omega:g}) ...",
          flush=True)
    gfu, mesh = solve_rve(
        spaces, mesh, contact_pairs, outer_contact_pairs, SHEAR,
        nu=nu, mu=mu, omega=ref_omega, solver=solver, rtol=rtol,
        junction_incidence=junction_incidence, diff_coeff=0.0)
    G_U = storage_modulus(gfu, mesh, nu, mu, num_grains)
    print(f"[calibrate_tau] solve 1/2 done: G_U = {G_U:.6e}", flush=True)

    # eta_ss: low-omega C''/omega plateau at the uncalibrated reference C_d=1.
    print(f"[calibrate_tau] solve 2/2: eta_ss  (diff_coeff=1, omega={low_omega:g}) ...",
          flush=True)
    gfu, mesh = solve_rve(
        spaces, mesh, contact_pairs, outer_contact_pairs, SHEAR,
        nu=nu, mu=mu, omega=low_omega, solver=solver, rtol=rtol,
        junction_incidence=junction_incidence, diff_coeff=1.0)
    Cpp = diffusional_loss(gfu, mesh, contact_pairs, gb_normal_indices,
                           low_omega, diff_coeff=1.0)
    eta_ss = Cpp / low_omega
    print(f"[calibrate_tau] solve 2/2 done: eta_ss = {eta_ss:.6e} "
          f"(C''={Cpp:.6e})", flush=True)

    tau_M = eta_ss / G_U
    info = dict(eta_ss=eta_ss, G_U=G_U, tau_M=tau_M,
                ln_crossover=float(np.log(1.0 / tau_M)))
    print(f"[calibrate_tau] RESULT: tau_M = eta_ss/G_U = {tau_M:.6e}  "
          f"(ln(1/tau_M) = {info['ln_crossover']:.4f})", flush=True)
    print("=" * 70, flush=True)
    return tau_M, info
