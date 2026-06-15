# Constitutive utilities: strain, stress, and complex inner products.

from ngsolve import *
from ngsolve import Grad
import numpy as np


def strain(w):
    return Sym(Grad(w))

def stress(w, lam, mu, dim=2):
    aa = strain(w)
    return 2 * mu * aa + lam * Trace(aa) * Id(dim)


def complexify(trial_func: tuple, test_func: tuple):
    """Re(a*conj(b)) split into real/imag blocks for saddle systems."""
    real_part = InnerProduct(trial_func[0], test_func[0]) + InnerProduct(trial_func[1], test_func[1])
    imag_part = InnerProduct(trial_func[0], test_func[1]) - InnerProduct(trial_func[1], test_func[0])
    return real_part + imag_part

def complexify_multi(trial_func: tuple, test_func: tuple):
    """Scalar version of complexify (no InnerProduct)."""
    real_part = trial_func[0]*test_func[0] + trial_func[1]*test_func[1]
    imag_part = trial_func[0]*test_func[1] - trial_func[1]*test_func[0]
    return real_part + imag_part


# Uncalibrated reference value of the nondimensional GB diffusion coefficient
# C_d = ΩδD^gb/kT. The physically meaningful value is the per-geometry Maxwell
# time τ_M = η_ss/G_U, which is MEASURED at runtime (calibrate_tau.measure_tau_M)
# and passed in via the ``diff_coeff`` argument of solve_rve — NOT hardwired,
# because τ_M depends on the grain geometry. Setting C_d = τ_M folds the Maxwell
# time into the form so code-ω reads ω·τ_M and the fluid↔solid crossover sits at
# ω = 1. The same C_d must scale the output-loss prefactor in the drivers
# (dissipation = (C_d/ω)∫|∂ₛt_n|²). This default (1.0) is only the uncalibrated
# state used by the calibration measurement itself. See docs/DAGBS_physics.md §5.
DIFF_COEFF = 1.0


def gb_diffusion_block(t_n_Re, t_n_Im, r_n_Re, r_n_Im, omega, diff_coeff=None):
    """DAGBS grain-boundary diffusion term (i·C_d/ω) ∫ ∂ₛt_n · ∂ₛr_n.

    Implements the Coble/Raj-Ashby normal channel (Rudge 2025, eq. A7):
    ∇²⊥ σ_nn = −(kT/ΩδD^gb)[∂u_n/∂t], whose weak form contributes
    +i·(ΩδD^gb/ωkT) ∫ ∂ₛt_n* · ∂ₛr_n over each grain boundary. The
    diffusion coefficient C_d = ΩδD^gb/kT (nondimensional) is the Maxwell-time
    calibration constant ``DIFF_COEFF`` (override via ``diff_coeff``; pass 0 to
    recover the ω→∞ normal-locked unrelaxed limit). It mirrors the EAGBS viscous
    term's 1/(ωη) prefactor.

    The surface gradient of an H1 field defined on the boundary is the
    arc-length derivative ∂ₛ (Laplace-Beltrami); in NGSolve this is
    ``Grad(u).Trace()`` for a boundary-restricted space. Multiplication by i is
    encoded in real/imag blocks via the same swap-and-sign trick used in the
    EAGBS sliding law: trial (Re, Im) → (-Im, Re).
    """
    c = DIFF_COEFF if diff_coeff is None else diff_coeff
    return (c / omega) * complexify(
        (-Grad(t_n_Im).Trace(), Grad(t_n_Re).Trace()),
        (Grad(r_n_Re).Trace(), Grad(r_n_Im).Trace()),
    )