# DAGBS mixed FE solver: assembles and solves the coupled displacement/multiplier
# system for one RVE under a prescribed macro loading tensor.
#
# Diffusion-accommodated grain-boundary sliding (Coble regime): interior grain
# boundaries slide freely in tangent (σ_ns = 0) and open diffusively in normal,
# governed by the surface-Laplacian law ∇²⊥σ_nn = −(kT/ΩδD^gb)[∂u_n/∂t]
# (Rudge 2025, App. A). See docs/DAGBS_physics.md.

from ngsolve import *
import numpy as np
from math import sqrt
from mpi4py import MPI
from physics import *
from types import SimpleNamespace


# --- helpers ---------------------------------------------------------------

def _setup_material_properties(gamma, nu, mu):
    """Return the scaled conductivity tensor and Lamé parameter."""
    Gamma = CoefficientFunction(gamma, dims=(2, 2)) * 1e-3
    lam = 2 * mu * nu / (1 - 2 * nu)
    return Gamma, lam


def _core_boundary_names(boundary_info):
    """Collect unique core boundary names from the pairing metadata."""
    names = []
    if boundary_info:
        for name in boundary_info.get("core_names") or []:
            if name and name not in names:
                names.append(name)
    return names


def _core_boundary_expression(boundary_info):
    """Join core boundary names into a single Netgen expression."""
    names = _core_boundary_names(boundary_info)
    return "|".join(names)


def _add_gb_normal_coupling(a, mesh, sym, contact_pairs, region_kind):
    """Add the DAGBS normal traction/kinematic coupling on interior boundaries.

    For every shared edge segment of the given kind ('core' or 'slide') this
    enforces the normal multiplier coupling
        −∫ [v_n] t_n*  −∫ [u_n]* r_n
    via a ContactBoundary (the [·] = · − ·.Other() gap across the left/right
    pair). No tangential terms are added: DAGBS boundaries slide freely
    (σ_ns = 0), so the tangential multiplier is absent on interior GBs.
    """
    for (i, j), (left_name, right_name) in contact_pairs.items():
        edge = f"{i}_{j}"
        # Core stubs carry a per-junction suffix (core_i_j_lr_jJID); match them
        # all with a regex. Slide is a single segment per side (no suffix).
        if region_kind == "core":
            right_bnd = mesh.Boundaries(f"core_{right_name}.*")
            left_bnd = mesh.Boundaries(f"core_{left_name}.*")
        else:
            right_bnd = mesh.Boundaries(f"slide_{right_name}")
            left_bnd = mesh.Boundaries(f"slide_{left_name}")
        cb = ContactBoundary(right_bnd, left_bnd, volume=False)

        r_n_Re = sym.__dict__[f"r_{edge}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{edge}_n_Im"]
        t_n_Re = sym.__dict__[f"t_{edge}_n_Re"]
        t_n_Im = sym.__dict__[f"t_{edge}_n_Im"]

        term4 = -complexify(
            (t_n_Re, t_n_Im),
            (sym.v_Re * specialcf.normal(2) - sym.v_Re.Other() * specialcf.normal(2),
             sym.v_Im * specialcf.normal(2) - sym.v_Im.Other() * specialcf.normal(2))
        )
        term5 = -complexify(
            (sym.u_Re * specialcf.normal(2) - sym.u_Re.Other() * specialcf.normal(2),
             sym.u_Im * specialcf.normal(2) - sym.u_Im.Other() * specialcf.normal(2)),
            (r_n_Re, r_n_Im)
        )
        cb.AddIntegrator(term4 + term5)
        cb.Update(bf=a, intorder=12, maxdist=1e-9, both_sides=False)


def _add_gb_diffusion(a, mesh, sym, contact_pairs, omega, diff_coeff=None):
    """Add the DAGBS surface-diffusion block on each interior grain boundary.

    Adds (i/ω) ∫ ∂ₛt_n · ∂ₛr_n over the whole boundary (core ∪ slide) as a
    self-term on the normal multiplier field. This is the Coble/Raj-Ashby
    diffusional opening channel; its natural BC is zero diffusive flux at the
    triple junctions (segment ends). Integrated via ds (not ContactBoundary)
    since it couples only the multiplier to itself.
    """
    for (i, j), (left_name, right_name) in contact_pairs.items():
        edge = f"{i}_{j}"
        region_expr = f"core_{right_name}.*|slide_{right_name}"

        r_n_Re = sym.__dict__[f"r_{edge}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{edge}_n_Im"]
        t_n_Re = sym.__dict__[f"t_{edge}_n_Re"]
        t_n_Im = sym.__dict__[f"t_{edge}_n_Im"]

        a += gb_diffusion_block(t_n_Re, t_n_Im, r_n_Re, r_n_Im, omega, diff_coeff) * ds(region_expr)


def _add_gb_junction_coupling(a, mesh, sym, junction_incidence):
    """Exact-Lagrange coupling of the GB pressure field across triple junctions.

    Without this, each interior grain boundary owns an isolated pressure field
    t_n with a zero-flux (natural) BC at its ends, so no matter is transported
    through the junctions: the model is a standard linear solid (E'' ∝ ω, no
    low-frequency attenuation divergence). To recover steady-state Coble creep
    we connect the boundaries at every triple junction J:

      - one reservoir potential  mu_J            (the junction chemical potential)
      - one flux multiplier       lambda_{e,J}    per incident edge e

    The mortar constraint, integrated over edge e's right-side core stub at J,

        ∫_stub  xi_{e,J} · (t_n^(e) − mu_J)  = 0

    enforces continuity  t_n^(e)|_J = mu_J  (stub-averaged); stationarity w.r.t.
    mu_J yields  Σ_e lambda_{e,J} = 0  — Kirchhoff flux balance at the junction.
    lambda_{e,J} is the diffusive flux from edge e into the junction. The blocks
    use complexify_multi (scalar Re/Im saddle pairing), mirroring term4/term5.
    """
    if not junction_incidence:
        return
    for jid, incidents in junction_incidence.items():
        mu_Re = sym.__dict__[f"mu_j{jid}_Re"]
        mu_Im = sym.__dict__[f"mu_j{jid}_Im"]
        rho_Re = sym.__dict__[f"rho_j{jid}_Re"]
        rho_Im = sym.__dict__[f"rho_j{jid}_Im"]
        for (i, j), stub_name in incidents:
            edge = f"{i}_{j}"
            t_n_Re = sym.__dict__[f"t_{edge}_n_Re"]
            t_n_Im = sym.__dict__[f"t_{edge}_n_Im"]
            r_n_Re = sym.__dict__[f"r_{edge}_n_Re"]
            r_n_Im = sym.__dict__[f"r_{edge}_n_Im"]
            lam_Re = sym.__dict__[f"lam_{edge}_j{jid}_Re"]
            lam_Im = sym.__dict__[f"lam_{edge}_j{jid}_Im"]
            xi_Re = sym.__dict__[f"xi_{edge}_j{jid}_Re"]
            xi_Im = sym.__dict__[f"xi_{edge}_j{jid}_Im"]

            # constraint row (tested by xi): t_n^(e) = mu_J on the stub
            constraint = complexify_multi(
                (t_n_Re - mu_Re, t_n_Im - mu_Im),
                (xi_Re, xi_Im),
            )
            # multiplier action (trial lambda) into the t_n / mu equations
            multiplier = complexify_multi(
                (lam_Re, lam_Im),
                (r_n_Re - rho_Re, r_n_Im - rho_Im),
            )
            a += (constraint + multiplier) * ds(stub_name)

def _add_outer_terms(
    a,
    mesh,
    sym,
    outer_contact_pairs,
    omega,
    sliding=True,
):
    """Impose periodic outer-boundary constraints via contact pairs."""
    if not outer_contact_pairs:
        return

    for key, data in outer_contact_pairs.items():
        minus_info = data["minus"]
        plus_info = data["plus"]
        minus_prefix = minus_info["prefix"]
        plus_prefix = plus_info["prefix"]
        edge = key
        region = "slide" if sliding else "core"

        displacement = data.get("displacement")
        disp_norm = None
        if displacement is not None:
            try:
                dx, dy = displacement
                disp_norm = float(np.hypot(float(dx), float(dy)))
            except Exception:
                disp_norm = None

        # Tighten pairing search radius to the expected periodic shift plus a small tolerance.
        maxdist_value = float(disp_norm + 1e-2)

        r_s_Re = sym.__dict__[f"r_{edge}_s_Re"]
        r_s_Im = sym.__dict__[f"r_{edge}_s_Im"]
        t_s_Re = sym.__dict__[f"t_{edge}_s_Re"]
        t_s_Im = sym.__dict__[f"t_{edge}_s_Im"]

        r_n_Re = sym.__dict__[f"r_{edge}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{edge}_n_Im"]
        t_n_Re = sym.__dict__[f"t_{edge}_n_Re"]
        t_n_Im = sym.__dict__[f"t_{edge}_n_Im"]

        term1 = -complexify(
            (t_s_Re, t_s_Im),
            (sym.v_Re * specialcf.tangential(2) - sym.v_Re.Other() * specialcf.tangential(2),
                sym.v_Im * specialcf.tangential(2) - sym.v_Im.Other() * specialcf.tangential(2))
        )
        term2 = -complexify(
            (sym.u_Re * specialcf.tangential(2) - sym.u_Re.Other() * specialcf.tangential(2),
                sym.u_Im * specialcf.tangential(2) - sym.u_Im.Other() * specialcf.tangential(2)),
            (r_s_Re, r_s_Im)
        )

        term4 = -complexify(
            (t_n_Re, t_n_Im),
            (sym.v_Re * specialcf.normal(2) - sym.v_Re.Other() * specialcf.normal(2),
                sym.v_Im * specialcf.normal(2) - sym.v_Im.Other() * specialcf.normal(2))
        )
        term5 = -complexify(
            (sym.u_Re * specialcf.normal(2) - sym.u_Re.Other() * specialcf.normal(2),
                sym.u_Im * specialcf.normal(2) - sym.u_Im.Other() * specialcf.normal(2)),
            (r_n_Re, r_n_Im)
        )

        if region == "core":
            plus_names = _core_boundary_names(plus_info)
            minus_names = _core_boundary_names(minus_info)
            for plus_name, minus_name in zip(plus_names, minus_names):
                plus_region = mesh.Boundaries(plus_name)
                minus_region = mesh.Boundaries(minus_name)
                cb = ContactBoundary(plus_region, minus_region, volume=False)
                cb.AddIntegrator(term1 + term2 + term4 + term5)
                cb.Update(bf=a, intorder=12, maxdist=maxdist_value, both_sides=False)
        else:
            plus_region = mesh.Boundaries(f"{region}_{plus_prefix}")
            minus_region = mesh.Boundaries(f"{region}_{minus_prefix}")
            cb = ContactBoundary(
                plus_region,
                minus_region,
                volume=False
            )
            cb.AddIntegrator(term1 + term2 + term4 + term5)
            cb.Update(bf=a, intorder=12, maxdist=maxdist_value, both_sides=False)

def _assemble_bilinear_form(
    mesh,
    sym,
    contact_pairs,
    outer_contact_pairs,
    lam,
    mu,
    omega,
    fes,
    junction_incidence=None,
    diff_coeff=None,
):
    """Assemble elasticity, GB, and periodic contributions into one bilinear form."""
    a = BilinearForm(fes, check_unused=False)

    elastic_bf = complexify(
        (stress(sym.u_Re, lam, mu), stress(sym.u_Im, lam, mu)),
        (strain(sym.v_Re), strain(sym.v_Im))
    )
    a += elastic_bf * dx

    # Interior grain boundaries: normal coupling over the whole edge (core+slide)
    # plus the diffusional opening channel; tangential is free (no terms).
    _add_gb_normal_coupling(a, mesh, sym, contact_pairs, region_kind="slide")
    _add_gb_normal_coupling(a, mesh, sym, contact_pairs, region_kind="core")
    _add_gb_diffusion(a, mesh, sym, contact_pairs, omega, diff_coeff)
    # Connect the per-edge pressure fields across triple junctions so matter can
    # circulate (steady-state Coble creep -> low-frequency 1/ω attenuation tail).
    _add_gb_junction_coupling(a, mesh, sym, junction_incidence)
    _add_outer_terms(
        a,
        mesh,
        sym,
        outer_contact_pairs,
        omega,
        sliding=True,
    )
    _add_outer_terms(
        a,
        mesh,
        sym,
        outer_contact_pairs,
        omega,
        sliding=False,
    )

    return a

def _add_corner_penalty(a, f, mesh, sym, CF_u, corner_bnd, gammaN=10):
    """Weakly anchor periodic corners to the macro field with a Nitsche-like term."""
    if not corner_bnd:
        return

    h = specialcf.mesh_size
    if isinstance(corner_bnd, str):
        boundary_names = [corner_bnd]
    else:
        try:
            boundary_names = [name for name in corner_bnd if name]
        except TypeError:
            boundary_names = [corner_bnd]

    for name in boundary_names:
        if not name:
            continue
        ds_region = ds(name)

        gamma_weight = (gammaN / h)

        # Bilinear: (gamma_weight) ∫ u_Re·v_Re
        a += gamma_weight * InnerProduct(sym.u_Re, sym.v_Re) * ds_region
        a += gamma_weight * InnerProduct(sym.u_Im, sym.v_Im) * ds_region

        # Linear: (gamma_weight) ∫ CF_u·v_Re
        f += gamma_weight * InnerProduct(CF_u, sym.v_Re) * ds_region


def _add_rbm_constraints(a, f, mesh, sym, CF_u, rot_target):
    """Stress-free rigid-body-mode removal via global integral constraints.

    Replaces the corner Nitsche pin (which injects a fixed, non-relaxing bulk
    stress near the corner and so floors the storage modulus). Instead, three
    global integral constraints + Lagrange multipliers remove the rigid-body
    nullspace without constraining any point, so no stress is injected and the
    relaxed modulus can reach its true value (≈0 for a flowing/creeping state):

        ∫_Ω u_x dΩ = ∫_Ω (Γx)_x dΩ        (translation x)
        ∫_Ω u_y dΩ = ∫_Ω (Γx)_y dΩ        (translation y)
        ∫_Ω (∂ₓu_y − ∂_y u_x) dΩ = (Γ_yx − Γ_xy)·|Ω|   (rotation)

    The rotation target matches the affine field's rotation (simple shear carries
    a rotation component — it is NOT forced to zero). Applied independently to
    the Re and Im displacement fields (the Im targets are 0, the field is real).
    Multipliers live in NumberSpace blocks created in build_spaces.
    """
    def curl(w):  # 2D scalar curl; Grad(w)[i,j] = ∂_j w_i
        g = Grad(w)
        return g[1, 0] - g[0, 1]

    # (name, functional, real-part RHS coefficient)
    specs = [
        ("tx",  lambda w: w[0],   CF_u[0]),
        ("ty",  lambda w: w[1],   CF_u[1]),
        ("rot", lambda w: curl(w), rot_target),
    ]
    for nm, L, rhs in specs:
        for part, u_p, v_p in (("Re", sym.u_Re, sym.v_Re), ("Im", sym.u_Im, sym.v_Im)):
            lam = sym.__dict__[f"rbm_{nm}_{part}"]   # multiplier (trial)
            nu = sym.__dict__[f"rbms_{nm}_{part}"]   # multiplier (test)
            a += nu * L(u_p) * dx          # constraint row (tested by nu)
            a += lam * L(v_p) * dx         # multiplier action into u-equation
            if part == "Re":
                f += nu * rhs * dx         # match affine field (Im target = 0)


def _assemble_linear_form(mesh, sym, Gamma, fes, outer_contact_pairs):
    """Assemble load vector enforcing macro traction jumps on periodic faces."""
    f = LinearForm(fes)
    if not outer_contact_pairs:
        return f

    for key, data in outer_contact_pairs.items():
        plus_info = data["plus"]
        prefix = plus_info["prefix"]
        core_names = _core_boundary_names(plus_info)
        ds_region = core_names + [f"slide_{prefix}"]

        displacement = data.get("displacement")
        if displacement is None:
            raise ValueError(f"Missing displacement vector for outer contact pair {key}.")
        try:
            dx, dy = displacement
            disp = CoefficientFunction((float(dx), float(dy)))
        except Exception as exc:
            raise ValueError(f"Invalid displacement data for outer contact pair {key}: {displacement}") from exc
        strain_jump = Gamma * disp
        jump_normal = InnerProduct(strain_jump, specialcf.normal(2))
        jump_tangential = InnerProduct(strain_jump, specialcf.tangential(2))

        r_n_Re = sym.__dict__[f"r_{key}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{key}_n_Im"]
        r_s_Re = sym.__dict__[f"r_{key}_s_Re"]
        r_s_Im = sym.__dict__[f"r_{key}_s_Im"]
        for name in ds_region:
            f += -jump_normal * r_n_Re * ds(name)
            f += -jump_normal * r_n_Im * ds(name)
            f += -jump_tangential * r_s_Re * ds(name)
            f += -jump_tangential * r_s_Im * ds(name)

    return f

def _solve(a, f, fes, solver, rtol):
    """Solve the saddle system with Pardiso or CG and report the relative residual."""
    from ngsolve.krylovspace import CGSolver

    if solver == "direct":
        with TaskManager():
            a.Assemble()
            f.Assemble()

            gfu = GridFunction(fes)
            gfu.vec.data = a.mat.Inverse(inverse="pardiso") * f.vec

            rel_residual = Norm(a.mat * gfu.vec.data - f.vec.data) / Norm(f.vec.data)
        print(f"Relative residual: {rel_residual:e}")
        convergence = bool(rel_residual < rtol)
        return gfu, convergence

    else:
        with TaskManager():
            a.Assemble()
            f.Assemble()
            c = Preconditioner(a, "multigrid")
            c.Update()

            gfu = GridFunction(fes)
            inv = CGSolver(mat=a.mat, pre=c.mat,
                           printrates='\r', maxiter=50, tol=rtol)
            gfu.vec.data = inv * f.vec

            rel_residual = Norm(a.mat * gfu.vec.data - f.vec.data) / Norm(f.vec.data)
        print(f"Relative residual: {rel_residual:e}")
        convergence = bool(rel_residual < rtol)
        return gfu, convergence

# --- public API ---
def build_spaces(mesh, contact_pairs, outer_contact_pairs=None, order_bulk=2, order_gb=1,
                 junction_incidence=None, remove_rbm=True):
    """Construct the block mixed space holding bulk fields and GB/outer multipliers."""
    # Bulk displacement: no strong Dirichlet; corners handled by Nitsche
    V_Re = VectorH1(mesh, order=order_bulk)
    V_Im = VectorH1(mesh, order=order_bulk)

    gb_spaces = []
    name_order_trial = ["u_Re", "u_Im"]
    name_order_test  = ["v_Re", "v_Im"]

    # Grain-boundary multiplier spaces. DAGBS interior boundaries slide freely
    # (σ_ns = 0), so only the normal multiplier t_n is needed (2 per edge). The
    # H1 field spans core∪slide so its surface gradient (the diffusion term) is
    # continuous across the core/slide split.
    for (a, b), (_, right) in contact_pairs.items():
        bdry = mesh.Boundaries(f"core_{right}.*|slide_{right}")
        gb_spaces += [
            H1(mesh, order=order_gb, definedon=bdry),  # Re_n
            H1(mesh, order=order_gb, definedon=bdry),  # Im_n
        ]
        name_order_trial += [
            f"t_{a}_{b}_n_Re", f"t_{a}_{b}_n_Im"
        ]
        name_order_test += [
            f"r_{a}_{b}_n_Re", f"r_{a}_{b}_n_Im"
        ]
    print(f"Added {len(contact_pairs)*2} grain-boundary multiplier spaces.")

    outer_spaces = []

    combined_outer_pairs = outer_contact_pairs or {}

    for edge_key, data in combined_outer_pairs.items():
        plus_info = data["plus"]
        plus_prefix = plus_info["prefix"]
        core_expr = _core_boundary_expression(plus_info)
        boundary_expr = f"{core_expr}|slide_{plus_prefix}" if core_expr else f"slide_{plus_prefix}"
        bdry = mesh.Boundaries(boundary_expr)
        outer_spaces += [
            H1(mesh, order=order_gb, definedon=bdry),
            H1(mesh, order=order_gb, definedon=bdry),
            H1(mesh, order=order_gb, definedon=bdry),
            H1(mesh, order=order_gb, definedon=bdry),
        ]
        name_order_trial += [
            f"t_{edge_key}_s_Re", f"t_{edge_key}_s_Im",
            f"t_{edge_key}_n_Re", f"t_{edge_key}_n_Im",
        ]
        name_order_test += [
            f"r_{edge_key}_s_Re", f"r_{edge_key}_s_Im",
            f"r_{edge_key}_n_Re", f"r_{edge_key}_n_Im",
        ]
    if combined_outer_pairs:
        print(f"Added {len(combined_outer_pairs) * 4} outer boundary multiplier spaces.")
    else:
        print("No outer boundary multiplier spaces added.")

    # Triple-junction coupling unknowns (exact Lagrange): one reservoir potential
    # mu_J per junction and one flux multiplier lambda_{e,J} per incident edge.
    # All are global scalars (NumberSpace, one DOF each); complex -> Re+Im pair.
    junction_spaces = []
    junctions = junction_incidence or {}
    n_lambda = 0
    for jid in sorted(junctions.keys()):
        junction_spaces += [NumberSpace(mesh), NumberSpace(mesh)]
        name_order_trial += [f"mu_j{jid}_Re", f"mu_j{jid}_Im"]
        name_order_test += [f"rho_j{jid}_Re", f"rho_j{jid}_Im"]
        for (a, b), _stub in junctions[jid]:
            junction_spaces += [NumberSpace(mesh), NumberSpace(mesh)]
            name_order_trial += [f"lam_{a}_{b}_j{jid}_Re", f"lam_{a}_{b}_j{jid}_Im"]
            name_order_test += [f"xi_{a}_{b}_j{jid}_Re", f"xi_{a}_{b}_j{jid}_Im"]
            n_lambda += 1
    if junctions:
        print(f"Added {2*len(junctions)} reservoir + {2*n_lambda} flux-multiplier "
              f"junction spaces ({len(junctions)} triple junctions).")

    # Stress-free rigid-body-mode removal: 3 modes (2 translations + 1 rotation)
    # x (Re, Im) = 6 global scalar Lagrange multipliers, replacing the corner pin.
    rbm_spaces = []
    if remove_rbm:
        for nm in ("tx", "ty", "rot"):
            rbm_spaces += [NumberSpace(mesh), NumberSpace(mesh)]
            name_order_trial += [f"rbm_{nm}_Re", f"rbm_{nm}_Im"]
            name_order_test += [f"rbms_{nm}_Re", f"rbms_{nm}_Im"]
        print("Added 6 rigid-body-mode constraint multipliers (tx,ty,rot)x(Re,Im).")

    fes = V_Re * V_Im
    with TaskManager():
        # build in groups to avoid too-large eval strings; join groups afterwards
        spaces = gb_spaces + outer_spaces + junction_spaces + rbm_spaces
        n = len(spaces)
        group_fes_list = []
        if n:
            group_size = int(sqrt(n)) + 1
            grp_idx = 0
            for start in range(0, n, group_size):
                end = min(start + group_size, n)
                # build the product for this group explicitly (avoid eval)
                group_fes = spaces[start]
                for i in range(start + 1, end):
                    group_fes = group_fes * spaces[i]
                grp_idx += 1
                group_fes_list.append(group_fes)
                # optionally expose fes1, fes2, ... in globals for inspection
                try:
                    globals()[f"fes{grp_idx}"] = group_fes
                except Exception:
                    pass
                print(f"Constructed fes{grp_idx} from spaces {start+1}-{end} of {n}")

            # combine all group finite element spaces into the final fes
            for idx, gf in enumerate(group_fes_list, start=1):
                fes = fes * gf
            print(f"Combined {len(group_fes_list)} groups into fes")
        else:
            print("No additional multiplier spaces to add.")
    print("Constructed finite element space")

    trial_tuple = fes.TrialFunction()
    test_tuple  = fes.TestFunction()

    if len(trial_tuple) != len(name_order_trial):
        raise ValueError(f"Trial arity mismatch: {len(trial_tuple)} vs {len(name_order_trial)}")
    if len(test_tuple) != len(name_order_test):
        raise ValueError(f"Test arity mismatch: {len(test_tuple)} vs {len(name_order_test)}")

    sym_dict = {n: f for n, f in zip(name_order_trial, trial_tuple)}
    sym_dict.update({n: f for n, f in zip(name_order_test, test_tuple)})
    sym = SimpleNamespace(**sym_dict)
    print(f"Constructed finite element space with",fes.ndof," dofs.")

    trial_index = {name: idx for idx, name in enumerate(name_order_trial)}
    gb_normal_indices = {}
    for (a, b) in contact_pairs.keys():
        key = (a, b)
        gb_normal_indices[key] = (
            trial_index[f"t_{a}_{b}_n_Re"],
            trial_index[f"t_{a}_{b}_n_Im"],
        )

    return (fes, V_Re, V_Im, sym, gb_normal_indices)

def solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
    gamma,
    nu=0.30,
    mu=1.0,
    omega=1.0,
    solver='direct',
    rtol=1e-4,
    corner_bnd=None,
    junction_incidence=None,
    remove_rbm=True,
    diff_coeff=None):
    """Solve the DAGBS mixed problem for a given macro loading tensor.

    ``diff_coeff`` overrides the GB diffusion coefficient (default: the
    calibrated ``physics.DIFF_COEFF``); pass 0.0 for the ω→∞ normal-locked
    unrelaxed limit (used to measure G_U).
    """
    Gamma, lam = _setup_material_properties(gamma, nu, mu)

    # Spaces
    fes, V_Re, V_Im, sym = spaces[0], spaces[1], spaces[2], spaces[3]

    # Bilinear & linear forms (elasticity + GB + periodic + junction coupling)
    a = _assemble_bilinear_form(
        mesh,
        sym,
        contact_pairs,
        outer_contact_pairs,
        lam,
        mu,
        omega,
        fes,
        junction_incidence=junction_incidence,
        diff_coeff=diff_coeff,
    )
    f = _assemble_linear_form(
        mesh,
        sym,
        Gamma,
        fes,
        outer_contact_pairs,
    )

    # Macro affine displacement
    disp = CoefficientFunction((x, y))
    CF_u = Gamma * disp

    if remove_rbm:
        # Stress-free RBM removal (default). Rotation target = affine rotation
        # (Γ_yx − Γ_xy) × MACRO_SCALE; for simple shear this is nonzero.
        rot_target = (gamma[1][0] - gamma[0][1]) * 1e-3
        _add_rbm_constraints(a, f, mesh, sym, CF_u, rot_target)
    else:
        _add_corner_penalty(a, f, mesh, sym, CF_u, corner_bnd)
    gfu, convergence = _solve(a, f, fes, solver, rtol)

    return gfu, mesh, convergence

