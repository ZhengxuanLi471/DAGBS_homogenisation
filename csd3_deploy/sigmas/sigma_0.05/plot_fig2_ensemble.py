"""fig2-style ensemble plots for the sigma=0.05 seed sweep vs the hex reference.

Produces 4 figures = {all-curves, mean+-1sigma} x {normalized by individual tau_M,
normalized by hex tau_M}, each a 4-panel reproduce_fig2 layout:
    (a) |G*|/G0   (b) Q^-1=G2/G1   (c) scaled compliances   (d) Cole-Cole
The hexagon reference curve is overlaid on every figure.

Derived-quantity math mirrors reproduce_fig2.py:derive().  Frequency axis is
omega*tau_M: the seed CSV `omega` column already equals omega*tau_M(seed) (the
sweep ran with diff_coeff=tau_M), so the "individual" set plots vs that column,
and the "hex" set rescales each seed by tau_M_hex/tau_M(seed).

NOTE: the seed grid bottoms out at omega*tau_M ~ 0.05, so the relaxation strength
Delta / steady-state modulus used by panels (c) and (d) are estimated from only
the lowest few points and are resolution-limited; panels (a)/(b) are robust.
"""
import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.special import gamma as Gammafn, hyp2f1
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMA = "0.05"
NU = 0.35
ALPHA = 0.3672092      # Andrade exponent for the 120-deg triple junction (as fig2)


def andrade_fit(x, Gc, G0, alpha, window=(300.0, 5000.0)):
    """tau_A/tau_M from the high-omega Andrade form (reproduce_fig2 helper)."""
    Ga = Gammafn(1.0 + alpha)
    x = np.asarray(x, float); Gc = np.asarray(Gc, complex)
    m = (x >= window[0]) & (x <= window[1]) & np.isfinite(Gc)
    xm, Gm = x[m], Gc[m]
    def resid(r):
        model = G0 / (1.0 + Ga * (1j * xm * r) ** (-alpha))
        return float(np.sum(np.abs(model - Gm) ** 2))
    return float(minimize_scalar(resid, bounds=(0.1, 8.0), method="bounded").x)


def model_quantities(Gc, x, G0, Delta):
    """4-panel quantities from a complex modulus (same math as derive())."""
    jn = G0 / Gc
    absG = np.abs(Gc) / G0
    Qinv = (-jn.imag) / jn.real
    sJ1 = (jn.real - 1.0) / Delta
    sJ2 = (-jn.imag - 1.0 / x) / Delta
    return absG, Qinv, sJ1, sJ2


def build_analytic_models(x, G0, Delta):
    """Maxwell complex modulus on grid x (fig2 form).

    Andrade and extended-Burgers overlays were removed: both rely on the fixed
    Andrade exponent alpha=0.367, which is the symmetric 120-deg triple-junction
    value (regular hex only). Random seeds have a distribution of junction angles,
    so imposing that alpha on the ensemble is not justified. Maxwell needs only
    G0 and Delta (no alpha), so it is the one analytic overlay kept.
    """
    iw = 1j * x
    jn_max = (1 + Delta) + 1.0 / iw
    return {
        "Maxwell": (G0 / jn_max, dict(color="C4", ls="-.", lw=1.3)),
    }


# --- load tau_M per seed ----------------------------------------------------
tau_df = pd.read_csv(os.path.join(HERE, "tau_M_sigma_0.05.csv"))
tau_of = {row.seed: float(row.tau_M) for row in tau_df.itertuples()}

# --- hex reference ----------------------------------------------------------
with open(os.path.join(HERE, "hex_reference_tau.txt")) as fh:
    TAU_HEX = float(fh.read().strip())
hex_df = pd.read_csv(os.path.join(HERE, "hex_reference_shear.csv"))


def derive(df):
    """Return (x_native, absG, Qinv, sJ1, sJ2, Delta) for one sweep CSV.

    x_native = omega*tau_M(self); G0 = unrelaxed (high-omega) storage plateau;
    Delta = G0/G_ss - 1 from the low-omega plateau of G2^2/G1.
    """
    G1 = df["Cxyxy_real"].to_numpy()
    G2 = df["Cxyxy_imag"].to_numpy()
    x = df["omega"].to_numpy()
    Gc = G1 + 1j * G2
    G0 = G1[-1]                       # unrelaxed plateau at the highest omega
    low = x < 0.1
    if low.sum() < 3:
        low = np.zeros_like(x, dtype=bool)
        low[:3] = True
    G_ss = float(np.median(G2[low] ** 2 / G1[low]))
    Delta = G0 / G_ss - 1.0
    jn = G0 / Gc                      # = G0 J*
    absG = np.abs(Gc) / G0
    Qinv = G2 / G1
    sJ1 = (jn.real - 1.0) / Delta
    sJ2 = (-jn.imag - 1.0 / x) / Delta
    return x, absG, Qinv, sJ1, sJ2, Delta


# --- compute per-seed curves ------------------------------------------------
files = sorted(glob.glob(os.path.join(HERE, "Seed_seeds_*_energy_real_im_data_shear.csv")),
               key=lambda p: int(re.search(r"seeds_(\d+)_", p).group(1)))
seeds = []                            # list of dicts
n_bad_delta = 0
for path in files:
    m = re.search(r"(seeds_\d+)_energy", path)
    name = m.group(1)
    df = pd.read_csv(path)
    x, absG, Qinv, sJ1, sJ2, Delta = derive(df)
    tau = tau_of.get(name, np.nan)
    if Delta <= 0:
        n_bad_delta += 1
    seeds.append(dict(name=name, x=x, absG=absG, Qinv=Qinv, sJ1=sJ1, sJ2=sJ2,
                      Delta=Delta, tau=tau,
                      G1=df["Cxyxy_real"].to_numpy(), G2=df["Cxyxy_imag"].to_numpy()))

hx, h_absG, h_Qinv, h_sJ1, h_sJ2, h_Delta = derive(hex_df)
print(f"Loaded {len(seeds)} seeds (Delta<=0 on {n_bad_delta}); "
      f"tau_hex={TAU_HEX:.4e}, Delta_hex={h_Delta:.3f}")

# native grids are identical across seeds (same ln_omega sweep)
XGRID = seeds[0]["x"]
assert all(np.allclose(s["x"], XGRID) for s in seeds), "seed omega grids differ"


# --- styling shared with reproduce_fig2 ------------------------------------
HEXST = dict(color="k", lw=2.2, zorder=5)
SEEDST = dict(color="C0", lw=0.6, alpha=0.22)
MEANST = dict(color="C0", lw=2)
COMP_C1 = "C1"                        # second compliance / second hex line


def style_axes(ax, Delta_note):
    ax[0, 0].set(xscale="log", xlabel=r"$\omega\tau_M$", ylabel=r"$|G^*|/G_0$",
                 ylim=(0, 1.05))
    ax[0, 1].set(xscale="log", yscale="log", xlabel=r"$\omega\tau_M$",
                 ylabel=r"$Q^{-1}$")
    ax[1, 0].set(xscale="log", xlabel=r"$\omega\tau_M$",
                 ylabel="Scaled compliance", ylim=(0, 1.05))
    ax[1, 1].set(xlabel=r"$(G_0 J_1-1)/\Delta$",
                 ylabel=r"$(G_0 J_2-1/\omega\tau_M)/\Delta$",
                 xlim=(0, 1), ylim=(0, 0.35))
    for a_ in ax.ravel():
        a_.grid(True, which="both", alpha=0.15)


def finish(fig, title):
    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.005,
             "panels (c)/(d) resolution-limited: seed grid only reaches "
             r"$\omega\tau_M\approx0.05$, so $\Delta$ is poorly constrained",
             ha="center", fontsize=7, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 1))


def plot_hex(ax):
    ax[0, 0].plot(hx, h_absG, label="hex ref", **HEXST)
    ax[0, 1].plot(hx, h_Qinv, **HEXST)
    if h_Delta > 0:
        ax[1, 0].plot(hx, h_sJ1, label=r"hex $sJ_1$", **HEXST)
        ax[1, 0].plot(hx, h_sJ2, ls="--", **HEXST)
        ax[1, 1].plot(h_sJ1, h_sJ2, **HEXST)


# ---------------------------------------------------------------------------
# ALL-CURVES figures
# ---------------------------------------------------------------------------
def xplot(s, norm):
    return s["x"] if norm == "indiv" else s["x"] * (TAU_HEX / s["tau"])


def fig_allcurves(norm):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    for s in seeds:
        xp = xplot(s, norm)
        ax[0, 0].plot(xp, s["absG"], **SEEDST)
        ax[0, 1].plot(xp, s["Qinv"], **SEEDST)
        if s["Delta"] > 0:
            ax[1, 0].plot(xp, s["sJ1"], **SEEDST)
            ax[1, 0].plot(xp, s["sJ2"], **{**SEEDST, "color": COMP_C1})
            ax[1, 1].plot(s["sJ1"], s["sJ2"], **SEEDST)
    plot_hex(ax)
    style_axes(ax, None)
    ax[0, 0].plot([], [], **SEEDST, label="98 seeds")
    ax[0, 0].legend(fontsize=8)
    normlabel = (r"individual $\tau_M$" if norm == "indiv"
                 else r"hex $\tau_M$")
    finish(fig, f"DAGBS sigma={SIGMA}, nu={NU}: 98 seeds + hex (all curves) "
                f"— freq normalized by {normlabel}  (tau_hex={TAU_HEX:.3e})")
    out = os.path.join(HERE, f"fig2_ens_allcurves_{'indivtau' if norm=='indiv' else 'hextau'}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", os.path.basename(out))


# ---------------------------------------------------------------------------
# MEAN +- 1 sigma figures
# ---------------------------------------------------------------------------
def stack_indiv(key):
    """Stack a per-seed quantity on the shared native grid (Delta-guarded for c)."""
    if key in ("sJ1", "sJ2"):
        rows = [s[key] for s in seeds if s["Delta"] > 0]
    else:
        rows = [s[key] for s in seeds]
    A = np.vstack(rows)
    return np.nanmean(A, axis=0), np.nanstd(A, axis=0)


def stack_hex(key, xc):
    """Interpolate each seed onto common log-grid xc, then mean/std."""
    rows = []
    for s in seeds:
        if key in ("sJ1", "sJ2") and s["Delta"] <= 0:
            continue
        xp = s["x"] * (TAU_HEX / s["tau"])
        y = np.interp(np.log10(xc), np.log10(xp), s[key],
                      left=np.nan, right=np.nan)
        rows.append(y)
    A = np.vstack(rows)
    return np.nanmean(A, axis=0), np.nanstd(A, axis=0)


def fig_mean(norm):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    if norm == "indiv":
        xc = XGRID
        m_absG, s_absG = stack_indiv("absG")
        m_Q, s_Q = stack_indiv("Qinv")
        m_J1, s_J1 = stack_indiv("sJ1")
        m_J2, s_J2 = stack_indiv("sJ2")
        mG1, _ = stack_indiv("G1"); mG2, _ = stack_indiv("G2")
    else:
        xc = XGRID                    # common grid spans the native range
        m_absG, s_absG = stack_hex("absG", xc)
        m_Q, s_Q = stack_hex("Qinv", xc)
        m_J1, s_J1 = stack_hex("sJ1", xc)
        m_J2, s_J2 = stack_hex("sJ2", xc)
        mG1, _ = stack_hex("G1", xc); mG2, _ = stack_hex("G2", xc)

    def band(a_, x, m, s, **kw):
        a_.plot(x, m, **{**MEANST, **kw})
        a_.fill_between(x, m - s, m + s, color=kw.get("color", "C0"), alpha=0.2)

    band(ax[0, 0], xc, m_absG, s_absG, label="seed mean")
    band(ax[0, 1], xc, m_Q, s_Q)
    band(ax[1, 0], xc, m_J1, s_J1, label=r"$sJ_1$")
    band(ax[1, 0], xc, m_J2, s_J2, color=COMP_C1, label=r"$sJ_2$")
    ax[1, 1].plot(m_J1, m_J2, **MEANST)    # mean Cole-Cole trajectory

    # Analytic overlays (Maxwell / Andrade / extended-Burgers) removed: the
    # Andrade/Burgers forms rely on the fixed alpha=0.367 (120-deg hex junction),
    # which is not justified for the random-angle seeds, and Maxwell was not
    # informative. The mean plots now show only the seed mean +-1sigma vs hex.

    plot_hex(ax)
    style_axes(ax, None)
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    normlabel = (r"individual $\tau_M$" if norm == "indiv" else r"hex $\tau_M$")
    finish(fig, f"DAGBS sigma={SIGMA}, nu={NU}: seed mean +-1sigma (N={len(seeds)}) + hex "
                f"— freq norm by {normlabel}")
    out = os.path.join(HERE, f"fig2_ens_mean_{'indivtau' if norm=='indiv' else 'hextau'}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {os.path.basename(out)}")
    return xc, m_absG, m_Q, m_J1, m_J2


# ---------------------------------------------------------------------------
fig_allcurves("indiv")
fig_allcurves("hex")
xc_i, a_i, q_i, j1_i, j2_i = fig_mean("indiv")
xc_h, a_h, q_h, j1_h, j2_h = fig_mean("hex")

pd.DataFrame({
    "omega_tauM": xc_i,
    "mean_absG_over_G0_indiv": a_i, "mean_Qinv_indiv": q_i,
    "mean_sJ1_indiv": j1_i, "mean_sJ2_indiv": j2_i,
    "mean_absG_over_G0_hex": a_h, "mean_Qinv_hex": q_h,
    "mean_sJ1_hex": j1_h, "mean_sJ2_hex": j2_h,
}).to_csv(os.path.join(HERE, "fig2_ensemble_mean.csv"), index=False)
print("saved fig2_ensemble_mean.csv")
