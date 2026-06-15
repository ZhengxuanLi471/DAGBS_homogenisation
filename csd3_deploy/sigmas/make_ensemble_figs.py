"""Per-sigma fig2-style ensemble figures for every sigma_* sweep.

For each sigma label it produces 4 figures =
    {all-curves, mean +-1sigma} x {normalized by individual tau_M, by hex tau_M},
each a 4-panel reproduce_fig2 layout:
    (a) |G*|/G0   (b) Q^-1=G2/G1   (c) scaled compliances   (d) Cole-Cole
plus a per-sigma mean CSV (fig2_ensemble_mean.csv) consumed by overlay_all_sigma.py.

The regular-hexagon reference is geometry-only (independent of sigma), so the
single nu=0.35 hex reference generated under sigma_0.05/ is reused for every sigma.
tau_M per seed is parsed from the SLURM .out files (built once per dir if missing).

Frequency axis is omega*tau_M: the seed CSV `omega` column already equals
omega*tau_M(seed) (the sweep ran with diff_coeff=tau_M), so the "individual" set
plots vs that column and the "hex" set rescales each seed by tau_M_hex/tau_M(seed).

NOTE: the seed grid bottoms out at omega*tau_M ~ 0.05, so Delta / the steady-state
modulus used by panels (c)/(d) are estimated from only the lowest few points and
are resolution-limited; panels (a)/(b) are robust.

Run with the ngsolve venv python (no ngsolve import here, but keep it uniform):
    /home/zl471/miniconda3/envs/ngsolve/bin/python make_ensemble_figs.py [label ...]
With no args it processes every complete sigma dir in SIGMA_LABELS.
"""
import glob
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
HEX_DIR = os.path.join(ROOT, "sigma_0.05")          # shared hex reference lives here
NU = 0.35
SIGMA_LABELS = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]

# --- shared hexagon reference (sigma-independent) ---------------------------
with open(os.path.join(HEX_DIR, "hex_reference_tau.txt")) as fh:
    TAU_HEX = float(fh.read().strip())
HEX_DF = pd.read_csv(os.path.join(HEX_DIR, "hex_reference_shear.csv"))

# --- styling shared with reproduce_fig2 ------------------------------------
HEXST = dict(color="k", lw=2.2, zorder=5)
SEEDST = dict(color="C0", lw=0.6, alpha=0.22)
MEANST = dict(color="C0", lw=2)
COMP_C1 = "C1"
# seeds whose panel-(b) Q^-1 tail flattens (non-asymptotic high-omega gradient)
FLAGST = dict(color="crimson", lw=1.1, alpha=0.9, zorder=4)
# seeds with a pronounced high-omega Q^-1 hump that then recovers the asymptote
BUMPST = dict(color="forestgreen", lw=1.1, alpha=0.9, zorder=3)

# --- non-asymptotic Q^-1 tail detectors ------------------------------------
# Most seeds settle to a power-law asymptote of slope ~ -1/3 in log Q^-1 vs
# log(omega*tau_M). Two distinct high-omega departures are flagged, mutually
# exclusive, each its own colour:
#
#  (1) FLATTEN-AND-STAY (crimson, `flag`): the slope over the last TAIL_NPTS
#      points rises above TAIL_SLOPE_THR -- the tail flattens toward 0 and never
#      recovers within the swept band. TAIL_SLOPE_THR=-0.25 is the ~95th pct of
#      the all-seed slope distribution (bulk peaks tightly at -0.33, sd~0.03).
#
#  (2) BUMP-AND-RETURN (forestgreen, `bump`): a pronounced hump rises above the
#      ensemble median in the high-omega band (omega*tau_M > BUMP_WT_MIN) but the
#      tail re-steepens back to a near-asymptotic slope -- so by construction it
#      is NOT also flattened. Flagged when the peak log10(Q^-1) excess over the
#      ensemble median in that band exceeds BUMP_PEAK_THR (~90th pct of the
#      non-flat seeds; ~top 10%). The user's canonical cases are the largest
#      humps in the set: sigma=0.05 seed_45 (excess 0.29) and sigma=0.25 seed_8
#      (0.21).
#
# Both are likely the same high-omega mesh-resolution effect (the diffusion
# boundary layer ~sqrt(D/omega) goes unresolved, worse at high sigma where grains
# are more distorted). Working hypothesis (left untested for now): the crimson
# flatten-and-stay tails may just be a bump-and-return hump whose recovery falls
# beyond the swept band -- i.e. an "incomplete hump", the same phenomenon as (2).
TAIL_NPTS = 4
TAIL_SLOPE_THR = -0.25
BUMP_WT_MIN = 100.0
BUMP_PEAK_THR = 0.10


def tail_slope(x, Qinv):
    """Slope of log10(Q^-1) vs log10(omega) over the last TAIL_NPTS points."""
    lx = np.log10(x[-TAIL_NPTS:])
    lq = np.log10(Qinv[-TAIL_NPTS:])
    return float(np.polyfit(lx, lq, 1)[0])


def classify_bumps(seeds):
    """Tag seeds with a pronounced high-omega Q^-1 hump (`bump`, `peak_excess`).

    Needs the whole ensemble: the hump is measured as the peak excess of
    log10(Q^-1) over the per-frequency ensemble median in the omega*tau_M >
    BUMP_WT_MIN band. Mutually exclusive with `flag` (flatten-and-stay) -- a
    seed already flagged crimson is never also tagged green.
    """
    x = seeds[0]["x"]                       # shared grid (all seeds same omega)
    LQ = np.vstack([np.log10(s["Qinv"]) for s in seeds])
    med = np.median(LQ, axis=0)
    hi = x > BUMP_WT_MIN
    for i, s in enumerate(seeds):
        s["peak_excess"] = float((LQ[i, hi] - med[hi]).max())
        s["bump"] = (not s["flag"]) and s["peak_excess"] > BUMP_PEAK_THR

_seed_re = re.compile(r"Processing\s+(seeds_\d+)")
_result_re = re.compile(
    r"RESULT:\s*tau_M\s*=\s*eta_ss/G_U\s*=\s*([0-9.eE+-]+)\s*"
    r"\(ln\(1/tau_M\)\s*=\s*([0-9.eE+-]+)\)")


def build_tau_csv(sdir, label):
    """Parse tau_M per seed from the SLURM .out files into tau_M_sigma_<label>.csv.

    A sigma dir accumulates .out files from BOTH the base sweep and the low-omega
    sweep (two job arrays), each re-measuring tau_M. We keep ONE row per seed, from
    the lowest SLURM job id -- i.e. the base sweep, whose tau_M is the one the base
    *_shear.csv `omega` column was generated with (the low-omega re-calibration can
    diverge; e.g. sigma_0.5 seed 54 measured 2.6e-8 vs the base 3.9e-6)."""
    def job_id(path):
        m = re.search(r"slurm-(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 1 << 62
    best = {}   # seed -> (job_id, row)
    for path in sorted(glob.glob(os.path.join(sdir, "slurm-*.out")), key=job_id):
        with open(path) as fh:
            text = fh.read()
        m_res = _result_re.search(text)
        if not m_res:
            continue
        m_seed = _seed_re.search(text)
        seed = m_seed.group(1) if m_seed else None
        jid = job_id(path)
        row = (seed, float(m_res.group(1)), float(m_res.group(2)),
               os.path.basename(path))
        if seed not in best or jid < best[seed][0]:   # keep lowest job id = base run
            best[seed] = (jid, row)
    rows = [v[1] for v in best.values()]
    rows.sort(key=lambda r: int(r[0].split("_")[1]) if r[0] else 1 << 30)
    out = os.path.join(sdir, f"tau_M_sigma_{label}.csv")
    pd.DataFrame(rows, columns=["seed", "tau_M", "ln_inv_tau_M",
                                "source_file"]).to_csv(out, index=False)
    print(f"  built {os.path.basename(out)} ({len(rows)} unique seeds)")
    return out


def derive(df):
    """Return (x_native, absG, Qinv, sJ1, sJ2, Delta) for one sweep CSV."""
    G1 = df["Cxyxy_real"].to_numpy()
    G2 = df["Cxyxy_imag"].to_numpy()
    x = df["omega"].to_numpy()
    Gc = G1 + 1j * G2
    G0 = G1[-1]                          # unrelaxed plateau at the highest omega
    low = x < 0.1
    if low.sum() < 3:
        low = np.zeros_like(x, dtype=bool)
        low[:3] = True
    G_ss = float(np.median(G2[low] ** 2 / G1[low]))
    Delta = G0 / G_ss - 1.0
    jn = G0 / Gc
    absG = np.abs(Gc) / G0
    Qinv = G2 / G1
    sJ1 = (jn.real - 1.0) / Delta
    sJ2 = (-jn.imag - 1.0 / x) / Delta
    return x, absG, Qinv, sJ1, sJ2, Delta


def load_seeds(sdir, tau_of):
    files = sorted(
        glob.glob(os.path.join(sdir, "Seed_seeds_*_energy_real_im_data_shear.csv")),
        key=lambda p: int(re.search(r"seeds_(\d+)_", p).group(1)))
    seeds = []
    for path in files:
        name = re.search(r"(seeds_\d+)_energy", path).group(1)
        df = pd.read_csv(path)
        x, absG, Qinv, sJ1, sJ2, Delta = derive(df)
        s_tail = tail_slope(x, Qinv)
        seeds.append(dict(name=name, x=x, absG=absG, Qinv=Qinv, sJ1=sJ1, sJ2=sJ2,
                          Delta=Delta, tau=tau_of.get(name, np.nan),
                          G1=df["Cxyxy_real"].to_numpy(),
                          G2=df["Cxyxy_imag"].to_numpy(),
                          tail_slope=s_tail, flag=s_tail > TAIL_SLOPE_THR))
    # The shared-grid mean/classify assume every seed is on the same frequency grid.
    # A seed that could not be low-omega-extended (its tail was a divergent solve and
    # was quarantined by merge_lowomega.py, e.g. sigma_0.2/62, sigma_0.5/54) keeps the
    # shorter base-only grid. Keep only seeds on the modal grid length and drop the
    # odd ones with a warning rather than crash the vstack.
    if seeds:
        from collections import Counter
        lens = Counter(len(s["x"]) for s in seeds)
        modal_len = lens.most_common(1)[0][0]
        dropped = [s["name"] for s in seeds if len(s["x"]) != modal_len]
        if dropped:
            print(f"  dropping {len(dropped)} off-grid seed(s) "
                  f"(not on modal grid len {modal_len}): "
                  f"{[d.split('_')[1] for d in dropped]}")
            seeds = [s for s in seeds if len(s["x"]) == modal_len]
    return seeds


def style_axes(ax):
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


def process(label):
    sdir = os.path.join(ROOT, f"sigma_{label}")
    print(f"sigma={label}")
    if not os.path.isdir(sdir):
        print(f"  no dir {sdir} -- skipping")
        return
    tau_csv = os.path.join(sdir, f"tau_M_sigma_{label}.csv")
    if not os.path.exists(tau_csv):
        build_tau_csv(sdir, label)
    if not os.path.exists(tau_csv):
        print(f"  no tau CSV / no SLURM .out in {sdir} -- skipping")
        return
    tau_df = pd.read_csv(tau_csv)
    tau_of = {row.seed: float(row.tau_M) for row in tau_df.itertuples()}

    seeds = load_seeds(sdir, tau_of)
    if not seeds:
        print(f"  no seed CSVs in {sdir} -- skipping")
        return
    classify_bumps(seeds)
    N = len(seeds)
    hx, h_absG, h_Qinv, h_sJ1, h_sJ2, h_Delta = derive(HEX_DF)
    XGRID = seeds[0]["x"]

    def plot_hex(ax):
        ax[0, 0].plot(hx, h_absG, label="hex ref", **HEXST)
        ax[0, 1].plot(hx, h_Qinv, **HEXST)
        if h_Delta > 0:
            ax[1, 0].plot(hx, h_sJ1, label=r"hex $sJ_1$", **HEXST)
            ax[1, 0].plot(hx, h_sJ2, ls="--", **HEXST)
            ax[1, 1].plot(h_sJ1, h_sJ2, **HEXST)

    def xplot(s, norm):
        return s["x"] if norm == "indiv" else s["x"] * (TAU_HEX / s["tau"])

    # --- all-curves figures ---
    def seed_style(s):
        if s["flag"]:
            return FLAGST
        if s["bump"]:
            return BUMPST
        return SEEDST

    def fig_allcurves(norm):
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
        # draw order: normal (blue) -> bump (green) -> flat (crimson) on top
        for s in sorted(seeds, key=lambda s: 2 if s["flag"] else (1 if s["bump"] else 0)):
            xp = xplot(s, norm)
            st = seed_style(s)
            ax[0, 0].plot(xp, s["absG"], **st)
            ax[0, 1].plot(xp, s["Qinv"], **st)
            if s["Delta"] > 0:
                comp_st = (st if (s["flag"] or s["bump"])
                           else {**SEEDST, "color": COMP_C1})
                ax[1, 0].plot(xp, s["sJ1"], **st)
                ax[1, 0].plot(xp, s["sJ2"], **comp_st)
                ax[1, 1].plot(s["sJ1"], s["sJ2"], **st)
        plot_hex(ax)
        style_axes(ax)
        n_flag = sum(s["flag"] for s in seeds)
        n_bump = sum(s["bump"] for s in seeds)
        ax[0, 0].plot([], [], **SEEDST,
                      label=f"{N - n_flag - n_bump} asymptotic seeds")
        if n_flag:
            ax[0, 1].plot([], [], **FLAGST,
                          label=f"{n_flag} flatten-and-stay "
                                f"(tail slope$>{TAIL_SLOPE_THR}$)")
        if n_bump:
            ax[0, 1].plot([], [], **BUMPST,
                          label=f"{n_bump} bump-and-return "
                                f"(excess$>{BUMP_PEAK_THR}$)")
        if n_flag or n_bump:
            ax[0, 1].legend(fontsize=7, loc="lower left")
        ax[0, 0].legend(fontsize=8)
        nl = r"individual $\tau_M$" if norm == "indiv" else r"hex $\tau_M$"
        finish(fig, f"DAGBS sigma={label}, nu={NU}: {N} seeds + hex (all curves) "
                    f"— freq normalized by {nl}  (tau_hex={TAU_HEX:.3e})")
        out = os.path.join(sdir, f"fig2_ens_allcurves_"
                           f"{'indivtau' if norm == 'indiv' else 'hextau'}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("  saved", os.path.basename(out))

    # --- mean +- 1sigma figures ---
    def stack_indiv(key):
        rows = ([s[key] for s in seeds if s["Delta"] > 0]
                if key in ("sJ1", "sJ2") else [s[key] for s in seeds])
        A = np.vstack(rows)
        return np.nanmean(A, axis=0), np.nanstd(A, axis=0)

    def stack_hex(key, xc):
        rows = []
        for s in seeds:
            if key in ("sJ1", "sJ2") and s["Delta"] <= 0:
                continue
            xp = s["x"] * (TAU_HEX / s["tau"])
            rows.append(np.interp(np.log10(xc), np.log10(xp), s[key],
                                  left=np.nan, right=np.nan))
        A = np.vstack(rows)
        return np.nanmean(A, axis=0), np.nanstd(A, axis=0)

    def fig_mean(norm):
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
        xc = XGRID
        st = stack_indiv if norm == "indiv" else (lambda k: stack_hex(k, xc))
        m_absG, s_absG = st("absG")
        m_Q, s_Q = st("Qinv")
        m_J1, s_J1 = st("sJ1")
        m_J2, s_J2 = st("sJ2")

        def band(a_, x, m, s, **kw):
            a_.plot(x, m, **{**MEANST, **kw})
            a_.fill_between(x, m - s, m + s, color=kw.get("color", "C0"), alpha=0.2)

        band(ax[0, 0], xc, m_absG, s_absG, label="seed mean")
        band(ax[0, 1], xc, m_Q, s_Q)
        band(ax[1, 0], xc, m_J1, s_J1, label=r"$sJ_1$")
        band(ax[1, 0], xc, m_J2, s_J2, color=COMP_C1, label=r"$sJ_2$")
        ax[1, 1].plot(m_J1, m_J2, **MEANST)
        plot_hex(ax)
        style_axes(ax)
        ax[0, 0].legend(fontsize=7)
        ax[1, 0].legend(fontsize=7)
        nl = r"individual $\tau_M$" if norm == "indiv" else r"hex $\tau_M$"
        finish(fig, f"DAGBS sigma={label}, nu={NU}: seed mean +-1sigma (N={N}) "
                    f"+ hex — freq norm by {nl}")
        out = os.path.join(sdir, f"fig2_ens_mean_"
                           f"{'indivtau' if norm == 'indiv' else 'hextau'}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("  saved", os.path.basename(out))
        return xc, m_absG, m_Q, m_J1, m_J2

    fig_allcurves("indiv")
    fig_allcurves("hex")

    # record the non-asymptotic Q^-1 seeds for this sigma (both categories)
    flagged = sorted((s for s in seeds if s["flag"]),
                     key=lambda s: s["tail_slope"], reverse=True)
    bumped = sorted((s for s in seeds if s["bump"]),
                    key=lambda s: s["peak_excess"], reverse=True)
    fl_path = os.path.join(sdir, f"qinv_nonasymptotic_seeds_{label}.csv")
    pd.DataFrame(
        [(s["name"], "flatten", round(s["tail_slope"], 4),
          round(s["peak_excess"], 4)) for s in flagged]
        + [(s["name"], "bump", round(s["tail_slope"], 4),
            round(s["peak_excess"], 4)) for s in bumped],
        columns=["seed", "category", "tail_slope", "peak_excess"],
    ).to_csv(fl_path, index=False)
    print(f"  {len(flagged)}/{N} flatten-and-stay (slope>{TAIL_SLOPE_THR}): "
          f"{[s['name'].split('_')[1] for s in flagged]}")
    print(f"  {len(bumped)}/{N} bump-and-return (excess>{BUMP_PEAK_THR}): "
          f"{[s['name'].split('_')[1] for s in bumped]}")

    xc_i, a_i, q_i, j1_i, j2_i = fig_mean("indiv")
    xc_h, a_h, q_h, j1_h, j2_h = fig_mean("hex")

    pd.DataFrame({
        "omega_tauM": xc_i,
        "mean_absG_over_G0_indiv": a_i, "mean_Qinv_indiv": q_i,
        "mean_sJ1_indiv": j1_i, "mean_sJ2_indiv": j2_i,
        "mean_absG_over_G0_hex": a_h, "mean_Qinv_hex": q_h,
        "mean_sJ1_hex": j1_h, "mean_sJ2_hex": j2_h,
    }).to_csv(os.path.join(sdir, "fig2_ensemble_mean.csv"), index=False)
    print(f"  saved fig2_ensemble_mean.csv  (N={N})")


if __name__ == "__main__":
    labels = sys.argv[1:] or SIGMA_LABELS
    print(f"hex reference: tau_hex={TAU_HEX:.4e}")
    for lab in labels:
        process(lab)
