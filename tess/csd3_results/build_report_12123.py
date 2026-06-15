#!/usr/bin/env python3
"""Build a single self-contained HTML report for the (3,12,12) ratio-sweep study.

Embeds every figure as base64 (portable: emailable, opens anywhere), reads the
geometry table (ratio, triangle facet L, tau_M, G_U, mesh ne) from the pulled
CSD3 _meta.txt files, and templates a 4-section report:
  rationale -> geometry -> numerical experiments -> conclusion.

Numbers (scaling-fit slopes, attribution shares) come from the analysis scripts
in this folder (analyze_12123.py / analyze_wide_12123.py / scaling_final_12123.py
/ attribute_12123.py); they are quoted here, with the source script named.

Stdlib only. Re-runnable:  python build_report_12123.py
"""
import base64
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TESS = os.path.normpath(os.path.join(HERE, ".."))
SW = os.path.join(HERE, "12123_sweep")
OUT = os.path.join(HERE, "report_12123.html")

# ----------------------------------------------------------------- assets
FIGS = {
    "sweep":     os.path.join(TESS, "12123_sweep", "sweep_overview.png"),
    "mesh_fe":   os.path.join(TESS, "demo_figures", "trunc_hex_31212_mesh_finite_elements.png"),
    "mesh_bnd":  os.path.join(TESS, "demo_figures", "trunc_hex_31212_mesh_boundaries.png"),
    "individual": os.path.join(HERE, "individual_12123.png"),
    "scaling0":  os.path.join(HERE, "scaling_12123.png"),
    "wideB":     os.path.join(HERE, "wideB_12123.png"),
    "wideC":     os.path.join(HERE, "wideC_12123.png"),
    "attrib":    os.path.join(HERE, "attribution_12123.png"),
    "final":     os.path.join(HERE, "scaling_final_12123.png"),
}


def img(key, width="100%"):
    with open(FIGS[key], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{width};max-width:100%;display:block;margin:0 auto;">')


# ----------------------------------------------------------------- geometry table
def triangle_facet(pts, regions):
    pa = [(p[0], p[1]) for p in pts]
    best = None
    for r in regions:
        if len(r) != 3:
            continue
        per = 0.0
        for k in range(3):
            a = pa[r[k] - 1]; b = pa[r[(k + 1) % 3] - 1]
            per += ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        L = per / 3.0
        best = L if best is None else min(best, L)
    return best


with open(os.path.join(TESS, "12123_sweep", "tessellation_output.json")) as f:
    geo = json.load(f)
facet = {k: triangle_facet(p, r) for k, (p, r) in geo.items()}


def meta(key):
    txt = open(os.path.join(SW, f"refine_{key}_frac50_shear_meta.txt")).read()
    g = lambda pat: re.search(pat, txt).group(1)
    return dict(tau_M=float(g(r"tau_M=([\d.e+-]+)")),
                G_U=float(g(r"G_U=([\d.e+-]+)")),
                ne=int(g(r"ne=(\d+)")))


# members present (have a frac50 meta), ordered by decreasing L
keys = sorted([k for k in geo
               if os.path.exists(os.path.join(SW, f"refine_{k}_frac50_shear_meta.txt"))],
              key=lambda k: -facet[k])
rows = ""
for n, k in enumerate(keys, start=1):
    m = meta(k)
    r = float(k.split("_")[1])
    rows += (f"<tr><td>{n}</td><td>{r:.3e}</td><td>{facet[k]:.3e}</td>"
             f"<td>{m['tau_M']:.3e}</td><td>{m['G_U']:.3f}</td><td>{m['ne']:,}</td></tr>\n")

# ----------------------------------------------------------------- HTML
CSS = """
:root{--fg:#1a1a1a;--mut:#666;--acc:#b3203a;--acc2:#2e8b57;--bg:#fff;--box:#f6f7f9;}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 color:var(--fg);background:var(--bg);line-height:1.6;margin:0;}
.wrap{max-width:920px;margin:0 auto;padding:48px 24px 80px;}
h1{font-size:2.0rem;margin:.2em 0 .1em;line-height:1.2}
.sub{color:var(--mut);font-size:1.05rem;margin-bottom:2.2em}
h2{font-size:1.5rem;margin:2.4em 0 .5em;padding-bottom:.25em;border-bottom:2px solid var(--acc)}
h3{font-size:1.15rem;margin:1.8em 0 .4em;color:var(--acc)}
p{margin:.7em 0}
figure{margin:1.6em 0;padding:0}
figcaption{color:var(--mut);font-size:.9rem;text-align:center;margin-top:.5em}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:1.2em 0}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:right}
th{background:var(--box);font-weight:600}
td:first-child,th:first-child{text-align:center}
.key{background:var(--box);border-left:4px solid var(--acc);padding:14px 18px;margin:1.4em 0;border-radius:4px}
.key.green{border-left-color:var(--acc2)}
code{background:var(--box);padding:1px 5px;border-radius:3px;font-size:.9em}
.eqn{font-size:1.05rem;text-align:center;margin:1em 0;font-family:Georgia,serif}
.foot{color:var(--mut);font-size:.85rem;margin-top:3em;border-top:1px solid #eee;padding-top:1em}
.tag{display:inline-block;background:var(--acc);color:#fff;font-size:.72rem;
 padding:2px 8px;border-radius:10px;vertical-align:middle;margin-left:8px}
"""

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>(3,12,12) ratio-sweep study — DAGBS secondary relaxations</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Grain-boundary relaxations in a truncated-hexagonal polycrystal</h1>
<div class="sub">DAGBS frequency-domain homogenisation of the (3,12,12) Archimedean
tiling with a tunable small grain &middot; controlled facet-length scaling test</div>

<h2>1 &nbsp;Rationale</h2>
<p>In the random-microstructure (&sigma;-ensemble) DAGBS sweeps, a minority of
seeds show a <b>secondary loss peak at high frequency</b>, above the main
grain-boundary-sliding relaxation. Per-facet analysis traced it to the
diffusion-accommodated relaxation of the <b>shortest grain-boundary facets</b>
(slivers at near-degenerate triple junctions): the relaxation frequency is set
by the facet length <i>L</i> (the diffusion path), so a shorter facet relaxes at
higher &omega;. But in random microstructures the sliver length is uncontrolled
and often falls below the mesh size, which compressed the measured scaling
exponent (cross-seed slope only &minus;0.70 instead of the expected
&minus;2&hellip;&minus;3).</p>

<p>This study replaces the random microstructure with a <b>controlled Archimedean
family</b> in which a single small grain shrinks while everything else is fixed,
turning facet length into a clean, mesh-resolved experimental knob. We use the
<b>(3,12,12) truncated-hexagonal tiling</b> &mdash; dodecagons (large grains)
with triangles (small grains) at every junction &mdash; because its topology is
hex-like: there are no continuous slip planes, so the Maxwell time
<i>&tau;<sub>M</sub></i> is set by the large grains and stays nearly constant as
the triangles shrink. The triangle facet length can therefore be varied
independently of <i>&tau;<sub>M</sub></i>, isolating the secondary relaxation.
(The complementary (4,8,8) tiling, by contrast, has collinear slip highways and
<i>&tau;<sub>M</sub></i> itself collapses with the small-grain size &mdash; a
different regime.)</p>

<div class="key">Design knob: the small/large boundary-length ratio
<i>r</i> = (triangle perimeter)/(dodecagon perimeter), swept log-spaced from the
Archimedean value <b>1/4</b> down to <b>1/5000</b>. Average face area
(1/50) and the periodic domain are held fixed along the sweep.</div>

<h2>2 &nbsp;Geometry</h2>
<p>Nine periodic RVEs were generated, each a rectangular cut-and-paste of the
tiling so that no grain boundary lies on the domain edge. From member&nbsp;1
(Archimedean, <i>r</i>&nbsp;=&nbsp;1/4) to member&nbsp;9 the triangles shrink by
a factor of ~250 in facet length while the dodecagon network is unchanged.</p>
<figure>{img("sweep")}<figcaption>The nine (3,12,12) geometries; triangles
(small grains) shrink left&rarr;right, top&rarr;bottom.</figcaption></figure>

<p>Each RVE is meshed with second-order elements on the bulk grains and a
normal-traction multiplier field on every grain boundary. The tiny triangle
facets are resolved by <b>per-facet proportional refinement</b>
(<code>meshes_refine.py</code>, element size <i>L</i>/50 on facets shorter than
0.02), so the diffusion field on the smallest triangles is captured.</p>
<figure>{img("mesh_fe")}<figcaption>Representative finite-element mesh
(Archimedean member), coloured by grain.</figcaption></figure>
<figure>{img("mesh_bnd")}<figcaption>Grain-boundary labelling: interior
sliding/diffusion boundaries and the periodic outer pairs.</figcaption></figure>

<table>
<tr><th>member</th><th>ratio <i>r</i></th><th>triangle facet <i>L</i></th>
<th>&tau;<sub>M</sub></th><th><i>G<sub>U</sub></i></th><th>mesh elements</th></tr>
{rows}</table>
<p style="color:var(--mut);font-size:.9rem">As designed,
<i>&tau;<sub>M</sub></i> saturates toward the hexagonal value
(~1.46&times;10<sup>&minus;4</sup>) and <i>G<sub>U</sub></i>&rarr;0.82 once the
triangles are small &mdash; the large-grain network sets the Maxwell time, the
triangles do not.</p>

<h2>3 &nbsp;Numerical experiments</h2>

<h3>3a &nbsp;Production sweep: the secondary peak marches up</h3>
<p>Each geometry was swept over <code>ln(&omega;&tau;<sub>M</sub>) &isin;
[&minus;3, 18]</code> (161 points, CG solver, shear). The loss modulus
<i>C&Prime;</i> shows the main relaxation at <code>ln&nbsp;&asymp;&nbsp;0</code>
for all members (confirming the per-geometry <i>&tau;<sub>M</sub></i>
calibration), plus a secondary peak that moves to higher frequency as the
triangle shrinks.</p>
<figure>{img("individual")}<figcaption>Loss modulus <i>C&Prime;</i> per
geometry; the red marker tracks the detected secondary peak marching up in
frequency as the triangle shrinks.</figcaption></figure>
<p>Across members 2&ndash;6 the secondary peak moves up smoothly and traces a
clean power law (quantified in the conclusion). The smallest members showed an
<i>additional</i> feature near <code>ln&nbsp;&asymp;&nbsp;14</code> that did
<i>not</i> lie on that line &mdash; prompting the follow-up experiments
below.</p>

<h3>3b &nbsp;Wide band: two coexisting modes</h3>
<p>Member&nbsp;9 (smallest triangle) was re-swept over the wider
<code>ln(&omega;&tau;<sub>M</sub>) &isin; [&minus;3, 24]</code> and now shows
<b>two distinct peaks at once</b>: the bump at
<code>ln&nbsp;&asymp;&nbsp;14.4</code> <i>and</i> the genuine triangle peak
entering the window at <code>ln&nbsp;&asymp;&nbsp;23.0</code> (the &minus;3 law
predicts 23.1). Two coexisting peaks prove the bump is a separate mode, not the
triangle peak misplaced. Member&nbsp;7 (larger triangle) shows its triangle peak
at <code>ln&nbsp;&asymp;&nbsp;18.1</code>, also on the &minus;3 law.</p>
<figure>{img("wideB")}<figcaption>Wide-band <i>C&Prime;</i> for members 7 and 9
(grey = production-band overlap). Member 9 (right) shows both the bump
(ln&asymp;14.4) and the triangle Coble peak (ln&asymp;23.0); the red dashed line
is the &minus;3 law prediction.</figcaption></figure>

<h3>3c &nbsp;Mesh independence</h3>
<p>Member&nbsp;9 was re-run at twice the refinement (<i>L</i>/100). The
<i>L</i>/50 and <i>L</i>/100 curves are indistinguishable across the whole range
&mdash; both the bump and the triangle peak are <b>physical, not numerical
artifacts</b>.</p>
<figure>{img("wideC")}<figcaption>Mesh independence: <i>L</i>/50 vs <i>L</i>/100
overlay exactly, including both high-frequency peaks.</figcaption></figure>

<h3>3d &nbsp;Where each mode dissipates</h3>
<p>Per-facet attribution of the GB dissipation across all three of member 9's
relaxations, plus a control. The four panels (left&rarr;right):</p>
<ul style="margin:.3em 0 .8em">
<li><b>Main peak (ln&asymp;0)</b> &mdash; dissipation is delocalised on the
<b>long facets</b> (99.9%, participation 18): the large-grain sliding
relaxation.</li>
<li><b>Bump (ln&asymp;14.4)</b> &mdash; <b>81% on the triangle facets</b>, spread
over all 16 triangles (participation 59, collective).</li>
<li><b>Triangle Coble peak (ln&asymp;23.0)</b> &mdash; also triangle-loaded
(74%, participation 52).</li>
<li><b>Control: member 5 at its genuine triangle peak (ln&asymp;13.8)</b> &mdash;
a different geometry whose known <i>L</i><sup>&minus;3</sup> peak sits at almost
the same frequency as member 9's bump; it loads the triangle edges (90%),
confirming the method correctly localises a genuine triangle relaxation.</li>
</ul>
<p>The main relaxation clearly lives on the long-facet network, while
<i>both</i> high-frequency modes dissipate on the triangles &mdash; as expected,
since both are triangle-diffusion phenomena. They are nonetheless two distinct
relaxations, separated by their frequency scaling
(<i>L</i><sup>&minus;1</sup> vs <i>L</i><sup>&minus;3</sup>) and by coexisting as
two separate peaks (section&nbsp;3b): the bump is the long-facet network charging
<i>through</i> the triangles as diffusive resistors, the Coble peak is the
triangles' own internal equilibration.</p>
<figure>{img("attrib")}<figcaption>Dissipation share vs facet length for member
9's three relaxations (main, bump, triangle Coble) and a same-frequency control
(member 5's genuine triangle peak). Colour = facet class (triangle edge /
long facet adjacent to a triangle / other).</figcaption></figure>

<h2>4 &nbsp;Conclusion</h2>
<p>The (3,12,12) family exhibits <b>two distinct grain-boundary relaxations</b>
above the main sliding peak, both mesh-converged and well separated in
frequency once the triangle is small:</p>

<div class="key"><b>Triangle Coble mode.</b>
<div class="eqn">ln(&omega;<sub>peak</sub>&tau;<sub>M</sub>) =
&minus;3.09&thinsp;ln&thinsp;<i>L</i> &minus; 4.45</div>
i.e. &omega;<sub>peak</sub> &prop; <i>L</i><sup>&minus;3</sup> &mdash; the
Coble/grain-boundary-diffusion law &mdash; over 7 points spanning 2.4 decades of
facet length. This is the controlled confirmation of the random-ensemble
secondary peak, free of the mesh-floor compression that flattened the random
slope to &minus;0.70. It is the triangle's <i>own</i> diffusive equilibration.</div>

<div class="key green"><b>Network &ldquo;RC&rdquo; mode.</b>
<div class="eqn">&omega;<sub>peak</sub> &prop; <i>L</i><sup>&minus;1</sup>
&nbsp;(fitted slope &minus;0.99)</div>
A collective relaxation of the long-facet diffusion network charging
<i>through</i> the tiny triangles, which act as diffusive bottleneck resistors
(resistance <i>R</i>&nbsp;&prop;&nbsp;<i>L</i> at fixed network capacitance
<i>C</i>, so &omega;&nbsp;&prop;&nbsp;1/<i>RC</i>&nbsp;&prop;&nbsp;<i>L</i><sup>&minus;1</sup>).
Its dissipation loads the triangle edges (section 3d). It only becomes visible once
the steeper <i>L</i><sup>&minus;3</sup> mode has marched out of the original
band.</div>

<figure>{img("final")}<figcaption>The two relaxation laws: triangle Coble mode
(slope &minus;3.09, red) and network RC mode (slope &minus;0.98, green), with the
&minus;3 and &minus;1 reference guides.</figcaption></figure>

<p>The bump initially looked like a numerical &ldquo;pinned&rdquo; artifact; the
three tests (coexistence in the wide band, mesh independence, and dissipation
attribution) show it is a real, collective, physically-interpretable second
relaxation. The two modes cross near
<code>ln&thinsp;<i>L</i>&nbsp;&asymp;&nbsp;&minus;5</code>; for smaller triangles
the network RC mode is the lower-frequency of the two.</p>

<p style="color:var(--mut);font-size:.9rem"><b>Future work.</b> The (4,8,8)
slip-highway family (&eta;<sub>ss</sub>&nbsp;&prop;&nbsp;<i>L</i><sup>3</sup>,
established by calibration) and the (4,6,12) family remain to be swept.</p>

<div class="foot">Generated by <code>build_report_12123.py</code> from the CSD3
sweep outputs in <code>tess/csd3_results/</code>. All figures embedded; this file
is self-contained.</div>

</div></body></html>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(keys)} members)")
