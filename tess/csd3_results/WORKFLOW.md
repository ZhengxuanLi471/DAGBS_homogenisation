# tess/csd3_results — (3,12,12) sweep analysis

Analysis of the `12123_sweep` frequency-sweep CSVs pulled back from CSD3.
Extracts the two secondary-relaxation scaling laws and attributes the dissipation
per facet.

## Files (run order)
- `merge_member8_parts.py` — stitch the partial CSVs of the wide-band member-8 run.
- `analyze_12123.py` — peak detection + per-member spectra → `individual_12123.png`,
  `scaling_12123.png`.
- `analyze_wide_12123.py` — wide-band members 7–9 (`ln ∈ [-3,24]`), separating the
  two modes once the L⁻³ peak leaves the standard band → `wideB/ wideC_12123.png`.
- `attribute_12123.py` — per-facet dissipation attribution / participation ratio
  → `attribution_12123.png`.
- `scaling_final_12123.py` — final two-law fit → **`scaling_final_12123.png`**
  (triangle Coble `L⁻³` + network "RC" `L⁻¹`).
- `build_report_12123.py` — assembles `report_12123.html`.

## Inputs
Per-geometry sweep CSVs in `12123_sweep/` (under this dir when pulled), each with
a `_meta.txt` recording τ_M. Always check `_meta.txt` τ_M vs production before
trusting a tail's x-axis (apply a `ln(τ_ref/τ_used)` shift if calibration
stagnated — τ_M is a pure frequency rescaling).

## Result
Two mesh-converged laws (den=50 ≡ den=100): triangle mode
`ln(ω_peak·τ_M) = −3.09 ln L − 4.45` (∝ L⁻³); network mode slope −0.98 ≈ −1
(81% dissipation on triangle facets, participation ratio 59).
