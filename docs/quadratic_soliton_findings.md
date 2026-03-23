# Quadratic Solitons — Dispersion Landscape & Questions

## Dispersion Landscape

**β₂ at 1550 nm is anomalous: −50.1 fs²/mm.** This is the clean case — soliton
formation with Δk > 0 works directly. No geometry engineering needed.

Two ZDWs at **1335 nm** and **2023 nm**, giving an anomalous window from
1335–2023 nm. Both candidate wavelengths (1550 and 2000 nm) are inside it, but
1550 nm has much stronger anomalous GVD (−50 vs −9 fs²/mm), making it the
better choice.

**Phase matching (1550 nm SHG, pp = 5.0 µm):** Δk crosses zero near T ≈ 35°C.
Temperature above that gives Δk > 0 (correct sign for soliton with anomalous
GVD). The regimes:

| Temperature | Δk·L (20 mm) | Regime |
|---|---|---|
| 25°C | −6 | Quadratic (wrong sign) |
| 50°C | +27 | Transitional |
| 100°C | +98 | Near cascaded limit |
| 150°C | +175 | Deep cascaded |

---

## Questions

### 1. GVM is enormous at 1550/775 nm: 153 ps/mm

A 100 fs pulse walks off in 0.65 µm — the SH completely separates from the FH
within a single coherence length. This severely complicates both regimes:

- **Cascaded:** The round-trip FH→SH→FH doesn't happen locally — the generated
  SH walks away before converting back. The effective Kerr picture assumes the
  SH is "slaved" to the FH envelope, but with this GVM it's not.
- **Quadratic soliton:** The nonlinear coupling needs to overcome a 153 ps/mm
  velocity difference to lock FH and SH together. This requires extreme
  coupling strength.

At 2000/1000 nm, GVM is 10× smaller (15.3 ps/mm) but still large.

**Is the 1550 nm case even viable for solitons, or should we go to 2000 nm
despite its weaker β₂?** Or use much longer pulses (picosecond) where walk-off
is less severe relative to pulse duration?

### 2. z-resolved output is essential but not available

SNOW's NEE returns only the final field. The brief's (z,t) color maps require
intermediate z-slices. Three options:

- **(a)** Run separate simulations at each z (propagate from 0 to z_k for each
  k). Simple but redundant — re-propagates the same initial section each time.
  For N=4096, ~5 s/point × 100 z-slices = 8 min. Tolerable.
- **(b)** Propagate in segments using the `z0` parameter — but the NEE's
  dispersion operator uses absolute z, so chaining segments requires injecting
  `exp(iD·z_start)` into the input spectrum. Doable but hacky.
- **(c)** Add an optional `z_save` array to `nlo_scipy.py` that records
  `ifft(Y·exp(−iDz))` at specified z positions during integration. Clean,
  small code change.

Do you want me to implement (c), or is (a) acceptable?

### 3. Grid sizing for pulsed operation

Unlike the CW notebook, this needs a pulse-appropriate grid: N = 2^12–2^13
spanning ~700–2000 nm to cover both FH (1550 nm) and SH (775 nm) with their
pulse bandwidths. This is the same regime as Tutorials 5 and 7. The
energy-conserving bin-alignment trick from the CW notebook is not needed here
since the pulse has broadband spectral content.

### 4. Poling approach: Kg vs explicit poling function?

For the CW OPA we used analytical Kg to avoid resolving the 5 µm poling
period. Here we're deliberately detuning from phase matching. Either approach
works:

- **Kg:** Set Kg = 2π/pp, which gives Δk ≠ 0 naturally from the waveguide
  dispersion at the detuned temperature. Cleaner numerically.
- **Explicit poling:** Needed for Cell 5c (poling disorder). Could use explicit
  poling throughout for consistency, but would need tighter tolerances or the
  v_ref trick to resolve the 5 µm period.

Recommend Kg for Cells 0–4, switching to explicit poling only for Cell 5c.
Agree?
