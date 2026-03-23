# CW Waveguide OPA — Research Findings

## Q1: CW representation — minimal grid

A minimal grid (N=4 or 8) **won't work**. The NEE's nonlinear product
(`a²·exp(iφ) + 2|a|²·a`) is computed via FFT ↔ time-domain multiplication
with 4x upsampling for anti-aliasing. With only 4 bins, the downsampling
truncation corrupts cross-coupling between the three waves.

**Practical minimum: N=64–256.** With BW spanning 775–1560 nm (~195 THz),
N=256 gives df ≈ 780 GHz — plenty to resolve three CW lines. The time window
is only ~1.3 ps, and with so few points the solver will be very fast. This is
effectively the "quasi-CW via tiny grid" approach — not a true ns flat-top
pulse, but a small grid where each frequency bin acts as a CW tone. Edge
effects are irrelevant because there are no pulse edges to worry about at
these timescales.

---

## Q4: Phase preservation

Phase is faithfully preserved through propagation. The NEE's dispersion
operator, carrier modulation, and nonlinear product all maintain the relative
phase φ\_pump − φ\_signal − φ\_idler. For phase-sensitive OPA, construct the
input as a sum of three CW tones with explicit complex amplitudes:

```python
a(t) = √P_pump·exp(i·2π·f_pump·t)
     + √P_signal·exp(i·(2π·f_signal·t + φ_s))
     + √P_idler·exp(i·(2π·f_idler·t + φ_i))
```

and sweep φ\_s + φ\_i relative to the pump. The only concern is ensuring the
three frequencies land on (or very near) grid bin centers to avoid spectral
leakage.

---

## Q6: QPM poling period

### SNOW's effective index method gives:

| Geometry | Λ (room temp) |
|---|---|
| 700 nm film, 350 nm etch, 1.8 µm width | **5.00 µm** |
| 600 nm film, 300 nm etch, 1.8 µm width | **4.39 µm** |

Temperature dependence is weak (~0.04 µm shift over 125°C).

### Effective indices (700 nm film, T=24.5°C):

- n\_eff(775 nm) = 2.11487
- n\_eff(1540 nm) = 1.96184
- n\_eff(1560 nm) = 1.95794

### Cross-check against literature

Published TFLN PPLN poling periods for 775 nm → ~1550 nm cluster at
**3.5–4.8 µm**:

| Reference | Λ | Film | Width | Etch |
|---|---|---|---|---|
| Mondal et al. 2024, arXiv:2411.17369 | 3.53 µm | 800 nm | 0.8 µm | 200 nm |
| Stokowski et al. 2023, Nat. Comm. 14 3355 | 3.7 µm | 500 nm | 1.2 µm | 300 nm |
| Wang et al. 2018, Optica 5 1438 | ~4.0 µm | 600 nm | ~1.4 µm | 300 nm |
| Renaud et al. 2024, Micromachines 15 1145 | 4.20 µm | 600 nm | 1.5–1.7 µm | 300 nm |
| Kashiwazaki et al. 2025, PMC12714032 | 4.76 µm | 600 nm | 2.0 µm | 300 nm |

**SNOW's 5.0 µm (700 nm film) is on the high end; its 4.4 µm (600 nm film)
is well within the literature range.** The trend is clear — thicker films and
wider ridges push Λ longer because mode confinement is weaker and effective
index dispersion is closer to bulk. SNOW's effective index method is
approximate (no 2D mode solving), so a ~10–20% offset from published values
for the same nominal geometry is expected.

**Recommendation:** Use the 700 nm film geometry (matching X0 = 1.1e-12
calibration) with Λ ≈ 5.0 µm from SNOW's model, and note the discrepancy vs
literature in the notebook.

---

## Q2: X0 calibration

The existing X0 = 1.1e-12 was calibrated for the 700 nm / 350 nm / 1800 nm
geometry at ~1–2 µm wavelengths (SHG/OPO in tutorials 5 and 7). For the CW
OPA at 775/1550 nm in the same geometry, this should be the right starting
point.

**Calibration procedure:** Run a low-pump-power simulation and compare signal
gain against cosh²(gL), where:

```
g = √(ω_s · ω_i · d_eff² · I_pump / (n_s · n_i · ε₀ · c³))
```

For a waveguide, I\_pump = P\_pump / A\_eff, where A\_eff is the effective mode
area. The catch is that A\_eff is not directly available from SNOW's effective
index method — it gives n\_eff but not the mode field profile. This is why X0
bundles d\_eff, mode overlap, and area together empirically.

**Key data points from literature for sanity-checking:**

| Reference | Gain | Pump | Length | Film | Notes |
|---|---|---|---|---|---|
| Mishra et al. 2025, arXiv:2602.05982 | 23.5 dB (PS) | 110 mW | 14 mm | 600 nm | Adapted poling, 2.2 µm wide |
| Chen et al. 2025, Optica 12 1242 | 13.9 dB (CW) | — | ~20 mm | x-cut TFLN | Domain-engineered, 110 nm 10-dB BW |
| Ledezma et al. 2025, arXiv:2509.26425 | >17 dB (PS) | <200 mW | ~18 mm | 700→500 nm | SH-resonant architecture |
| Xin et al. 2025, Sci. Rep. | 71.5 dB (sim) | 0.5–3 W | 20 mm | — | Simulation only, 0.3 dB/cm loss |

If our simulation gives wildly different gain at similar pump powers and
lengths, X0 needs adjustment.

---

## Next step

Proceed to building the notebook: waveguide setup, X0 calibration cell, then
the brief's Cell 1–7 progression.
