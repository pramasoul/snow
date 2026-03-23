# Quadratic Soliton Brief — Revised After Dispersion/GVM Analysis

## What changed and why

The original brief assumed the cascaded-Kerr regime (large Δk, SH
perturbative) would be the primary demonstration, with the quadratic regime
as an advanced topic. Simulation and corrected arithmetic show the opposite:

1. **Soliton period z₀ = T₀²/|β₂| is much longer than estimated.** My
   original calculation had a factor-of-1000 units error. Corrected values
   show z₀ > 20 mm only for |β₂| ≳ 50 fs²/mm and T₀ ≲ 50 fs, which means
   only the 1550 nm operating point (β₂ = −50 fs²/mm) has a short enough
   soliton period — but GVM kills it there.

2. **GVM at 1550/775 nm (153 ps/mm) prevents cascaded Kerr action.** The SH
   walks through the FH in sub-micron distances. Simulations confirm: all
   energies show pure dispersive broadening, no self-trapping.

3. **At 2000 nm, the accessible regime is genuinely quadratic.** Low
   temperatures (T ≈ 25°C, Δk·L ≈ 71) give 40–60% energy transfer to SH,
   dramatic pulse compression, and breathing dynamics in 40 mm — but the SH
   is a full dynamical partner, not a perturbation.

**The cascaded-Kerr soliton is not accessible in practical TFLN waveguide
lengths at these wavelengths.** It requires meters of waveguide or hundreds
of pJ, neither of which is realistic. The clean cascaded-Kerr regime becomes
an analytical discussion and limiting case rather than a simulation target.

---

## Revised notebook structure

### Cell 0: Dispersion and Regime Landscape

This cell now has a much more important job: it explains WHY we operate in
the quadratic regime rather than the cascaded-Kerr regime.

**Plot A:** β₂ vs. wavelength for the TFLN waveguide. Mark 1550 nm and
2000 nm. Show the anomalous window (1335–2023 nm).

**Plot B:** GVM between FH and SH vs. FH wavelength. Show the enormous
value at 1550 nm (153 ps/mm) and the more manageable value at 2000 nm
(~15 ps/mm).

**Plot C:** Soliton period z₀ vs. pulse duration for both wavelengths.
Draw a horizontal line at L = 40 mm (crystal length). Only the region
below this line is accessible. Show that 1550 nm / 50 fs barely fits,
2000 nm requires ≲ 25 fs (impractical with current grid resolution and
GVM).

**Plot D:** Walk-off length L_w = T₀/GVM vs. pulse duration for both
wavelengths, with coherence length L_coh at several Δk values overlaid.
The cascaded-Kerr regime requires L_w ≳ L_coh. Show that this condition
is never satisfied at 1550 nm for sub-ps pulses, and only marginally
satisfied at 2000 nm for ps pulses (but then z₀ is kilometers).

**Text:** Explain the three-way tension — short pulses for visible soliton
dynamics (z₀ < L), long pulses for intact cascading (L_w > L_coh), and
moderate Δk for reasonable soliton power. In TFLN waveguides, these
constraints leave the genuinely quadratic regime (moderate Δk, SH
dynamically active) as the accessible operating point. The cascaded-Kerr
limit would require meters of waveguide.

### Cell 1: Two-Color Quadratic Soliton Dynamics at 2000 nm

**Operating point:** T ≈ 25°C, Λ = 5.18 µm (Tutorial 7 period), Δk ≈ 1785/m,
Δk·L ≈ 71 (for 40 mm). FH at 2000 nm, SH generated at 1000 nm.

**Pulse:** 100 fs sech², energy scan from 0.5 to 15 pJ.

**Plot A (the money plot):** z–t color maps for FH and SH fields, side by
side, at several energies. Use the z_save infrastructure from option (c).
At low energy: dispersive broadening, SH is weak and walks away. At
moderate energy (~3–5 pJ): the FH begins to self-trap, SH becomes a
visible co-propagating companion. At high energy (~7–10 pJ): dramatic
compression, breathing, possible fission.

**Plot B:** Output pulse FWHM vs. input energy. Should show the transition
from broadening to compression.

**Plot C:** FH and SH energy fractions vs. z at several input energies.
Shows the energy exchange dynamics — in the quadratic soliton, energy
oscillates between FH and SH as they propagate.

**Plot D:** Output spectra at several energies. Show spectral broadening
and any dispersive wave features at high energy.

### Cell 2: The Bound State — Velocity Locking

This is the signature property of the quadratic soliton: the nonlinear
coupling forces the FH and SH to co-propagate at the SAME group velocity,
despite having different linear group velocities.

**Experiment:** At the soliton energy from Cell 1, track the temporal center
of mass of both FH and SH pulses as a function of z.

**Plot:** FH pulse center vs. z and SH pulse center vs. z on the same axes.
At low energy (no soliton), the curves diverge at a rate set by GVM. At
soliton energy, they converge and travel together — the nonlinear coupling
has "captured" the SH. The capture distance and the residual velocity
difference (if any) are measures of how tightly bound the soliton is.

**Reference lines:** Draw the linear (uncoupled) trajectories at the
respective group velocities. The departure from these lines IS the
soliton binding.

### Cell 3: Energy-Δk Phase Diagram

Sweep temperature (equivalently Δk) and input energy on a 2D grid.

**Parameters:**
- Temperature: 20–175°C (Δk from ~1500 to ~10000 /m)
- Energy: 0.1–50 pJ (log scale)
- For each (T, E) pair, propagate 40 mm and record output FH FWHM, peak
  power, and SH energy fraction.

**Plot A:** Color map of output FWHM / input FWHM in the (Δk, energy) plane.
Blue = compression, white = unchanged, red = broadening. The soliton
existence region is the boundary between compression and broadening.

**Plot B:** Color map of SH energy fraction. Shows where the interaction
transitions from perturbative (SH < 5%, cascaded regime) to genuinely
quadratic (SH > 20%). The cascaded regime will be at high Δk and will show
no soliton dynamics (too weak) — confirming the inaccessibility argument
from Cell 0.

**This map is the central result of the notebook.** It shows where quadratic
solitons live, how they differ from cascaded-Kerr solitons, and why the
operating point was chosen where it was.

### Cell 4: Soliton Self-Compression and Spectral Broadening

At the operating point from Cell 1, push to high energy (3–5× the
fundamental soliton energy) and use a longer crystal if possible (50 mm).

**Expected dynamics:** Higher-order soliton compression followed by
temporal splitting (fission) and dispersive wave emission. This is the
χ⁽²⁾ analog of supercontinuum-generating soliton fission in fiber.

**Plot A:** z–t evolution showing compression and fission.

**Plot B:** z–λ evolution showing spectral broadening and discrete
dispersive wave peaks. The dispersive wave wavelengths are set by the
phase-matching condition between the soliton and linear waves at the
same group velocity — these can be computed analytically from the
dispersion curve and compared to the simulation.

**Connection to earlier discussion:** If visible spectral broadening occurs,
this demonstrates cascaded-χ⁽²⁾ supercontinuum generation — the phenomenon
I previously said SNOW couldn't model. SNOW can't model χ⁽³⁾ supercontinuum,
but it CAN model χ⁽²⁾-mediated spectral broadening through soliton dynamics.
Note this explicitly in the notebook.

### Cell 5: Robustness (as in original brief)

**5a: Propagation loss.** Add 0.1–1 dB/cm loss, show soliton adiabatic
decay (broadening, amplitude drop). In the quadratic regime, loss affects
FH and SH differently (SH typically has higher loss in LiNbO₃), which
breaks the bound state asymmetrically.

**5b: Non-sech input.** Launch a Gaussian at soliton energy. Show reshaping
and radiation shedding. The quadratic soliton profile is NOT exactly sech
(it differs from the Kerr soliton shape), so the reshaping dynamics are
genuinely different.

**5c: Poling disorder.** Switch to explicit poling function with random
domain length fluctuations (±5%, ±10%). At moderate Δk the soliton is in
the quadratic regime where the interaction is strong — it should be fairly
robust. Verify.

### Cell 6: Cascaded-Kerr Limit as Analytical Discussion

Rather than simulating the cascaded regime (which would require impractical
parameters), present it analytically:

- Derive n₂_eff = ±d_eff² / (n² · ε₀ · c · Δk) from the large-Δk limit
  of the coupled equations
- Compute the equivalent Kerr soliton energy as a function of Δk
- Plot this on the phase diagram from Cell 3, showing that the cascaded-Kerr
  soliton line falls in the inaccessible (high Δk, high energy) corner
- Overlay the quadratic soliton existence boundary from Cell 3 to show where
  the two descriptions diverge

**This is the intellectual closure:** the cascaded-Kerr soliton is the
asymptotic limit of the quadratic soliton as Δk → ∞, but for finite Δk in
real devices, the quadratic description is essential.

### Cell 7: Comparison Summary

| Property               | Quadratic soliton (this notebook) | Cascaded-Kerr soliton (limit) | Kerr soliton (fiber) |
|------------------------|-----------------------------------|-------------------------------|----------------------|
| Nonlinear mechanism    | FH ↔ SH parametric coupling       | Same, perturbative limit      | χ⁽³⁾ SPM            |
| Fields involved        | FH + SH, both active              | FH dominant, SH slaved        | Single field         |
| SH energy fraction     | 10–60%                            | < 1%                          | N/A                  |
| Soliton shape          | Non-sech (wider tails)            | sech (Kerr-like)              | sech                 |
| Velocity locking       | Yes — FH and SH co-propagate      | Not relevant (SH negligible)  | N/A                  |
| Accessible in TFLN?    | **Yes (this notebook)**           | No (needs meters of WG)       | No (needs χ⁽³⁾)     |
| Key parameter          | Δk (via temperature)              | Δk (must be large)            | γ (material)         |

---

## Notes for CC

1. **z_save implementation (option c) is critical.** Almost every cell
   requires z-resolved evolution maps. Do this first before any soliton
   simulation.

2. **Grid: N = 2¹²–2¹³, spanning ~900–2200 nm.** Needs to cover both
   2000 nm FH and 1000 nm SH with their pulse bandwidths, plus room for
   dispersive wave emission in Cell 4.

3. **Poling: analytical Kg throughout, explicit only for Cell 5c.** Same
   recommendation as before — unchanged by the regime shift.

4. **X0 = 1.1e-12 should apply** since we're using the same 700 nm film
   waveguide geometry at similar wavelengths (2000/1000 nm) to Tutorial 7.
   Validate with a low-energy SHG check: at Δk = 0 (T ≈ 35°C for 1550 nm,
   whatever temperature gives phase matching at 2000 nm), a weak pulse
   should show the familiar SHG conversion efficiency from Tutorial 5.

5. **SH separation in post-processing.** For the two-color plots (Cells 1–3),
   bandpass-filter the full-grid field into FH (>1200 nm) and SH (<1200 nm)
   spectral windows. The separation is clean — no spectral overlap between
   FH and SH at these wavelengths unless Cell 4 produces truly extreme
   broadening.

6. **Computational cost for Cell 3.** Phase diagram requires ~20 × 20 = 400
   simulations, each 40 mm propagation with N = 2¹². With z_save recording
   ~50 slices, this is substantial but tractable on GPU with JAX. Estimate
   ~10–30 minutes. Start with a coarse 10 × 10 grid to check the landscape,
   then refine.

7. **The irony from the technical note.** In the FDTD SHG work, you fought
   to eliminate parasitic Δk from numerical dispersion. Here we're using a
   physical Δk as the mechanism for pulse self-trapping. Same interaction,
   opposite goals. Note this in the notebook — it's a nice callback and
   illustrates that "phase mismatch" is a resource, not just a defect.

8. **Publication potential.** A systematic numerical study of quadratic
   temporal solitons in TFLN waveguides using a validated simulation tool,
   with a z-resolved phase diagram mapping the transition from quadratic to
   cascaded-Kerr regimes, would be a solid paper for Optics Letters or
   Physical Review A. The TFLN community is actively interested in soliton
   dynamics and the simulation infrastructure you've built is nontrivial.
   Keep the notebook clean enough to support this.
