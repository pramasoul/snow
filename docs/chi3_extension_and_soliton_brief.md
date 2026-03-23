# Adding χ⁽³⁾ Kerr to SNOW — Implementation Brief + Revised Soliton Notebook

## Part 1: The χ⁽³⁾ Extension

### Why

Pure χ⁽²⁾ solitons are inaccessible in TFLN waveguides at any wavelength
with the existing geometry: 1550 nm has sufficient |β₂| but GVM kills the
cascaded coupling, and 2000 nm has manageable GVM but |β₂| is ~10× too weak.
The fundamental problem is that temporal soliton formation in χ⁽²⁾ alone
requires either very long propagation (meters) or very large |β₂|, neither
of which is available in cm-scale TFLN.

Lithium niobate has a substantial χ⁽³⁾ (n₂ ≈ 1.8 × 10⁻¹⁹ m²/W), and in
tightly confining TFLN waveguides the effective Kerr coefficient γ is
~100–400× larger than in silica fiber. This is enough for soliton formation
in centimeter-scale devices. Adding χ⁽³⁾ to the NEE unlocks not just
solitons but also supercontinuum generation, combined χ⁽²⁾+χ⁽³⁾ dynamics,
and makes SNOW competitive with state-of-the-art TFLN simulation tools.

### What to add

A single term in the nonlinear product: the instantaneous Kerr self-phase
modulation (SPM).

In the time domain, the Kerr polarization is:

    P_NL^(3) = ε₀ · χ⁽³⁾ · |E|² · E

For an envelope A in the NEE, this becomes a term proportional to:

    i · γ · |A|² · A

where γ is the effective Kerr nonlinearity parameter. The exact form depends
on SNOW's field normalization — see implementation notes below.

### What NOT to add (yet)

- **Raman scattering:** The delayed χ⁽³⁾ response (intrapulse Raman) causes
  soliton self-frequency shift and contributes to supercontinuum dynamics.
  It's important for quantitative accuracy but is a separable addition. The
  soliton notebook doesn't need it — Raman mainly shifts soliton center
  frequency rather than affecting formation. Can be added as a follow-up.

- **Self-steepening / shock term:** The frequency-dependent correction
  i·(γ/ω₀)·∂/∂t(|A|²·A). Important for few-cycle pulses and octave-
  spanning supercontinuum. Not needed for the ≥50 fs pulses in this
  notebook. Can be added later.

- **Cross-phase modulation between FH and SH:** In the full χ⁽³⁾ treatment,
  the FH and SH each experience SPM from their own intensity AND XPM from
  each other's intensity. For the soliton notebook, the dominant χ⁽³⁾ effect
  is FH SPM (the SH is weak in the cascaded regime). XPM can be added later
  for completeness.

Starting with just the instantaneous SPM term keeps the implementation
minimal and the physics interpretable.

### Implementation

#### Where in the code

The NEE's nonlinear product is computed (presumably) in a function that
takes the time-domain envelope A(t) and returns the nonlinear contribution
to dA/dz. Currently this contains the χ⁽²⁾ terms. The Kerr term is
additive:

    NL_total(A) = NL_chi2(A) + NL_chi3(A)

where:

    NL_chi3(A) = i · γ_eff · |A(t)|² · A(t)

This is local in time (no convolutions, no FFTs beyond what's already done
for the χ⁽²⁾ term), so it's computationally cheap.

#### The coupling parameter γ_eff

The effective Kerr parameter for a waveguide is:

    γ = n₂ · ω₀ / (c · A_eff)

For LiNbO₃:
- n₂ ≈ 1.8 × 10⁻¹⁹ m²/W (extraordinary polarization, 1550 nm)
- At 2000 nm, n₂ is similar (weak wavelength dependence)

For the 700 nm film / 350 nm etch / 1800 nm width TFLN waveguide:
- A_eff at 1550 nm ≈ 2–4 µm² (estimated; SNOW doesn't compute mode profiles)

This gives γ ≈ 180–360 /W/km at 1550 nm.

**The same normalization challenge as X0 applies here.** SNOW's field
envelope A has some normalization (|A|² = power? intensity? something else?)
that determines how γ_eff maps to an actual code parameter. The approach:

1. Implement the Kerr term with a single free parameter `gamma_eff`
2. Calibrate by propagating a CW signal (no χ⁽²⁾, Δk → ∞ or χ⁽²⁾ off)
   through the waveguide and measuring the SPM-induced spectral broadening
3. Compare to the analytical result: Δφ_NL = γ · P · L, where P is the
   CW power and L is the propagation length
4. Adjust gamma_eff until the simulation matches

This is the same empirical calibration strategy that worked for X0. The
analytical value provides the starting point; the simulation validates it.

#### Enabling/disabling

Add a boolean flag or a parameter that defaults to gamma_eff = 0 (no Kerr).
This ensures all existing notebooks are unaffected. The soliton notebook
turns it on explicitly.

#### JAX compatibility

The |A|² · A operation is trivially JAX-compatible — it's element-wise
multiplication in the time domain, fully differentiable, and GPU-native.
No special handling needed.

---

## Part 2: Revised Soliton Notebook

### Operating point: 1550 nm with χ⁽³⁾ Kerr

With χ⁽³⁾ available, 1550 nm becomes the clear choice:

| Parameter | Value |
|-----------|-------|
| λ_FH | 1550 nm |
| β₂ | −50 fs²/mm (anomalous, confirmed by SNOW) |
| γ | ~180–360 /W/km (to be calibrated) |
| n₂ | 1.8 × 10⁻¹⁹ m²/W |
| Pulse duration | 50–100 fs sech² |

Soliton parameters (using γ = 300 /W/km as estimate):

| T₀ (sech) | FWHM | z₀ | z_π/2 (soliton period) | P_sol | E_sol |
|------------|------|----|------------------------|-------|-------|
| 28 fs | 50 fs | 16 mm | 25 mm | 590 W | 17 pJ |
| 57 fs | 100 fs | 64 mm | 101 mm | 150 W | 8.5 pJ |

**50 fs pulses give 1.6 soliton periods in 40 mm — ideal.** The pulse
completes enough of a soliton period to show clear self-trapping, and
higher-order solitons (N=2,3) will show one compression cycle.

**100 fs pulses give 0.4 soliton periods in 40 mm — marginal.** Can show
the onset of self-trapping but not a full period. Useful for contrast.

### Three-act structure

The notebook now tells a much richer story:

**Act I (Cells 1–3):** Pure Kerr solitons in TFLN. Familiar NLSE physics,
validates the χ⁽³⁾ implementation.

**Act II (Cells 4–5):** Turn on χ⁽²⁾ coupling (move temperature toward
SHG phase matching). The cascaded contribution modifies the effective
nonlinearity — it can add to or subtract from the Kerr n₂ depending on
the sign of Δk. Show soliton energy shifting with temperature.

**Act III (Cells 6–7):** Push into the regime where cascaded χ⁽²⁾ and Kerr
χ⁽³⁾ are comparably strong. Novel dynamics that neither nonlinearity
produces alone.

---

### Cell 0: χ⁽³⁾ Calibration and Validation

Before any soliton physics, validate the Kerr implementation.

**Experiment A: SPM spectral broadening.** Propagate a transform-limited
pulse (100 fs, moderate energy) through the waveguide with χ⁽²⁾ OFF
(set X0=0 or Δk→∞). Measure the output spectral width. Compare to the
analytical SPM result:

    Δω_max = 0.86 · γ · P_peak · L / T₀

(for a Gaussian pulse; slightly different prefactor for sech²). Adjust
gamma_eff until simulation matches. Record the calibrated value.

**Experiment B: B-integral check.** The accumulated nonlinear phase at pulse
center is B = γ · P_peak · L. For soliton-range parameters (P ~ 200 W,
L = 40 mm, γ = 300 /W/km), B ≈ 2.4 rad. Verify the simulation gives the
correct peak phase shift by comparing input and output spectral phase at the
center frequency.

**Experiment C: Soliton existence at N=1.** Set pulse energy exactly to
the fundamental soliton energy E_sol = 2|β₂|/(γ·T₀), propagate 40 mm
with χ⁽²⁾ off. The pulse should emerge with the same shape and duration
as the input (within a few percent). This is the definitive validation:
if the soliton doesn't maintain its shape at the analytically predicted
energy, gamma_eff is wrong.

### Cell 1: Pure Kerr Soliton — Energy Scan

χ⁽²⁾ OFF. 50 fs sech² pulse. Scan energy from 0.2× to 5× E_sol.

**Plot A:** z–t color map at N = 0.5 (dispersive), N = 1 (soliton), N = 2
(breathing/compression), N = 3 (complex higher-order dynamics). Four panels.

**Plot B:** Output FWHM vs. input energy (normalized to E_sol). Should show:
- N < 1: broadening (dispersion wins)
- N = 1: unchanged (soliton)
- N > 1: output narrower than input (higher-order soliton has compressed
  at least once); for integer N, periodic breathing

**Plot C:** Output spectrum vs. input energy. At N = 1, spectrum is unchanged
(sech in frequency). At N > 1, spectrum shows characteristic modulation
from SPM.

**This cell should look textbook.** The NLSE soliton is one of the most
studied objects in nonlinear optics — if these plots don't match the
standard results (Agrawal Chapter 5), something is wrong.

### Cell 2: Soliton Self-Compression and Fission

χ⁽²⁾ OFF. Launch at N = 3–5. Use 50 mm if possible.

**Plot A:** z–t color map showing initial compression, fission into
fundamental solitons, and dispersive wave emission.

**Plot B:** z–λ color map showing spectral broadening and discrete
dispersive wave peaks. The dispersive wave wavelength is set by:

    β(ω_DW) = β(ω_s) + (ω_DW − ω_s)/v_g,s + γ·P_s/2

where ω_s, v_g,s, P_s are the soliton center frequency, group velocity,
and peak power. Compute this analytically from SNOW's dispersion curve
and overlay as a vertical line on the spectral plot.

**This is the first step toward supercontinuum.** If the broadening spans
more than an octave, note it.

### Cell 3: Soliton Robustness

χ⁽²⁾ OFF.

**3a: Gaussian input.** Launch a Gaussian (not sech) at N = 1 soliton energy.
Show that it reshapes to sech with radiation shedding. The attractor property.

**3b: Propagation loss.** N = 1 soliton with 0.1, 0.5, 1.0 dB/cm loss.
Show adiabatic broadening (the soliton area theorem: amplitude drops, width
increases, product A·T₀ adjusts to maintain N = 1 at the local power).

---

### Cell 4: Adding χ⁽²⁾ — Cascaded Contribution to Effective Nonlinearity

Now turn χ⁽²⁾ ON. Set temperature far from SHG phase matching (T = 150°C,
Δk ≈ 8750/m, deep cascaded regime).

The cascaded χ⁽²⁾ produces an additional effective SPM with:

    n₂_casc ∝ d_eff² / Δk

The sign depends on Δk: positive Δk → positive n₂_casc (adds to Kerr),
negative Δk → negative n₂_casc (subtracts from Kerr).

**Experiment:** At fixed pulse energy (N = 1 for pure Kerr), propagate with
χ⁽²⁾ on at several temperatures:

- T >> T_PM (large positive Δk): n₂_casc small and positive, soliton
  slightly over-compressed (effective N slightly > 1)
- T > T_PM (moderate positive Δk): n₂_casc comparable to n₂_Kerr, soliton
  significantly perturbed — needs less pulse energy to maintain N = 1
- T < T_PM (negative Δk): n₂_casc negative, partially cancels Kerr,
  soliton broadens (effective N < 1)

**Plot A:** Output FWHM vs. temperature at fixed input energy. Shows the
soliton getting tighter or looser as the cascaded contribution adds to
or subtracts from the Kerr nonlinearity.

**Plot B:** Effective soliton energy (the input energy needed for output
FWHM = input FWHM) vs. temperature. This directly measures:

    γ_eff(T) = γ_Kerr + γ_casc(T)

and can be compared to the analytical prediction:

    γ_casc = ω · n₂_casc / (c · A_eff)

where n₂_casc = ±(2/ε₀) · d_eff² / (n_FH² · n_SH · c · Δk).

**This is the key cell.** It demonstrates that temperature is a knob for
continuously adjusting the effective nonlinearity of the waveguide — from
enhanced (Δk > 0) through pure Kerr (Δk → ∞) to reduced or even
sign-reversed (Δk < 0). No χ⁽³⁾-only system can do this.

### Cell 5: Tunable Soliton via Temperature

Same physics as Cell 4, but now actively maintain the soliton by adjusting
energy to track the shifting γ_eff.

**Experiment:** At each temperature, find the fundamental soliton energy
(binary search on energy for output FWHM = input FWHM, or use the
analytical estimate). Then propagate at that energy and record the soliton.

**Plot A:** Soliton energy vs. temperature. The curve shows how much the
χ⁽²⁾ cascading augments or diminishes the Kerr nonlinearity.

**Plot B:** Overlay z–t color maps at three temperatures: one where χ⁽²⁾
adds strongly to Kerr (low soliton energy), one at high T where it's nearly
pure Kerr, and one where χ⁽²⁾ partially cancels Kerr (high soliton energy).
All three show clean soliton propagation but at different energies.

### Cell 6: Strong Coupling Regime — χ⁽²⁾ and χ⁽³⁾ Comparable

Move temperature close to SHG phase matching (Δk · L ~ 10–50). Now the
cascaded phase shift per unit length is comparable to the Kerr phase shift,
AND the SH field is no longer negligible — energy transfer between FH and
SH becomes visible.

**Experiment:** Propagate at the adjusted soliton energy (from Cell 5) at
progressively smaller Δk.

**Expected dynamics:** At some point the perturbative picture breaks down —
the SH field grows large enough to disrupt the soliton, energy oscillates
between FH and SH, and the clean NLSE soliton picture fails. The pulse may
still self-trap (as a two-color bound state) but its properties deviate
from the Kerr soliton predictions.

**Plot A:** z–t color maps for FH and SH at several Δk values. Show the
transition from "Kerr soliton with small SH perturbation" to "two-color
dynamics."

**Plot B:** SH energy fraction vs. Δk at the soliton operating point.
When this exceeds ~5–10%, the perturbative cascaded-Kerr description is
breaking down.

**Plot C:** Output FH pulse FWHM vs. Δk at fixed input energy = pure Kerr
soliton energy. Shows when the soliton destabilizes.

**Connection to the earlier quadratic soliton attempt:** This is where the
pure χ⁽²⁾ soliton would live if it were accessible. With χ⁽³⁾ providing
the self-trapping and χ⁽²⁾ providing additional coupling, we can now
approach the quadratic regime from a soliton that already exists, rather
than trying to create one from scratch in a regime where z₀ is too long.

### Cell 7: Supercontinuum Preview

High-order soliton (N = 5–8) at a temperature where cascaded χ⁽²⁾
enhances the effective nonlinearity. 50 fs pulse, 50 mm waveguide.

**Expected dynamics:** Soliton compression is stronger (because γ_eff >
γ_Kerr), fission happens earlier, and the dispersive wave landscape is
modified by the χ⁽²⁾ coupling — dispersive waves can be emitted into
the SH band as well as around the FH. The result should be broader
spectral coverage than pure Kerr fission.

**Plot:** z–λ color map spanning 700–2200 nm showing the full spectral
evolution. If this spans more than an octave, it's a supercontinuum
generated by combined χ⁽²⁾+χ⁽³⁾ dynamics — the phenomenon the TFLN
community is actively pursuing, modeled by SNOW.

**This is the payoff cell for the entire χ⁽³⁾ extension.**

### Cell 8: Comparison Summary

| Regime       | Cells | χ⁽²⁾ | χ⁽³⁾ | SH energy | Soliton energy | Notes |
|--------------|-------|-------|-------|-----------|----------------|-------|
| Pure Kerr    | 1–3   | Off   | On    | 0         | E_Kerr         | Textbook NLSE |
| Cascaded-enhanced | 4–5 | On (large Δk) | On | < 5% | < E_Kerr | Temperature-tunable γ |
| Strong coupling | 6 | On (moderate Δk) | On | 5–50% | Complex | Two-color dynamics |
| Supercontinuum | 7 | On (optimized Δk) | On | Varies | High N | Broadest spectrum |

---

## Part 3: Notes for CC

1. **Implementation priority: χ⁽³⁾ term first, notebook second.** The Kerr
   term in the NEE is the gating item. Get it working and calibrated (Cell 0)
   before building the soliton cells. The term itself is simple — the
   normalization calibration is the work.

2. **Disabling χ⁽²⁾:** For Cells 0–3, we need χ⁽²⁾ completely off to
   isolate the Kerr physics. Options: set X0 = 0, or set Δk to a huge value
   (no poling, K_g = 0). Whichever is cleaner in the code. The point is that
   Cells 1–3 should exactly reproduce textbook NLSE soliton behavior with no
   χ⁽²⁾ contamination.

3. **The γ calibration will have the same normalization ambiguity as X0.**
   The textbook formula γ = n₂·ω/(c·A_eff) requires knowing A_eff, which
   SNOW's effective-index method doesn't provide. So gamma_eff will likely
   need empirical calibration against analytical SPM predictions, exactly as
   X0 was calibrated against analytical parametric gain. The analytical SPM
   broadening of a sech² pulse is very well characterized (Agrawal Chapter 4),
   so the reference curve is known precisely.

4. **Literature calibration targets for γ in TFLN:**

   | Reference | γ | Geometry | λ |
   |---|---|---|---|
   | Jankowski et al. 2021, Optica 8 532 | ~200 /W/km | 700 nm film | 2 µm |
   | Yu et al. 2019, Opt. Express 27 23271 | ~100 /W/km | 600 nm film | 1.5 µm |
   | Mishra et al. 2022, Opt. Lett. 47 | ~150 /W/km | 700 nm film | 1.55 µm |

   These are order-of-magnitude guides. Exact values depend on geometry.

5. **n₂ for lithium niobate:** The commonly cited value is n₂ ≈ 1.8 × 10⁻¹⁹
   m²/W at 1064 nm (DeSalvo et al. 1996), but it varies with wavelength and
   crystal composition. At 1550 nm it may be somewhat lower. The calibration
   procedure sidesteps this uncertainty — we measure γ_eff directly rather
   than computing it from n₂.

6. **GPU performance:** The |A|²·A term adds one element-wise multiplication
   per step — negligible compared to the FFTs in the dispersion operator.
   No performance concern.

7. **Backward compatibility:** The default gamma_eff = 0 means every existing
   notebook runs identically. No risk of breaking Tutorials 1–7 or the CW
   OPA notebook.

8. **Future extensions (not for now, but the roadmap):**
   - Raman delayed response: replace |A|²·A with
     (1−f_R)·|A|²·A + f_R·A·∫h_R(t')|A(t−t')|²dt'. Adds soliton
     self-frequency shift, important for supercontinuum.
   - Self-steepening: add (i/ω₀)·∂/∂t(|A|²·A). Important for few-cycle
     pulses.
   - XPM between FH and SH: adds |A_SH|²·A_FH and |A_FH|²·A_SH cross terms.
     Important when both fields are strong.
   Each of these is a separable addition to the same nonlinear product
   function.

9. **Publication angle strengthened.** A TFLN simulation tool with validated
   χ⁽²⁾ AND χ⁽³⁾, demonstrating temperature-tunable soliton properties via
   cascading, is a genuinely useful contribution. The story — "Kerr solitons
   in TFLN waveguides with continuously tunable nonlinearity via quasi-phase-
   mismatched χ⁽²⁾ cascading" — is clean, novel at the simulation level,
   and directly relevant to the TFLN photonics community.
