# Quadratic Soliton — Simulation Findings

## The z₀ arithmetic error

The brief's response estimated z₀ = (500 fs)² / (9 fs²/mm) ≈ 28 mm. The
correct value is **27,778 mm = 27.8 m** (factor of 1000 error — the units
didn't cancel as written). This changes the entire operating picture:

| λ_FH | FWHM | β₂ | z₀ | Soliton period | Fits in 40mm? |
|------|------|----|-----|----------------|---------------|
| 2000 nm | 500 fs | -9 fs²/mm | 9.0 m | 14.1 m | No (0.3%) |
| 2000 nm | 100 fs | -9 fs²/mm | 360 mm | 565 mm | No (7%) |
| 2000 nm | 50 fs | -9 fs²/mm | 90 mm | 141 mm | No (28%) |
| 1550 nm | 100 fs | -50 fs²/mm | 64 mm | 101 mm | Marginal (40%) |
| **1550 nm** | **50 fs** | **-50 fs²/mm** | **16 mm** | **25 mm** | **Yes (160%)** |

Only 1550 nm / 50 fs gives z₀ short enough for visible soliton dynamics in
a waveguide-length crystal.

## The GVM problem at 1550 nm

GVM(1550/775 nm) = 153 ps/mm. For a 50 fs pulse, the walk-off length is
0.33 µm. The SH walks entirely through the FH within a fraction of one
coherence length. Tested at T=60°C (Δk=2007/m):

- All energies (0.01–5 pJ) show pure dispersive broadening
- FWHM grows 25–52× over 20 mm
- Peak power drops to 10–15% of input
- The cascaded Kerr from walk-through is too weak to balance β₂=-50 fs²/mm

**Verdict: 1550 nm doesn't work** for the cascaded-Kerr soliton despite
having the right z₀. The GVM kills the cascading mechanism.

## What DOES work at 2000 nm

At T=25°C (Δk=1785/m, Δk·L=71), 100 fs pulses show clear nonlinear dynamics:

| Energy | Ppeak in→out | FWHM ratio | Regime |
|--------|-------------|------------|--------|
| 1 pJ | 9→15 W | 1.128 | Slight broadening |
| 3 pJ | 26→41 W | 1.674 | Broadening |
| 7 pJ | 62→95 W | 0.109 | Strong compression |
| 10 pJ | 88→203 W | 0.073 | Extreme compression |

**But this is not the cascaded-Kerr regime.** At T=25°C, 40–60% of the
pulse energy transfers to the SH. This is a genuinely quadratic (two-color)
interaction, not a perturbative cascading. The SH is a dynamically active
partner, not a slaved perturbation.

Peak power oscillates along z (breathing), and the compression at 7–10 pJ
is dramatic — consistent with higher-order quadratic soliton dynamics.

## The regime map

| Δk | SH fraction | Physics | Soliton? |
|----|------------|---------|----------|
| Small (~1800/m, T≈25°C) | 40–60% | Quadratic: two-color bound state | Complex dynamics, hard to isolate |
| Moderate (~5000/m, T≈100°C) | 5–10% | Transitional | Needs ~50 pJ for visible effect |
| Large (~10000/m, T≈175°C) | <1% | Cascaded Kerr | Needs ~200 pJ and/or much longer L |

The clean cascaded-Kerr soliton (SH negligible, effective-NLSE behavior)
requires either very large Δk (weak effective n₂, needing very high power)
or much longer crystals than 40 mm. Neither is practical with β₂ = -9 fs²/mm.

## Options for the notebook

**Option A: Embrace the quadratic regime at 2000 nm.**
Work at T≈25°C where the two-color dynamics are strong and visible in 40 mm.
This skips the "clean cascaded Kerr" case but demonstrates the richer
quadratic soliton physics (Cell 3 of the brief). The cascaded-Kerr case
becomes a high-Δk limiting discussion rather than a primary demonstration.

**Option B: Engineer a different waveguide for stronger β₂.**
A narrower or thinner TFLN waveguide could push |β₂| to 50–200 fs²/mm at
2000 nm (by moving the ZDW). SNOW's effective index method supports arbitrary
geometries. This would bring z₀ down to ~10 mm and make cascaded solitons
visible at moderate Δk. But it requires finding a geometry that works, and
the mode area changes would invalidate X0 = 1.1e-12.

**Option C: Use bulk crystal mode with large beam area.**
The bulk crystal class (`crystals.py`) allows arbitrary n(λ) and beam area.
In bulk LN at 2000 nm, β₂ is similar but the interaction length can be much
longer (50–100 mm PPLNs exist). The calibrated X0 = 1.2e-14 applies. But
this loses the waveguide confinement advantage.

**Recommendation: Option A.** The quadratic soliton at moderate Δk is the
most interesting physics that SNOW can demonstrate with existing parameters.
The cascaded-Kerr limit can be discussed analytically and shown as a
limiting case rather than a primary demonstration.
