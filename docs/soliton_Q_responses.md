# Quadratic Soliton Brief — Response to CC's Questions

## Q1: GVM at 1550/775 nm — the walk-off problem

CC is right to flag this. 153 ps/mm GVM is a dealbreaker for the naive
cascaded-Kerr picture at femtosecond timescales.

### The core tension

The cascaded-Kerr effective n₂ assumes the SH is "slaved" to the FH — it's
generated, accumulates a phase mismatch, and converts back, all locally. This
requires the walk-off length (L_w = T₀/GVM) to be at least comparable to the
coherence length (L_coh = π/Δk). Let's check:

- At T = 100°C: Δk ≈ 4900/m → L_coh ≈ 640 µm
- At T₀ = 100 fs: L_w = 100 fs / 153 fs/mm = 0.65 µm

L_w is **1000× shorter** than L_coh. The SH leaves the FH pulse entirely
before completing even a fraction of one cascade cycle. The local effective-
Kerr picture is broken.

Longer pulses help L_w but hurt the soliton dynamics. The soliton period is
z₀ = T₀²/|β₂|. For soliton evolution to be visible in a 20 mm crystal, we
need z₀ ≲ 20 mm, which means T₀ ≲ √(z₀·|β₂|) = √(20 mm × 50 fs²/mm) ≈ 1 ps.
But at T₀ = 1 ps, L_w = 6.5 µm — still 100× too short. At T₀ = 50 ps, L_w
and L_coh become comparable, but z₀ = 50 m — no soliton dynamics in any
reasonable crystal.

**At 1550/775 nm, there is no pulse duration that simultaneously allows
cascaded-Kerr soliton formation and fits within a ≤50 mm crystal.**

### Recommendation: use 2000/1000 nm for the primary demonstration

At 2000/1000 nm:

| Parameter        | Value                     |
|------------------|---------------------------|
| GVM              | ~15.3 ps/mm (10× better)  |
| β₂ at 2000 nm    | −9 fs²/mm (weaker but OK) |

With T₀ = 500 fs:
- L_w = 500 fs / 15.3 fs/mm ≈ 33 µm
- z₀ = (500 fs)² / 9 fs²/mm ≈ 28 mm
- L_coh at moderate Δk (say 3000/m) ≈ 1 mm

L_w is still 30× shorter than L_coh, so the SH walks through the FH pulse
quickly. But the *integrated* phase shift accumulated over many L_coh periods
along the crystal still produces the effective Kerr effect — the walk-through
averages the cascading rather than destroying it. The key is that the total
crystal length L >> L_coh, which it is (20 mm / 1 mm = 20 coherence lengths).

The soliton period z₀ ≈ 28 mm means you'll see the onset of self-trapping in
20 mm but not a full period. Going to 40–50 mm (available in TFLN) gives
nearly two soliton periods, enough for clear soliton dynamics.

**Parameters for the clean cascaded-Kerr soliton demonstration:**

- Wavelength: 2000 nm FH / 1000 nm SH
- Poling period: 5.18 µm (from Tutorial 7 — reuse directly)
- Pulse duration: 300–500 fs sech²
- Crystal length: 40–50 mm (or 20 mm, accepting partial evolution)
- Temperature: scan from ~50°C upward for increasing Δk

### 1550 nm as an advanced cell: walk-off-dominated regime

The 1550/775 nm case isn't useless — it operates in a different and arguably
more interesting regime. When L_w << L_coh, the SH pulse walks entirely
through the FH pulse, and the integrated effect is a **nonlocal** nonlinear
response — the phase shift at a given point in the FH pulse depends on the
intensity at earlier (or later) points, mediated by the transiting SH. This
is mathematically analogous to the Raman delayed nonlinear response in fiber,
and it supports "walk-off solitons" that are distinct from Kerr solitons.

This is publishable-grade physics but pedagogically harder. Recommend: do the
clean cascaded-Kerr case at 2000 nm first (Cells 1–4), then add a Cell
exploring the walk-off-dominated 1550 nm regime with commentary explaining
why it behaves differently.

### Revised operating parameters table

| Cell     | λ_FH   | T₀      | Crystal | Regime                 |
|----------|--------|---------|---------|------------------------|
| 1–4      | 2000 nm| 500 fs  | 40 mm   | Cascaded Kerr          |
| Advanced | 1550 nm| 500 fs  | 20 mm   | Walk-off / nonlocal    |

---

## Q2: z-resolved output — implement option (c)

Option (a) is tolerable but wasteful. Option (c) — a `z_save` array added to
`nlo_scipy.py` that records `ifft(Y·exp(−iDz))` at specified z positions
during integration — is the right investment. Reasons:

1. It's a small, clean code change (a callback during the ODE integration)
2. It benefits all notebooks going forward, not just this one
3. The bulk OPA notebook (GVM cell) and even the CW OPA could use it
   retroactively
4. The (z, t) color maps are the single most illuminating soliton
   visualization — they're worth getting right

Go with (c).

---

## Q3: Grid sizing — agreed

N = 2¹²–2¹³ spanning ~700–2200 nm (needs to cover both FH and SH for either
operating wavelength, plus room for spectral broadening in the soliton
compression cell). No bin-alignment tricks needed — broadband pulse content
means spectral leakage from bin misalignment is negligible.

If using 2000/1000 nm, the spectral window should extend to at least 900 nm
on the short side (to capture the full SH pulse bandwidth) and 2200 nm on the
long side (dispersive wave emission, spectral broadening).

---

## Q4: Kg vs explicit poling — agreed

Use analytical Kg for Cells 0–4. Switch to explicit poling function only for
Cell 5c (poling disorder). This is the right separation:

- Kg is cleaner numerically, avoids resolving the ~5 µm period on the z-grid,
  and the phase mismatch Δk emerges naturally from the difference between
  Kg and the waveguide dispersion
- Explicit poling is only needed when the point of the cell is to perturb the
  domain structure
- Mixing the two approaches within a single cell would be confusing

One implementation note: when switching to explicit poling in Cell 5c, verify
that the clean-poling explicit case reproduces the Kg result to within
numerical precision before adding disorder. This confirms the two approaches
are consistent and any differences in 5c are genuinely from the disorder.

---

## Updated brief summary

The main change from the original brief: **primary operating wavelength moves
to 2000 nm** to avoid the GVM problem. This reuses the Tutorial 7 poling
period (5.18 µm) with temperature detuning, which is a nice continuity.
The 1550 nm case becomes an advanced exploration of walk-off-dominated
nonlocal soliton physics rather than the primary demonstration.

Cell 0 (dispersion landscape) should now include both wavelengths —
compute β₂, GVM, L_w, and L_coh as functions of wavelength to make the
case for why 2000 nm works and 1550 nm doesn't for the standard cascaded
picture. This comparison IS the physics lesson.
