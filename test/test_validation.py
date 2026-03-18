"""
Physics-backed validation tests for snow.

These tests verify the simulation results against known analytical solutions,
published literature values, and fundamental physical laws.
"""
import pytest
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy.constants import pi, c

from snow import materials, waveguides, pulses, nlo, util

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
nm = 1e-9
um = 1e-6
mm = 1e-3
ps = 1e-12
fs = 1e-15
MHz = 1e6
pJ = 1e-12


def make_grid(lam_start=800*nm, lam_stop=3*um, N=2**10):
    """Standard time/frequency grid for tests."""
    f_max = c / lam_start
    f_min = c / lam_stop
    BW = f_max - f_min
    dt = 1 / BW
    T = N / BW
    t = -T/2 + np.arange(0, T, step=dt)
    f_ref = (f_max + f_min) / 2
    return t, f_ref


def make_waveguide(alpha_dBcm=0.0, X0=1.1e-12, L=4*mm, pp=5.18*um):
    """Standard TFLN waveguide used in tutorials."""
    wg = waveguides.waveguide(
        w_top=1800*nm, h_thinfilm=700*nm, h_etch=350*nm,
        tf_material='LN_MgO_e', box_material='SiO2', clad_material='Air')
    alpha = util.absorption_coeff(alpha_dBcm) if alpha_dBcm > 0 else 0.0
    wg.set_loss(alpha)
    wg.set_length(L)
    wg.add_poling(lambda z: np.sign(np.cos(z * 2*pi / pp)))
    wg.set_nonlinear_coeffs(N=1, X0=X0)
    return wg


def rms_width(t, x):
    """RMS temporal width of a pulse."""
    I = np.abs(x)**2
    dt = t[1] - t[0]
    norm = np.sum(I) * dt
    t_mean = np.sum(t * I) * dt / norm
    return np.sqrt(np.sum((t - t_mean)**2 * I) * dt / norm)


# ===========================================================================
# Group 1: Materials — Sellmeier equations vs published literature
# ===========================================================================

class TestMaterials:
    """Validate refractive indices against published Sellmeier data."""

    def test_SiO2_literature(self):
        """SiO2 (Malitson 1965) at standard wavelengths."""
        # Values from refractiveindex.info / Malitson
        assert materials.refractive_index('SiO2', 0.5893) == pytest.approx(1.4585, rel=1e-3)
        assert materials.refractive_index('SiO2', 1.064) == pytest.approx(1.4496, rel=1e-3)
        assert materials.refractive_index('SiO2', 1.55) == pytest.approx(1.4440, rel=1e-3)

    def test_LN_MgO_birefringence(self):
        """Ordinary index must exceed extraordinary for LN (negative uniaxial)."""
        for wl in [0.5, 1.0, 1.55, 2.0, 3.0]:
            no = materials.refractive_index('LN_MgO_o', wl)
            ne = materials.refractive_index('LN_MgO_e', wl)
            assert no > ne, f"LN birefringence violated at {wl} um"

    def test_LN_MgO_e_literature(self):
        """MgO:LN extraordinary at common wavelengths.
        Reference values from the Sellmeier coefficients used in materials.py
        (Gayer et al. / Schlarb & Betzler parametrization for 5% MgO:LN)."""
        assert materials.refractive_index('LN_MgO_e', 1.064) == pytest.approx(2.1474, rel=2e-3)
        assert materials.refractive_index('LN_MgO_e', 1.55) == pytest.approx(2.1314, rel=2e-3)
        assert materials.refractive_index('LN_MgO_e', 2.0) == pytest.approx(2.1180, rel=2e-3)

    def test_temperature_model_consistency(self):
        """Temperature-dependent model at T=24.5 should roughly match the static model."""
        for wl in [1.0, 1.55, 2.0]:
            n_static = materials.refractive_index('LN_MgO_e', wl)
            n_T = materials.refractive_index('LN_MgO_e_T', wl, T=24.5)
            assert n_T == pytest.approx(n_static, rel=5e-3)

    def test_wavelength_unit_invariance(self):
        """Auto-detection of wavelength units (um, nm, m) should give same result."""
        n_um = materials.refractive_index('SiO2', 1.55)
        n_nm = materials.refractive_index('SiO2', 1550)
        n_m = materials.refractive_index('SiO2', 1.55e-6)
        assert n_um == pytest.approx(n_nm, rel=1e-6)
        assert n_um == pytest.approx(n_m, rel=1e-6)

    def test_normal_dispersion_monotonicity(self):
        """In the transparency window, n should decrease with wavelength (normal dispersion)."""
        wl = np.linspace(0.5, 3.5, 50)
        for mat in ['SiO2', 'LN_MgO_e']:
            n = materials.refractive_index(mat, wl)
            assert np.all(np.diff(n) < 0), f"{mat} not monotonically decreasing"


# ===========================================================================
# Group 2: Pulse properties
# ===========================================================================

class TestPulses:
    """Validate pulse generation and properties."""

    @pytest.fixture
    def grid(self):
        return make_grid(lam_start=800*nm, lam_stop=4.5*um, N=2**12)

    def test_gaussian_energy(self, grid):
        """Gaussian pulse energy matches requested value."""
        t, f_ref = grid
        E_req = 40 * pJ
        p = pulses.gaussian_pulse(t, 80*fs, f_ref=f_ref, Energy=E_req,
                                  f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        assert p.energy_td() == pytest.approx(E_req, rel=1e-3)

    def test_sech_energy(self, grid):
        """Sech pulse energy matches requested value."""
        t, f_ref = grid
        E_req = 40 * pJ
        p = pulses.sech_pulse(t, 80*fs, f_ref=f_ref, Energy=E_req,
                              f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        assert p.energy_td() == pytest.approx(E_req, rel=1e-3)

    def test_parseval_theorem(self, grid):
        """Energy computed in time and frequency domains must agree (Parseval)."""
        t, f_ref = grid
        p = pulses.gaussian_pulse(t, 80*fs, f_ref=f_ref, Energy=10*pJ,
                                  f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        assert p.energy_td() == pytest.approx(p.energy_fd(), rel=1e-4)

    def test_gaussian_peak_power(self, grid):
        """Gaussian peak power = 0.94 * Energy / FWHM."""
        t, f_ref = grid
        E = 40 * pJ
        tau = 80 * fs
        p = pulses.gaussian_pulse(t, tau, f_ref=f_ref, Energy=E,
                                  f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        Ppeak_expected = 0.94 * E / tau
        Ppeak_measured = np.max(np.abs(p.a)**2)
        assert Ppeak_measured == pytest.approx(Ppeak_expected, rel=1e-2)

    def test_sech_peak_power(self, grid):
        """Sech peak power = 0.88 * Energy / FWHM."""
        t, f_ref = grid
        E = 40 * pJ
        tau = 80 * fs
        p = pulses.sech_pulse(t, tau, f_ref=f_ref, Energy=E,
                              f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        Ppeak_expected = 0.88 * E / tau
        Ppeak_measured = np.max(np.abs(p.a)**2)
        assert Ppeak_measured == pytest.approx(Ppeak_expected, rel=1e-2)

    def test_gaussian_from_peak_power(self, grid):
        """Creating pulse from Ppeak should give correct energy."""
        t, f_ref = grid
        Ppeak = 500.0
        tau = 80 * fs
        p = pulses.gaussian_pulse(t, tau, f_ref=f_ref, Ppeak=Ppeak,
                                  f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        E_expected = Ppeak * tau / 0.94
        assert p.energy_td() == pytest.approx(E_expected, rel=1e-2)

    def test_pulse_centering(self, grid):
        """Pulse center should be near zero, and shift correctly with offset."""
        t, f_ref = grid
        p = pulses.gaussian_pulse(t, 80*fs, f_ref=f_ref, Energy=10*pJ,
                                  f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        assert p.time_center() == pytest.approx(0.0, abs=1*fs)
        t_offset = 0.5 * ps
        pulses.add_t_offset(p, t_offset)
        # add_t_offset applies exp(j*t_offset*Omega) which shifts by -t_offset
        assert abs(p.time_center()) == pytest.approx(t_offset, rel=1e-2)

    def test_pulse_addition_energy(self, grid):
        """Adding two non-overlapping pulses: total energy = sum of energies."""
        t, f_ref = grid
        p1 = pulses.gaussian_pulse(t, 50*fs, f_ref=f_ref, Energy=10*pJ,
                                   f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        p2 = pulses.gaussian_pulse(t, 50*fs, f_ref=f_ref, Energy=20*pJ,
                                   f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        pulses.add_t_offset(p2, 2*ps)  # separate them in time
        p3 = p1 + p2
        assert p3.energy_td() == pytest.approx(
            p1.energy_td() + p2.energy_td(), rel=1e-3)


# ===========================================================================
# Group 3: Energy conservation in NEE solver
# ===========================================================================

class TestEnergyConservation:
    """Verify energy conservation and loss behavior of the NEE solver."""

    @pytest.fixture
    def grid(self):
        return make_grid()

    def test_linear_lossless(self, grid):
        """Zero nonlinearity, zero loss: energy must be exactly conserved."""
        t, f_ref = grid
        pump = pulses.sech_pulse(t, 100*fs, f_ref=f_ref, Energy=10*pJ,
                                 f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        wg = make_waveguide(alpha_dBcm=0.0, X0=0.0, L=5*mm)
        v_ref = 1 / wg.beta1(1*um)
        out, _ = wg.propagate_NEE(pump, v_ref=v_ref, verbose=False)
        assert out.energy_td() == pytest.approx(pump.energy_td(), rel=1e-4)

    def test_nonlinear_lossless(self, grid):
        """Nonlinear but lossless: Manley-Rowe conservation."""
        t, f_ref = grid
        pump = pulses.sech_pulse(t, 100*fs, f_ref=f_ref, Energy=10*pJ,
                                 f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        wg = make_waveguide(alpha_dBcm=0.0, X0=1.1e-12, L=5*mm)
        v_ref = 1 / wg.beta1(1*um)
        out, _ = wg.propagate_NEE(pump, v_ref=v_ref, verbose=False)
        assert out.energy_td() == pytest.approx(pump.energy_td(), rel=1e-3)

    def test_linear_with_loss(self, grid):
        """Linear propagation with loss: Beer-Lambert attenuation."""
        t, f_ref = grid
        pump = pulses.sech_pulse(t, 100*fs, f_ref=f_ref, Energy=10*pJ,
                                 f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        Alpha_dBcm = 1.0  # fairly strong loss to make effect visible
        L = 5 * mm
        alpha = util.absorption_coeff(Alpha_dBcm)
        wg = make_waveguide(alpha_dBcm=Alpha_dBcm, X0=0.0, L=L)
        v_ref = 1 / wg.beta1(1*um)
        out, _ = wg.propagate_NEE(pump, v_ref=v_ref, verbose=False)
        E_expected = pump.energy_td() * np.exp(-alpha * L)
        assert out.energy_td() == pytest.approx(E_expected, rel=5e-2)


# ===========================================================================
# Group 4: Linear dispersive propagation (GVD broadening)
# ===========================================================================

class TestDispersion:
    """Validate linear dispersion against analytical GVD broadening."""

    def test_gaussian_gvd_broadening(self):
        """Transform-limited Gaussian broadens by known factor under GVD."""
        t, f_ref = make_grid(lam_start=800*nm, lam_stop=1.5*um, N=2**12)
        tau = 100 * fs
        lam_p = 1.0 * um
        pump = pulses.gaussian_pulse(t, tau, f_ref=f_ref, Energy=1*pJ,
                                     f0=c/lam_p, Npwr_dB=200, frep=250*MHz)
        wg = make_waveguide(alpha_dBcm=0.0, X0=0.0, L=20*mm)
        wg.set_length(20*mm)
        beta2_val = float(wg.beta2(np.array([lam_p])))
        L = 20 * mm

        v_ref = 1 / wg.beta1(lam_p)
        out, _ = wg.propagate_NEE(pump, v_ref=v_ref, verbose=False)

        # Analytical: sigma_out = sigma_in * sqrt(1 + (beta2*L / (2*sigma^2))^2)
        sigma_in = rms_width(t, pump.a)
        broadening = np.sqrt(1 + (beta2_val * L / (2 * sigma_in**2))**2)
        sigma_out_expected = sigma_in * broadening
        sigma_out_measured = rms_width(t, out.a)

        assert sigma_out_measured == pytest.approx(sigma_out_expected, rel=5e-3)

    def test_spectrum_invariant_under_gvd(self):
        """Power spectrum must not change under pure linear dispersion."""
        t, f_ref = make_grid(lam_start=800*nm, lam_stop=1.5*um, N=2**12)
        pump = pulses.gaussian_pulse(t, 100*fs, f_ref=f_ref, Energy=1*pJ,
                                     f0=c/(1*um), Npwr_dB=200, frep=250*MHz)
        wg = make_waveguide(alpha_dBcm=0.0, X0=0.0, L=20*mm)
        v_ref = 1 / wg.beta1(1*um)
        out, _ = wg.propagate_NEE(pump, v_ref=v_ref, verbose=False)

        spec_in = np.abs(fft(pump.a))**2
        spec_out = np.abs(fft(out.a))**2
        # Normalize and compare
        spec_in /= np.max(spec_in)
        spec_out /= np.max(spec_out)
        np.testing.assert_allclose(spec_out, spec_in, atol=1e-3)


# ===========================================================================
# Group 5: SHG — quadratic power/length scaling
# ===========================================================================

class TestSHG:
    """Validate SHG in the undepleted-pump regime."""

    @staticmethod
    def _shg_energy(t, f_ref, energy, L, X0=1.1e-12):
        """Run SHG and return energy in the SH band."""
        pump = pulses.sech_pulse(t, 100*fs, f_ref=f_ref, Energy=energy,
                                 f0=c/(2*um), Npwr_dB=200, frep=250*MHz)
        wg = make_waveguide(alpha_dBcm=0.0, X0=X0, L=L)
        v_ref = 1 / wg.beta1(1*um)
        out, _ = wg.propagate_NEE(pump, v_ref=v_ref, verbose=False)
        # Filter the SH around 1 um
        sh = out.apply_filter(c/(1*um), 50e12)  # 50 THz bandwidth
        return sh.energy_td()

    def test_quadratic_power_scaling(self):
        """SH energy scales as pump energy squared (undepleted regime)."""
        t, f_ref = make_grid()
        # Very low energy to stay in undepleted regime
        E1 = 0.1 * pJ
        E2 = 0.2 * pJ
        E_sh1 = self._shg_energy(t, f_ref, E1, L=1*mm)
        E_sh2 = self._shg_energy(t, f_ref, E2, L=1*mm)
        ratio = E_sh2 / E_sh1
        assert ratio == pytest.approx(4.0, rel=0.15)

    def test_quadratic_length_scaling(self):
        """SH energy scales as L squared (undepleted regime, short crystal)."""
        t, f_ref = make_grid()
        E = 0.1 * pJ
        E_sh1 = self._shg_energy(t, f_ref, E, L=0.5*mm)
        E_sh2 = self._shg_energy(t, f_ref, E, L=1.0*mm)
        ratio = E_sh2 / E_sh1
        assert ratio == pytest.approx(4.0, rel=0.15)


# ===========================================================================
# Group 6: Waveguide dispersion properties
# ===========================================================================

class TestWaveguideDispersion:
    """Validate waveguide mode calculations against physical constraints."""

    @pytest.fixture
    def wg(self):
        return waveguides.waveguide(
            w_top=1800*nm, h_thinfilm=700*nm, h_etch=350*nm,
            tf_material='LN_MgO_e', box_material='SiO2', clad_material='Air')

    def test_neff_bounded_by_materials(self, wg):
        """Effective index must satisfy n_clad < n_eff < n_core."""
        for wl in [0.8*um, 1.0*um, 1.55*um, 2.0*um]:
            neff = wg.neff(wl)
            n_core = materials.refractive_index('LN_MgO_e', wl)
            n_clad = materials.refractive_index('SiO2', wl)
            assert np.all(neff > n_clad), f"neff below cladding at {wl/um} um"
            assert np.all(neff < n_core), f"neff above core at {wl/um} um"

    def test_group_velocity_subluminal(self, wg):
        """Group velocity must be positive and less than c."""
        for wl in [0.8*um, 1.0*um, 1.55*um, 2.0*um]:
            vg = 1.0 / wg.beta1(wl)
            assert np.all(vg > 0), f"negative group velocity at {wl/um} um"
            assert np.all(vg < c), f"superluminal group velocity at {wl/um} um"

    def test_gvm_ff_sh(self, wg):
        """GVM between 2 um and 1 um matches tutorial value (~13.3 fs/mm)."""
        gvm = wg.beta1(1*um) - wg.beta1(2*um)
        assert gvm / (fs / mm) == pytest.approx(13.27, rel=0.02)

    def test_beta2_sign(self, wg):
        """GVD at 1 um should be normal (positive beta2) for this waveguide."""
        beta2 = wg.beta2(np.array([1*um]))
        assert beta2 > 0

    def test_beta2_anomalous_at_2um(self, wg):
        """GVD at 2 um should be anomalous (negative beta2) for this waveguide."""
        beta2 = wg.beta2(np.array([2*um]))
        assert beta2 < 0


# ===========================================================================
# Group 7: Utility functions
# ===========================================================================

class TestUtil:

    def test_absorption_coeff_conversion(self):
        """1 dB/cm should convert to ~23.03 /m."""
        alpha = util.absorption_coeff(1.0)
        assert alpha == pytest.approx(23.026, rel=1e-3)

    def test_richardson_derivative_polynomial(self):
        """Richardson extrapolation on x^3 should give 3x^2."""
        f = lambda x: x**3
        d = util.derivative(f, 2.0, 2, 0.1)
        assert d == pytest.approx(12.0, rel=1e-6)

    def test_richardson_derivative_trig(self):
        """Richardson extrapolation on sin(x) should give cos(x)."""
        d = util.derivative(np.sin, 1.0, 3, 0.01)
        assert d == pytest.approx(np.cos(1.0), rel=1e-8)
