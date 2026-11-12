"""
Infinite approximation for horizontal disks, based on Tsang et al. 2000 (page 51) and LeVine et al. 1983, 1984, 1985.
The height T must be small with respect to the wavelength: k T sqrt(eps) << 1.

This approximation assumes that the disk responds to an incoming wave as if it
were of infinite radius (the electric field is uniform horizontally), but radiates as a disk of finite radius.
"""

import numpy as np
import scipy.integrate
import scipy.special

# local import
from smrt.core.error import SMRTError
from smrt.core.fresnel import fresnel_reflection_coefficients_maezawa09_rigorous_full_output
from smrt.core.globalconstants import C_SPEED
from smrt.core.lib import abs2, cached_roots_legendre, smrt_matrix
from smrt.emmodel.emmodel_utils import (
    GenericFTPhaseMixin,
    extinction_matrix,
    phase_matrix_from_scattering_amplitude,
    vectorize_angles,
)


class InfiniteDisk(GenericFTPhaseMixin):
    """
    InfiniteDisk  model class.

    Args:
        sensor: Object containing sensor characteristics.
        layer: Object containing snow layer characteristics (single layer).
        use_optical_theorem: use the optical_theorem to compute ke instead of summing ks and ka. This can be faster but
            usually less accurate.
    """

    _respect_reciprocity_principle = False

    def __init__(self, sensor, layer, *, precompute=True, use_optical_theorem=False):
        self.sensor = sensor
        self.layer = layer
        self.use_optical_theorem = use_optical_theorem

        # Set size of phase matrix: active needs an extended phase matrix
        self.npol = 2 if sensor.mode == "P" else 3

        self.k0 = 2 * np.pi * sensor.frequency / C_SPEED

        if precompute:
            self.precompute()

    def precompute(self):
        sensor = self.sensor
        layer = self.layer

        self.e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        self.eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        self.effective_permittivity = self.e0  # Rayleigh is for sparse medium only

        radius = layer.microstructure.radius
        height = layer.microstructure.height
        self.volume = np.pi * radius**2 * height
        self.n0 = self.layer.microstructure.frac_volume / self.volume  # number density of disks

    def basic_check(self):
        # if we want to be strict, only IndependentDisk should be valid, but in practice any
        # model with a radius and height can make it!
        if not hasattr(self.layer.microstructure, "radius") or not hasattr(self.layer.microstructure, "height"):
            raise SMRTError(
                "Only microstructure_model which defined a `radius` and a `height`can be used for disk scattering"
            )

    def scattering_amplitude(self, mu_s, mu_i, dphi, compute_cross_product=True):
        """compute the scattering amplitude using Tsang et al. 2002 (page 51)"""
        radius = self.layer.microstructure.radius
        height = self.layer.microstructure.height
        half_height = height / 2

        mu_s, sin_s, mu_i, sin_i, cosphi, sinphi = vectorize_angles(
            mu_s, mu_i, dphi, compute_cross_product=compute_cross_product
        )

        radius_range = getattr(self.layer.microstructure, "radius_range", 0)
        radius_sampling = getattr(self.layer.microstructure, "radius_sampling", 0)

        if radius_range > 0:
            # Reshape to get the correct number of axes (axis -1 is the radius variations)
            radius = np.linspace(radius - radius_range / 2, radius + radius_range / 2, radius_sampling).reshape(
                [1] * len(mu_s.shape) + [-1]
            )
            assert (len(radius.shape) == 2) | (len(radius.shape) == 4), (
                "Incorrect number of dimensions for reshaped radius"
            )

            mu_s = mu_s[..., np.newaxis]
            mu_i = mu_i[..., np.newaxis]
            sin_s = sin_s[..., np.newaxis]
            sin_i = sin_i[..., np.newaxis]
            cosphi = cosphi[..., np.newaxis]
            sinphi = sinphi[..., np.newaxis]

        sin2_i = sin_i**2
        si2 = sin_s**2 + sin2_i - 2 * sin_s * sin_i * cosphi
        si2[si2 < 0] = 0

        qsi = self.k0 * np.sqrt(si2)  # Eq 19 in Karam 1998 and Eq. 1.6.120 in Tsang I
        assert np.all(np.isfinite(qsi))
        qsi_a = qsi * radius

        modifying_function = np.ones_like(qsi_a)
        np.divide(
            2 * scipy.special.j1(qsi_a), qsi_a, out=modifying_function, where=qsi_a > 1e-6
        )  # Eq 1.6.121  in Tsang I
        # modifying_function[qsi_a < 1e-6] = 1  # this occurs in the forward direction

        coef = (
            self.k0**2 * (self.eps - self.e0) * (radius**2 * height / 4) * modifying_function
        )  # Eq 1.6.121 in Tsang I

        mu_pi = np.sqrt(self.eps / self.e0 - sin2_i)  # the ratio kipz / k in Eq 1.6.121 in Tsang I
        k_1 = np.sqrt(self.e0) * self.k0  # wavelength in medium 1
        kipz = k_1 * mu_pi
        kiz = k_1 * mu_i
        ksz = k_1 * mu_s

        # The Transverse Electric case first (H-pol)

        RTE_op = (kiz - kipz) / (kiz + kipz)  # Eq 1.6.92

        exp2jkipzt = np.exp(2j * kipz * height)
        expjkipzt = np.exp(1j * kipz * height)

        RTE_expjkizt = RTE_op * (1 - exp2jkipzt) / (1 - RTE_op**2 * exp2jkipzt)  # eq 1.6.95

        # check equation 1.6.94
        # print("left 1.6.94=", kiz / kipz * (1 - RTE_expjkizt) / (1 + RTE_expjkizt))
        # print("right 1.6.94=", (1 + RTE_op * exp2jkipzt) / (1 - RTE_op * exp2jkipzt))

        kratio = kiz / kipz
        Ap = 0.5 * ((1 + kratio) + RTE_expjkizt * (1 - kratio)) * np.exp(1j * (kipz - kiz) * half_height)  # Eq 1.6.96
        Bp = -RTE_op * expjkipzt * Ap  # Eq. 1.6.91

        # check Eq 1.6.97
        # # # Eq 1.6.97:
        # Bp2 = 0.5 * ((1 - kratio) + RTE_expjkizt * (1 + kratio)) * np.exp(1j * (-kipz - kiz) * half_height)
        # print("Bs=", Bp, Bp2)
        # assert np.allclose(Bp, Bp2)

        # the dipole is in the horizontal plane for H-pol
        sinc_plus = sinc((kipz + ksz) * half_height)
        sinc_moins = sinc((kipz - ksz) * half_height)

        p_h = Ap * sinc_moins + Bp * sinc_plus  # Eq 1.6.121

        fhh = coef * (p_h * cosphi)
        fvh = coef * (p_h * mu_s * sinphi)

        # the Transverse Magnetic second (V-pol)

        RTM_op = (self.eps * kiz - self.e0 * kipz) / (self.eps * kiz + self.e0 * kipz)  # Eq 1.6.92
        RTM_expjkizt = RTM_op * (1 - exp2jkipzt) / (1 - RTM_op**2 * exp2jkipzt)  # eq 1.6.112

        eratio = self.e0 / self.eps
        Cp = (
            0.5 * ((eratio + kratio) + RTM_expjkizt * (eratio - kratio)) * np.exp(1j * (kipz - kiz) * half_height)
        )  # Eq 1.6.114
        Dp = -RTM_op * expjkipzt * Cp  # same calculation as Eq 1.6.91  # double checked.

        # check Eq 1.6.115
        #   # Eq 1.6.115  sign error in the first exp
        # Dp2 = 0.5 * ((eratio - kratio) + RTM_expjkizt * (eratio + kratio)) * np.exp(1j * (kipz - kiz) * half_height)
        #   # Eq 1.6.115 with corrected sign
        # Dp3 = 0.5 * ((eratio - kratio) + RTM_expjkizt * (eratio + kratio)) * np.exp(1j * (-kipz - kiz) * half_height)
        # print("Ds", Dp, Dp2, Dp3)
        # assert np.allclose(Dp, Dp3)

        # the dipole has two components for V-pol
        p_hv = Cp * sinc_moins - Dp * sinc_plus  # Eq 1.6.121
        p_vv = Cp * sinc_moins + Dp * sinc_plus  # Eq 1.6.121

        fvv = coef * (p_vv * sin_i * sin_s + p_hv * mu_pi * mu_s * cosphi)  # see notebook sympy
        fhv = -coef * (p_hv * mu_pi * sinphi)  # see notebook sympy

        return fvv, fvh, fhv, fhh

    def ks(self, mu, npol=2):
        k = 8  # number of samples. This should be adaptative depending on the size/wavelength
        mu_s, weights = cached_roots_legendre(2**k + 1)  # np.linspace(-1, 1, 2**k + 1)
        dphi = np.linspace(0, np.pi, 2**k + 1)

        fvv, fvh, fhv, fhh = self.scattering_amplitude(mu_s, mu, dphi, compute_cross_product=True)

        def ks_integrations(y):
            if len(y.shape) == 4:
                # we have first to average on the radius
                y = scipy.integrate.romb(y, 1 / y.shape[3], axis=-1)
            # integrate over phi between 0 and 2 pi  (trapeze rule is optimal for periodic functions)
            y = (y[0] + 2 * np.sum(y[1:-1], axis=0) + y[-1]) * (dphi[1] - dphi[0])  # taking into account the symmetry
            # integrate on mu between -1 and 1
            # y = scipy.integrate.romb(y, mu_s[1] - mu_s[0], axis=0)  use romb but should use Gauss
            y = np.sum(y * weights[:, np.newaxis], axis=0)
            return y

        ks_vv = ks_integrations(abs2(fvv) + abs2(fhv))
        ks_hh = ks_integrations(abs2(fhh) + abs2(fvh))

        if npol == 2:
            return self.n0 * smrt_matrix(np.array([ks_vv, ks_hh]))
        elif npol == 3:
            return self.n0 * smrt_matrix(np.array([ks_vv, ks_hh, 0.5 * (ks_vv * ks_hh)]))

    def phase(self, mu_s, mu_i, phi_s, phi_i=0, npol=2):
        dphi = phi_s - phi_i
        fvv, fvh, fhv, fhh = self.scattering_amplitude(mu_s, mu_i, dphi, compute_cross_product=True)

        if len(fvv.shape) == 4:
            # we must average over the radius first.
            n = fvv.shape[3]
            # the sqrt is is a bit hacky here... To be improved with manual integration of the phase function
            fvv = np.sqrt(scipy.integrate.romb(abs2(fvv), 1 / n, axis=-1))
            fhh = np.sqrt(scipy.integrate.romb(abs2(fhh), 1 / n, axis=-1))
            fvh = np.sqrt(scipy.integrate.romb(abs2(fvh), 1 / n, axis=-1))
            fhv = np.sqrt(scipy.integrate.romb(abs2(fhv), 1 / n, axis=-1))

        return smrt_matrix(
            self.n0 * (4 * np.pi) * phase_matrix_from_scattering_amplitude(fvv, fvh, fhv, fhh, npol=npol)
        )

    def estimate_ft_number_samples(self, m_max: int):
        """
        Estimate the number of samples of dphi for fourier decomposition. Highest efficiency for 2^n numbers.
        """

        # aperture:
        argument = self.k0 * self.layer.microstructure.radius
        nsamples = 16.0 * argument  # computes 16 subsamples per argument
        nsamples = int(2 ** (np.ceil(np.log(nsamples) / np.log(2))))

        return max(4, nsamples)

    def ke(self, mu, npol=2):
        """return the extinction coefficient."""
        if self.use_optical_theorem:
            return self.ke_from_optical_theorem(mu, npol=npol)
        else:
            return self.ke_from_kska(mu, npol=npol)

    def ke_from_optical_theorem(self, mu, npol=2):
        """return the extinction coefficient using the optical theorem.

        LeVine 1983 writes that the infinite disk approximation respect the optical theorem (Eq 27)
        """

        fvv, fvh, fhv, fhh = self.scattering_amplitude(mu, mu, 0, compute_cross_product=False)

        if len(fvv.shape) == 2:
            # we must average over the radius first.
            n = fvv.shape[-1]
            # the sqrt is is a bit hacky here... To be improved with manual integration of the phase function
            fvv = np.sqrt(scipy.integrate.romb(abs2(fvv), 1 / n, axis=-1))
            fhh = np.sqrt(scipy.integrate.romb(abs2(fhh), 1 / n, axis=-1))

        # coefficient for the optical theorem
        coef = self.n0 * (4 * np.pi) / self.k0

        ka_background = (1 - self.layer.frac_volume) * 2 * self.k0 * np.sqrt(self.e0).imag

        return extinction_matrix(coef * fvv.imag + ka_background, coef * fhh.imag + ka_background, npol=npol)

    def ke_from_kska(self, mu, npol=2):
        """return the extinction coefficient using ks and ka calculation"""

        # compute the absorption from the integration of the internal field

        rv, rh, tv, th, *_ = fresnel_reflection_coefficients_maezawa09_rigorous_full_output(self.e0, self.eps, mu)
        r = np.array([rv, rh])
        r2 = r**2  # squared reflection coefficient
        t2 = np.array([tv, th]) ** 2  # squared transmission coefficient

        # beta is expressed in the text between eq 20c and 21
        beta_t = np.sqrt(1 - mu**2)  # sin of the refracted wave (=sin of the incidence wave, Snell-Descartes)
        beta_z = np.sqrt(self.eps - 1 + mu**2).real  # cosine of the refracted in the medium

        height_factor = 2 * self.k0 * beta_z * self.layer.microstructure.height

        pola_factor = np.array([(beta_t**2 - beta_z**2) / self.eps.real, np.ones_like(beta_z)])

        ka = (
            (self.layer.frac_volume * self.k0 * self.eps.imag * t2)
            * ((1 + r2) - pola_factor * r * np.sinc(height_factor / np.pi))
            / (1 + r2**2 - 2 * r2 * np.cos(height_factor))
        )

        ks_v, ks_h = self.ks(mu).values

        ka_background = (1 - self.layer.frac_volume) * 2 * self.k0 * np.sqrt(self.e0).imag

        return extinction_matrix(ks_v + ka[0].real + ka_background, ks_h + ka[1].real + ka_background, npol=npol)


def sinc(x):
    y = np.ones_like(x)
    np.divide(np.sin(x), x, out=y)
    return y
