"""
Provide Generalized Rayleigh-Gans approximation for horizontal thin disks and vertical thin cylinders,
based on Karam et al. 1988. This base class performs most of the calculation, only specific code to the disk and
thin_cylinder are in the module :py:mod:`grg_disk` and :py:mod:`grg_cylinder`

The smallest dimension D (either the radius or height) must be small with respect to the wavelength:
k D sqrt(eps) << 1.

References:
M. A. Karam, A. K. Fung and Y. M. M. Antar, "Electromagnetic wave scattering from some vegetation samples,"
in IEEE Transactions on Geoscience and Remote Sensing, vol. 26, no. 6, pp. 799-808, Nov. 1988, doi: 10.1109/36.7711.

For a less restrictive formulation for disks, see Le Vine et al. 1983, 1985 or Tsang's book I, page 51.
TODO: implement also infinite cylinders

"""

import numpy as np
import scipy.integrate
import scipy.special

from smrt.core.globalconstants import C_SPEED

# local import
from smrt.core.lib import abs2, cached_roots_legendre, smrt_matrix
from smrt.emmodel.emmodel_utils import (
    GenericFTPhaseMixin,
    extinction_matrix,
    phase_matrix_from_scattering_amplitude,
    vectorize_angles,
)


class GeneralizedRayleighGansBase(GenericFTPhaseMixin):
    _respect_reciprocity_principle = False

    def __init__(self, sensor, layer, *, precompute=True):
        self.sensor = sensor
        self.layer = layer

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

        self.basic_check()

        self.effective_permittivity = self.e0  # Rayleigh is for sparse medium only

        self.volume = np.pi * layer.microstructure.radius**2 * layer.microstructure.height
        self.n0 = layer.microstructure.frac_volume / self.volume  # number density of disks

        gT, gN = self.demagnetizing_factors()

        # polarizability factors
        eps_contrast = self.eps / self.e0 - 1
        self.aT = 1 / (eps_contrast * gT + 1)  # Eq 64
        self.aN = 1 / (eps_contrast * gN + 1)  # Eq 6

        a2 = (2 * abs2(self.aT) + abs2(self.aN)) / 3  # Eq 21 in Maetzler 1998

        self.ka = (
            layer.frac_volume * self.k0 * self.eps.imag * a2
            + (1 - layer.frac_volume) * 2 * self.k0 * np.sqrt(self.e0).imag
        )  # Eq 24 in Maetzler 1998

    def scattering_amplitude(self, mu_s, mu_i, dphi, compute_cross_product=True):
        """compute the scattering amplitude using Karam et al. 1988"""
        radius = self.layer.microstructure.radius
        height = self.layer.microstructure.height

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

        modifying_function = self.modifying_function(mu_s, sin_s, mu_i, sin_i, cosphi, sinphi)

        # coef = self.k0**2 * (self.eps - self.e0) * self.volume / (4 * np.pi) * modifying_function  # Eq 13
        coef = self.k0**2 * (self.eps - self.e0) * (radius**2 * height / 4) * modifying_function  # Eq 13

        fvv = coef * (self.aN * sin_i * sin_s + self.aT * mu_i * mu_s * cosphi)  # eq 14  **sign
        fvh = coef * (self.aT * mu_s * sinphi)  # eq 14
        fhv = -coef * (self.aT * mu_i * sinphi)  # eq 14   **sign
        fhh = coef * (self.aT * cosphi)  # eq 14
        # the sign of cos_i is different in Karam vs Tsang's because of the convention for thetai

        return fvv, fvh, fhv, fhh

    def ks(self, mu, npol=2):
        k = 7  # number of samples. This should be adaptative depending on the size/wavelength
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

    def phase(self, mu_s, mu_i, dphi, npol=2):
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

    def ke(self, mu, npol=2):
        """return the extinction coefficient."""

        ks_v, ks_h = self.ks(mu)

        return extinction_matrix(ks_v + self.ka, ks_h + self.ka, npol=npol)
