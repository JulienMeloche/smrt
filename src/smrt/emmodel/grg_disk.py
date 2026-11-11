"""
Generalized Rayleigh-Gans approximation for horizontal disks, based on Karam et al. 1988.
The height T must be small with respect to the wavelength: k T sqrt(eps) << 1.

References:
    - M. A. Karam, A. K. Fung and Y. M. M. Antar, "Electromagnetic wave scattering from some vegetation samples,"
      in IEEE Transactions on Geoscience and Remote Sensing, vol. 26, no. 6, pp. 799-808, Nov. 1988,
      doi: 10.1109/36.7711.

For a less restrictive formulation, see Le Vine et al. 1983, 1985 or Tsang's book I, page 51.
"""

import numpy as np
import scipy.special

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.emmodel.grg_common import GeneralizedRayleighGansBase


class GeneralizedRayleighGansDisk(GeneralizedRayleighGansBase):
    """ """

    def demagnetizing_factors(self):
        m = self.layer.microstructure.radius / self.layer.microstructure.height * 2
        m2 = m**2
        n = np.sqrt(m2 - 1)
        arcsin = np.arcsin(n / m)

        gT = (m2 / n * arcsin - 1) / (2 * (m2 - 1))  # Eq 15
        # gN = (1 - arcsin / n) * m2 / (m2 - 1)
        gN = (
            1 - 2 * gT
        )  # see Eq 2.7 in J. A. Osborn, Demagnetizing Factors of the General Ellipsoid, Phys. Rev. 67, 351, 1945

        return gT, gN

    def basic_check(self):
        # if we want to be strict, only IndependentDisk should be valid, but in practice any
        # model with a radius and height can make it!
        if not hasattr(self.layer.microstructure, "radius") or not hasattr(self.layer.microstructure, "height"):
            raise SMRTError(
                "Only microstructure_model which defined a `radius` and a `height`can be used for disk scattering"
            )

        # quite laxist limit? eps is the scatterer dielectric constant because k0 is in vacuum
        if np.any(self.layer.microstructure.height * self.k0 * np.sqrt(self.eps).real > 0.8):
            smrt_warn("The Generalized Rayleigh-Gans approximation requires thin disks w/r to the wavelength.")

        if np.any(self.layer.microstructure.height / (2 * self.layer.microstructure.radius) > 0.1):
            smrt_warn("The Generalized Rayleigh-Gans approximation requires thin disks w/r to their diameter.")

    def modifying_function(self, mu_s, sin_s, mu_i, sin_i, cosphi, sinphi):
        """compute the modifying function for disk Eq 18 using Karam et al. 1988"""

        si2 = sin_s**2 + sin_i**2 - 2 * sin_s * sin_i * cosphi
        si2[si2 < 0] = 0
        qsi = self.k0 * np.sqrt(self.e0.real * si2)  # Eq 19
        # .real is used because we assume the absorption is small w/r to the size of the scatterer
        assert np.all(np.isfinite(qsi))

        qsi_a = qsi * self.layer.microstructure.radius
        modifying_function = 2 * scipy.special.j1(qsi_a) / qsi_a  # Eq 18
        modifying_function[qsi_a < 1e-6] = 1  # this occurs in the forward direction

        return modifying_function
