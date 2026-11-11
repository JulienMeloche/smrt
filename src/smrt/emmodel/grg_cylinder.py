"""
Provide Generalized Rayleigh-Gans approximation for horizontal thin cylinder / needles, based on Karam et al. 1988.
The radius a must be small with respect to the wavelength: k a sqrt(eps) << 1.

References:
    - M. A. Karam, A. K. Fung and Y. M. M. Antar, "Electromagnetic wave scattering from some vegetation samples,"
in IEEE Transactions on Geoscience and Remote Sensing, vol. 26, no. 6, pp. 799-808, Nov. 1988, doi: 10.1109/36.7711.

For a less restrictive formulation, see :py:mod:~smrt.emmodels.`infinite_cylinder` (also in Karam et al. 1988)
"""

import numpy as np

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.emmodel.grg_common import GeneralizedRayleighGansBase


class GeneralizedRayleighGansCylinder(GeneralizedRayleighGansBase):
    """ """

    def demagnetizing_factors(self):
        # Karam 1988 equations diverge from the expressions found in other references when running numerical
        # simulations. We use instead equations 2.10 and 2.11 in J. A. Osborn, Demagnetizing Factors of the General
        # Ellipsoid, Phys. Rev. 67, 351, 1945.
        # Derivation of the equations found in Tsang et al., 2000, p13 give the same expressions as Osborn 1945.

        # Eq 20 in Karam 1988, considering h = height / 2
        b2 = 1 - (self.layer.microstructure.radius / self.layer.microstructure.height * 2) ** 2

        # b = np.sqrt(b2)
        # log_term = 0.5 * np.log((1 - b) / (b + 1))  # erreur in eq 20 b -1 => 1-b
        # gT = 0.5 * b * (b2 - 1) * (b / (b2 - 1) + log_term)  # Eq 20 in Karam 1988
        # gN = -(b2 - 1) * (b * log_term + 1)

        # Equations in Osborn 1945
        m2 = 1 / (1 - b2)
        m = np.sqrt(m2)

        log_term = np.log((m + np.sqrt(m2 - 1)) / (m - np.sqrt(m2 - 1)))  # Eq 2.10

        gT = m * 1 / (2 * (m2 - 1)) * (m - 1 / (2 * np.sqrt(m2 - 1)) * log_term)  # Eq 2.10

        gN = 1 - 2 * gT  # see Eq 2.7 in Osborn 1945

        return gT, gN

    def basic_check(self):
        # if we want to be strict, only IndependentDisk should be valid, but in practice any
        # model with a radius and height can make it!
        if not hasattr(self.layer.microstructure, "radius") or not hasattr(self.layer.microstructure, "height"):
            raise SMRTError(
                "Only microstructure_model which defined a `radius` and a `height`can be used for cylinder scattering"
            )

        if np.any(
            2 * self.layer.microstructure.radius * self.k0 * np.sqrt(self.eps).real > 0.8
        ):  # quite laxist limit? eps is the scatterer dielectric constant because k0 is in vacuum
            smrt_warn("The Generalized Rayleigh-Gans approximation requires thin cylinders w/r to the wavelength")

        if np.any(2 * self.layer.microstructure.radius / self.layer.microstructure.height > 0.1):
            smrt_warn("The Generalized Rayleigh-Gans approximation requires thin cylinders w/r to their height.")

    def modifying_function(self, mu_s, sin_s, mu_i, sin_i, cosphi, sinphi):
        """compute the modifying function for disk Eq 21 using Karam et al. 1988"""

        return np.sinc(self.k0 * self.layer.microstructure.height / np.pi * (mu_i + mu_s))
