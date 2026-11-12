"""Provide thin vertical cylinders following Saranbandi and Senior, 1990.

References:
    - K. Sarabandi and T. B. A. Senior, "Low-frequency scattering from cylindrical structures at oblique incidence,"
      in IEEE Transactions on Geoscience and Remote Sensing, vol. 28, no. 5, pp. 879-885, Sept. 1990,
      doi: 10.1109/36.58977

This theory requires the cylinders to be thin with respect to the wavelength and to be either very high or very short
with respect to the wavelength.

We also assumes that the medium is sparsely populated, i.e. the cylinders occupies a small fraction of the volume.

This model is only compatible with the Independent Cylinder microstructure model

"""

import numpy as np

from smrt.core.error import SMRTError
from smrt.core.globalconstants import C_SPEED
from smrt.core.lib import GenericFTPhaseMixin, phase_matrix_from_scattering_amplitude, vectorize_angles
from smrt.core.vector3 import vector3


class ThinVerticalCylinder(GenericFTPhaseMixin):
    """ """

    def __init__(self, sensor, layer, *, precompute=True):
        super().__init__()
        self.sensor = sensor
        self.layer = layer

        if precompute:
            self.precompute()

    def precompute(self):
        sensor = self.sensor
        layer = self.layer

        # check here the limit of the Rayleigh model

        f = layer.frac_volume

        e0 = layer.permittivity(0, sensor.frequency)  # background permittivity
        eps = layer.permittivity(1, sensor.frequency)  # scatterer permittivity

        self.effective_permittivity = e0  # sparse medium only

        self.radius = layer.microstructure.radius
        self.height = layer.microstructure.height

        self.k0 = C_SPEED / sensor.wavenumber
        # compute absorption (from Rayleigh scattering). It is unlikely that this approximation is good...
        # but if the medium is sparse, this contribution should be small anyway (????)
        self.ka = f * self.k0 * eps.imag * 9 * abs(e0 / (eps + 2 * e0)) ** 2 + (1 - f) * e0.imag * self.k0

        # polarizability
        self.Pxx = 2 * (eps - e0) / (eps + e0)  # Eq 25
        self.Pzz = eps - e0  # Eq 26

    def basic_check(self):
        # TODO Ghi: check the microstructure model is compatible.
        # if we want to be strict, only IndependentShpere should be valid, but in practice any
        # model of sphere with a radius can make it!
        if not (hasattr(self.layer.microstructure, "radius") and hasattr(self.layer.microstructure, "height")):
            raise SMRTError(
                "Only microstructure_model which defined a `radius` and a `height` can be used with ThinCylinder"
                " scattering"
            )

    def phase(self, mu_s, mu_i, phi_s, phi_i=0, npol=2):
        # sin of the incident and scattered directions
        mu_s, sin_s, mu_i, sin_i, cosphi, sinphi = vectorize_angles(mu_s, mu_i, phi_s - phi_i)

        # compute scattering direction vector
        ks = vector3.from_xyz(sin_s * cosphi, sin_s * sinphi, mu_s)

        # a is the direction of the incidence E field and we must calculate the tensor product P.a where
        # the polarizability matrix is diagonal with elements [Pxx, Pyy, Pzz] and Pyy=Pxx for a cylinder
        # compute P.a for incident V and H polarizations
        # we use:  hi = vector3.from_xyz(0, 1, 0) and vi = vector3.from_xyz(-mu_i, 0, -sin_i)
        Pa_H = vector3.from_xyz(0, self.Pxx, 0)
        Pa_V = vector3.from_xyz(-self.Pxx * mu_i, 0, -self.Pzz * sin_i)

        # coefficient to account for the finite heigth of the cylinder
        U = self.k0 * self.height / (2 * np.pi) * (mu_s - mu_i)  # eq 33. The pi comes from np.sinc definition

        # scattering amplitude matrix in the x,y coordinates
        coef = -(self.k0**2) / 4 * np.sinc(U)
        S_H = coef * ks.cross(ks.cross(Pa_H))
        S_V = coef * ks.cross(ks.cross(Pa_V))

        # prepare projection onto H and V polarization in the scattering direction
        # horizontal and vertical polarizations in the scattering direction
        hs = vector3.from_xyz(-sinphi, cosphi, 0)
        vs = vector3.from_xyz(mu_s * cosphi, mu_s * sinphi, -sin_s)

        return phase_matrix_from_scattering_amplitude(
            fvv=vs.dot(S_V), fvh=vs.dot(S_H), fhv=hs.dot(S_V), fhh=hs.dot(S_H), npol=npol
        )
