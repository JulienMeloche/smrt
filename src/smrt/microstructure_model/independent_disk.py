"""Provide the microstructure for independent disks. "Independent" means that the disks are
distributed completely randomly in the medium, even overlapping each other. This is why this representation only
applies to low fractional volume.

"""

from smrt.microstructure_model.autocorrelation import Autocorrelation


class IndependentDisk(Autocorrelation):
    args = ("frac_volume", "radius", "height")
    optional_args = {
        "radius_range": 0,
        "radius_sampling": 9,
    }  # scattering is sampled between radius - radius_range / 2 and radius + radius_range / 2

    def __init__(self, params):
        super().__init__(params)  # don't forget this line in our classes!

    @property
    def corr_func_at_origin(self):
        # value of the correlation function at the origin
        return self.frac_volume * (1.0 - self.frac_volume)

    @property
    def inv_slope_at_origin(self):
        # inverse slope of the normalized correlation function at the origin
        raise NotImplementedError()
        # return 4.0 / 3 * self.radius

    def basic_check(self):
        """check consistency between the parameters"""
        pass

    def compute_ssa(self):
        """Compute the ssa for a sphere"""
        # implement here the calculation of the SSA from the microstructure parameters.
        raise NotImplementedError()
        # return 3.0 / (DENSITY_OF_ICE * self.radius)

    def autocorrelation_function(self, r):
        """compute the real space autocorrelation function for an independent sphere"""
        # spherical correlation function

        raise NotImplementedError()

    def ft_autocorrelation_function(self, k):
        """Compute the 3D Fourier transform of the isotropic correlation
        function for an independent sphere for given magnitude k of the 3D wave vector
        (float).

        """
        raise NotImplementedError()
