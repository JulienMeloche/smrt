# coding: utf-8

"""Compute scattering using multiple emmodels on a multicomponent microstructure.
The first emmodel for the first microstructure is used to compute the effective permittivity which serves
as the background permittivity for the other microstructures and emmodels. For this reason, the first medium
can be dense, but the others must be sparse. Iterative updating the background permittivity is another possibility but
is not implemented.

'frac_volume' in each layer gives the proportion of the second materials with respect to the background.
For the first layer, it should be the proportion of the material 2 (e.g. ice) w/r to material 1 (e.g. air).
For all the other layers, it should be 1 in general.
'proportion' of each layer gives the volume occupied by each component mixture.

Scattering is computed as the sum of scattering by each component weighted by their fractional volume.
The final effective permittivity is obtained by linear mixing all the components, assuming they do not interact with
each other (=each component is much bigger than the wavelegnth).

All these approximations are strictly valid for sparse media with independent scatterers only (no multiple scattering),
 and should be acceptable when the first component is dense but the other are all sparse.
"""

import copy

# other import
import numpy as np

# local import
from smrt.core.model import make_emmodel

#
# For developers: all emmodel must implement the `effective_permittivity`, `ke` and `phase` functions with the same
# arguments as here initialisation and precomputation can be done in the prepare method that is called only once for
# each layer whereas phase, ke and effective_permittivity can be called several times.
#


class IndependentScattering(object):
    """this class is used to create an electromagnetic model object that dispatch the calculation to a list of emmodels
    and associated micrstructure.

    :param sensor: object containing sensor characteristics
    :param layer: object containing snow layer characteristics (single layer)
    :param emmodel_list: list of emmodel (class or string)
    :param emmodel_options_list: optional list of options for each emmodel (list of dictionary)
    """

    def __init__(self, sensor, layer, emmodel_list, emmodel_options_list=None):
        super().__init__()

        self.layer = layer

        self.component_list = layer.component_list
        self.proportion_list = layer.proportion_list

        assert all(self.layer.thickness == lay.thickness for lay in self.component_list), (
            f"all the layer components must have the same thickness:\n"
            f"{self.layer.thickness} vs {[lay.thickness for lay in self.component_list]}"
        )
        assert all(self.layer.temperature == lay.temperature for lay in self.component_list), (
            f"all the layer components must have the same temperature:\n"
            f"{self.layer.temperature} vs {[lay.temperature for lay in self.component_list]}"
        )

        self.frac_volume = sum(lay.frac_volume for lay in self.component_list)

        assert len(emmodel_list) == len(self.component_list), (
            f"The same number of components and emmodel is expected:\n{len(emmodel_list)} vs {len(self.component_list)}"
        )

        if emmodel_options_list is None:
            emmodel_options_list = [{}] * len(emmodel_list)
        else:
            assert len(emmodel_options_list) == len(emmodel_list), (
                f"The same number of emmodel and emmodel_options is expected:\n{len(emmodel_list)} vs "
                f"{len(self.emmodel_options_list)}"
            )

        # create the first (dense) emmodel
        self.emmodel_list = list()
        self.emmodel_list.append(
            make_emmodel(
                emmodel_list[0],
                sensor,
                self.component_list[0],
                **emmodel_options_list[0],
            )
        )
        self.background_permittivity = self.emmodel_list[0].effective_permittivity()

        # create the other (sparse) emmodel
        for i, lay in enumerate(self.component_list[1:]):
            lay = copy.deepcopy(lay)
            assert lay.frac_volume == 1  # this is likely an error if not
            lay.permittivity_model = (
                self.background_permittivity,
                lay.permittivity_model[1],
            )
            self.emmodel_list.append(make_emmodel(emmodel_list[i + 1], sensor, lay, **emmodel_options_list[i + 1]))

    def basic_check(self):
        # Need to be defined
        pass

    @property
    def ka(self):
        """Calculate the absorption coefficient."""
        return 2 * self.k0 * np.sqrt(self._effective_permittivity).imag

    def phase(self, mu_s, mu_i, dphi, npol=2):
        """Independant scattering phase function."""
        p = self._optimized_compute_average(lambda em: em.phase(mu_s, mu_i, dphi, npol=npol))
        return p

    def ft_even_phase(self, mu_s, mu_i, m_max, npol=None):
        """Calculation of the Fourier decomposed phase function."""
        # this is not necessarly the fastest, if evey ft_even_phase must be calculated
        # for em in self.emmodel_list:
        #    print(type(em))
        #    print(type(em), em.ft_even_phase(mu_s, mu_i, m_max, npol=npol).values.shape)
        p = self._optimized_compute_average(lambda em: em.ft_even_phase(mu_s, mu_i, m_max, npol=npol))
        # if self.proportion_list[1] > 0:
        #     print("p=", p)
        return p

    def ks(self, mu, npol=2):
        """Scattering coefficient matrix

        :param mu: 1-D array of cosines of radiation stream incidence angles
        :param npol: number of polarization
        :returns ke: extinction coefficient matrix [m :sup:`-1`]

        .. note::

            Spherical isotropy assumed (all elements in matrix are identical).

            Size of extinction coefficient matrix depends on number of radiation
            streams, which is set by the radiative transfer solver.

        """
        ks = self._optimized_compute_average(lambda em: em.ks(mu=mu, npol=npol))
        # print(self.proportion_list)
        # print([em.ks(mu, npol=npol) for em in self.emmodel_list])
        # assert(np.all(np.isfinite(ks.values)))
        # if self.proportion_list[1] > 0:
        #     print("ks=", ks)
        return ks

    def ke(self, mu, npol=2):
        """Extinction coefficient matrix

        The extinction coefficient is defined as the sum of scattering and absorption
        coefficients. However, the radiative transfer solver requires this in matrix form,
        so this method is called by the solver.

        :param mu: 1-D array of cosines of radiation stream incidence angles
        :param npol: number of polarization
        :returns ke: extinction coefficient matrix [m :sup:`-1`]

        .. note::

            Spherical isotropy assumed (all elements in matrix are identical).

            Size of extinction coefficient matrix depends on number of radiation
            streams, which is set by the radiative transfer solver.

        """
        ke = self._optimized_compute_average(lambda em: em.ke(mu=mu, npol=npol))
        # if self.proportion_list[1] > 0:
        #     print("ke=", ke)
        # assert(np.all(np.isfinite(ke.values)))
        return ke

    def effective_permittivity(self):
        """Calculation of complex effective permittivity of the medium.

        :returns effective_permittivity: complex effective permittivity of the medium

        """
        return self._optimized_compute_average(lambda em: em.effective_permittivity())

    def _optimized_compute_average(self, func):
        results = [(func(em), weight) for em, weight in zip(self.emmodel_list, self.proportion_list) if weight > 0]

        return np.average([res[0] for res in results], weights=[res[1] for res in results])
