# coding: utf-8

"""
Solve the radiative transfer equation using the first order iteration of the iterative solution to calculate
the backscatter. The solver calculate the zeroth and first order backscatter. This solver is most efficient
than 'dort' but needs to be use with caution. Single scattering albedo should be < 0.3. Muliple scattering are
not taken into account in the first order. The second order solution would be needed when the scattering albedo
and the multiple scattering becomes a factor. The advantage of this solver is the ability to investigate the
different mechanism of backscatter.


Zeroth order should be zero for flat interface and off-nadir. Simply the reduced incident intensity, which
attenuates exponentially inside the medium. Scattering is not included, except for its contribution to
extinction. (Ulaby et al. 2014, first term of 11.74)

First order calculates three contributions. (Ulaby et al. 2014, eqn : 11.75 and 11.62 )

    'direct_backscatter' : Single volume backscatter upwards by the layer.

    'reflection_backscatter' : Single volume backscatter downward by the layer and double specular reflection by the boundary.

    'double_bounce' : 2x Single bistatic scattering by the layer and single reflection by the lower boundary

Usage::
     # Create a model using a nonscattering medium and the rtsolver 'iterative_first'.
     m = make_model("nonscattering", "iterative_first")

Backscatter only!

"""


# Stdlib import

# other import
import numpy as np
import xarray as xr

# local import
from ..core.error import SMRTError, smrt_warn
from ..core.lib import smrt_matrix, is_equal_zero
from ..core.result import make_result
from ..core.fresnel import snell_angle


class IterativeFirst(object):
    """
    Iterative RT solver. Compute the zeroth and first order of the iterative solution.

    :param error_handling: If set to "exception" (the default), raise an exception in cause of error, stopping
                            the code. If set to "nan", return a nan, so the calculation can continue, but the
                            result is of course unusuable and the error message is not accessible. This is only
                            recommended for long simulations that sometimes produce an error.

    :param return_contributions: If set to "False" (the default), only the total backscatter is return. If set
                                to "True", then the four contributions ('direct_backscatter',
                                'reflection_backscatter', 'double_bounce', 'zeroth') described above and the 'total'
                                backscatter are returned.
    """

    # this specifies which dimension this solver is able to deal with.
    #  Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "polarization"}

    def __init__(self, error_handling="exception", return_contributions=False):
        self.error_handling = error_handling
        self.return_contributions = return_contributions

    def solve(self, snowpack, emmodels, sensor, atmosphere=None):
        # """solve the radiative transfer equation for a given snowpack, emmodels and sensor configuration."""
        if sensor.mode != "A":
            raise SMRTError(
                "the iterative_rt solver is only suitable for activate microwave. Use an adequate sensor falling in"
                + "this catergory."
            )

        if atmosphere is not None:
            raise SMRTError(
                "the iterative_rt solver can not handle atmosphere yet. Please put an issue on github if this"
                + "feature is needed."
            )

        thickness = snowpack.layer_thicknesses
        temperature = snowpack.profile("temperature")

        effective_permittivity = [emmodel.effective_permittivity() for emmodel in emmodels]

        substrate = snowpack.substrate
        if substrate is not None and substrate.permittivity(sensor.frequency) is not None:
            substrate_permittivity = substrate.permittivity(sensor.frequency)
            if substrate_permittivity.imag < 1e-8:
                smrt_warn("the permittivity of the substrate has a too small imaginary part for reliable results")
                thickness.append(1e10)
                temperature.append(snowpack.substrate.temperature)
        else:
            substrate = snowpack.substrate
            substrate_permittivity = None

        # Active sensor
        # only V and H are necessary for first order
        self.pola = ["V", "H"]
        self.npol = len(self.pola)
        self.nlayer = snowpack.nlayer
        self.temperature = None

        # effective_permittivity = np.array(effective_permittivity)

        mu0 = np.cos(sensor.theta)
        self.dphi = np.pi

        # solve with first order iterative solution
        I = self.calc_intensity(snowpack, emmodels, sensor, snowpack.interfaces, substrate, effective_permittivity, mu0)

        #  describe the results list of (dimension name, dimension array of value)
        coords = [("theta_inc", sensor.theta_inc_deg), ("polarization_inc", self.pola), ("polarization", self.pola)]

        # store other diagnostic information
        layer_index = "layer", range(self.nlayer)
        other_data = {
            "effective_permittivity": xr.DataArray(effective_permittivity, coords=[layer_index]),
            "ks": xr.DataArray([getattr(em, "ks", np.nan) for em in emmodels], coords=[layer_index]),
            "ka": xr.DataArray([getattr(em, "ka", np.nan) for em in emmodels], coords=[layer_index]),
            "thickness": xr.DataArray(snowpack.layer_thicknesses, coords=[layer_index]),
        }

        # get total intensity from the three contributions
        # first index is the number of mu
        total_I = I[0] + I[1] + I[2] + I[3]

        if self.return_contributions:
            # add total to the intensity array
            intensity = np.array([total_I, I[0], I[1], I[2], I[3]])
            return make_result(
                sensor,
                intensity,
                coords=[
                    (
                        "contribution",
                        ["total", "direct_backscatter", "reflection_backscatter", "double_bounce", "zeroth"],
                    )
                ]
                + coords,
                other_data=other_data,
            )
        else:
            return make_result(sensor, total_I, coords=coords, other_data=other_data)

    def calc_intensity(self, snowpack, emmodels, sensor, interfaces, substrate, effective_permittivity, mu0):
        nlayer = snowpack.nlayer
        dphi = self.dphi
        len_mu = len(mu0)
        npol = self.npol

        # no need for 3x3 in first order solution
        I_i = np.array([[1, 0], [0, 1]]).T

        interface_l = _InterfaceProperties(
            sensor.frequency,
            interfaces,
            substrate,
            effective_permittivity,
            mu0,
            npol,
            nlayer,
            dphi,
        )

        # get list of thickness for each layer
        thickness = snowpack.layer_thicknesses

        # mu for all layer and can have more than 1 if theta from sensor is a list
        mus = interface_l.mu

        # dense snow factor (I think) eq 22a and eq 22b in Tsang et al 2007
        # reshape size to match intensity
        dense_factor_0 = np.atleast_3d((1 / effective_permittivity[0].real) * (mu0 / mus[0])).reshape(len_mu, 1, 1)
        # Intensity incident transmitted to first layer from air
        I_l = get_np_matrix(interface_l.Tbottom_coh[-1], npol, len_mu) @ I_i * dense_factor_0

        # 3 for the number of contribution for the first order backscatter
        intensity_up = np.zeros((4, len_mu, npol, npol))
        optical_depth = 0
        for l in range(nlayer):
            # check scat albedo for validity of iterative solution
            scat_albedo = emmodels[l].ks / (emmodels[l].ks + emmodels[l].ka)
            if scat_albedo > 0.5:
                smrt_warn(
                    f"Warning : scattering albedo ({np.round(scat_albedo,2)}) might be too high"
                    + " for iterative method. Limit is around 0.5."
                )

            # prepare matrix of interface
            # transmission matrix of the top layer to l-1
            Ttop_coh_m = get_np_matrix(interface_l.Ttop_coh[l], npol, len_mu)

            # transmission matrix of the bottom layer to l+1
            Tbottom_coh_m = get_np_matrix(interface_l.Tbottom_coh[l], npol, len_mu)

            # Specular Reflection matrix of the bottom layer
            Rbottom_coh_m = get_np_matrix(interface_l.Rbottom_coh[l], npol, len_mu)

            # Diffuse reflection matrix of the bottom layer
            Rbottom_diff_m = get_np_matrix(interface_l.Rbottom_diff[l], npol, len_mu)

            # get phase function for array of mu and -mu
            mus_sym = np.concatenate([-mus[l], mus[l]])
            phases = emmodels[l].phase(mus_sym, mus_sym, dphi, npol).values.squeeze()

            # 1/4pi normalization of the RT equation for SMRT
            # applied to phase here, interface and subsrate already have the smrt_norm
            phases = phases / (4 * np.pi)

            P_Up = np.array([phases[:, :, i, i + len_mu] for i in range(len_mu)])  # P(-mu, mu)
            P_Down = np.array([phases[:, :, i + len_mu, i] for i in range(len_mu)])  # P(mu, -mu)
            P_Bi_Up = np.array([phases[:, :, i + len_mu, i + len_mu] for i in range(len_mu)])  # P(mu, mu)
            P_Bi_Down = np.array([phases[:, :, i, i] for i in range(len_mu)])  # P(-mu, -mu)

            ke = emmodels[l].ks + emmodels[l].ka
            layer_optical_depth = ke * thickness[l]
            optical_depth += layer_optical_depth

            # convert to 3d array for computation of intensity
            # allow computation of incident angle
            # two way attenuation (ulaby et al 2014, eq: 11.2)
            gammas2 = np.atleast_3d(np.exp(-2 * layer_optical_depth / mus[l])).reshape(len_mu, 1, 1)
            mu_3d = np.atleast_3d(mus[l]).reshape(len(mu0), 1, 1)

            """
            Zeroth order,  ulaby et al 2014 (first term of 11.74) should be zero for flat interface and off-nadir. 
            Simply the reduced incident intensity, which attenuates exponentially inside the medium.
            Scattering is not included, except for its contribution to extinction
            """
            I0_mu = Ttop_coh_m @ (gammas2 * (Rbottom_diff_m @ I_l))

            """
            First order, ulaby et al 2014 (11.75 and 11.62 )
            Four contributions are taken into account
            - Single volume backscatter upwards by the layer (direct backscatter)
            - 2x single bistatic scattering by the layer and single reflection by the lower boundary. (double bounce)
            - Single volume backscatter downward by the layer and double specular reflection by the boundary (reflected backscatter)

            """

            I1_back = Ttop_coh_m @ ((1 - gammas2) / (2 * ke) * P_Up) @ I_l

            I1_ref_back = (
                Ttop_coh_m @ (((1 - gammas2) / (2 * ke) * gammas2) * (Rbottom_coh_m @ P_Down @ Rbottom_coh_m)) @ I_l
            )

            I1_2B = (
                Ttop_coh_m
                @ (thickness[l] * gammas2 / mu_3d * (P_Bi_Down @ Rbottom_coh_m + Rbottom_coh_m @ P_Bi_Up))
                @ I_l
            )

            # shape of intensity (incident angle, first order contribution, npo, npol)
            I1 = np.array([I1_back, I1_ref_back, I1_2B, I0_mu]).reshape(4, len_mu, npol, npol)

            # add intensity
            intensity_up += I1

            if l < nlayer - 1:
                # dense snow factor (I think) eq 22a and eq 22b in Tsang et al 2007
                dense_factor_l = np.atleast_3d(effective_permittivity[l].real / effective_permittivity[l + 1].real) * (
                    mus[l] / mus[l + 1]
                ).reshape(len_mu, 1, 1)
                # intensity in the layer transmitted downward for upper layer with one way attenuation
                # one way attenuation??? sqrt of gamma2?
                I_l = Tbottom_coh_m @ (gammas2 * dense_factor_l * I_l)

        if substrate is None and optical_depth < 5:
            smrt_warn(
                "The solver has detected that the snowpack is optically shallow (tau=%g) and no substrate has been set, meaning that the space "
                "under the snowpack is vaccum and that the snowpack is shallow enough to affect the signal measured at the surface."
                "This is usually not wanted. Either increase the thickness of the snowpack or set a substrate."
                " If wanted, add a transparent substrate to supress this warning" % optical_depth
            )

        # snow to air final transmission upward
        intensity = get_np_matrix(interface_l.Ttop_coh[0], npol, len_mu) @ intensity_up

        # 1/4pi normalization of the RT equation like DORT
        return intensity


def get_np_matrix(smrt_m, npol, n_mu):
    # input are smrt matrix, out numpy matrix
    if is_equal_zero(smrt_m):
        np_m = np.zeros((n_mu, npol, npol))
        return np_m

    if smrt_m.mtype.startswith("diagonal"):
        np_m = np.zeros((n_mu, npol, npol))
        for i in range(n_mu):
            np.fill_diagonal(np_m[i], smrt_m.diagonal[:, i])
        return np_m

    else:
        raise NotImplementedError
        # print('dense')
        # return smrt_m.values.squeeze()


#### Potential bug!
# added z in the name of the class InterfaceProperties
# issue with import class in plugin.py line 75
# get a list of sorted classs by name not order of appearance. Needs to be discuss with Ghi.
# So far, not in issu with dort or other module.
# catch by test_normal_call()


class _InterfaceProperties(object):
    # prepare interface properties of multi-layer snowpack
    # top and bottom interface of layer l, index -1 refers to air layer

    def __init__(self, frequency, interfaces, substrate, permittivity, mu0, npol, nlayer, dphi):
        permittivity = permittivity

        self.Rtop_coh = dict()
        self.Rtop_diff = dict()
        self.Ttop_coh = dict()
        # self.Ttop_diff = dict()
        self.Rbottom_coh = dict()
        self.Rbottom_diff = dict()
        self.Tbottom_coh = dict()
        # self.Tbottom_diff = dict()

        self.mu = dict()
        self.mu[-1] = mu0
        # air-snow DOWN
        # index -1 refers to air layer
        self.Tbottom_coh[-1] = interfaces[0].coherent_transmission_matrix(frequency, 1, permittivity[0], mu0, npol)

        # air-snow DOWN
        self.Rbottom_coh[-1] = interfaces[0].specular_reflection_matrix(frequency, 1, permittivity[0], mu0, npol)
        self.Rbottom_diff[-1] = (
            interfaces[0].diffuse_reflection_matrix(frequency, 1, permittivity[0], mu0, mu0, dphi, npol)
            if hasattr(interfaces[0], "diffuse_reflection_matrix")
            else smrt_matrix(0)
        )

        for l in range(nlayer):
            # define permittivity
            # #for permittivity, index 0 = air, length of permittivity is l+1
            eps_lm1 = permittivity[l - 1] if l > 0 else 1
            eps_l = permittivity[l]
            if l < nlayer - 1:
                eps_lp1 = permittivity[l + 1]
            else:
                eps_lp1 = None

            # going in the medium of layer l
            # calcule angle of medium l
            self.mu[l] = snell_angle(eps_lm1, eps_l, mu0)

            self.Rtop_coh[l] = interfaces[l].specular_reflection_matrix(frequency, eps_l, eps_lm1, self.mu[l], npol)

            self.Rtop_diff[l] = (
                interfaces[l].diffuse_reflection_matrix(frequency, eps_l, eps_lm1, self.mu[l], self.mu[l], dphi, npol)
                if hasattr(interfaces[l], "diffuse_reflection_matrix")
                else smrt_matrix(0)
            )

            self.Ttop_coh[l] = interfaces[l].coherent_transmission_matrix(frequency, eps_l, eps_lm1, self.mu[l], npol)

            if l < nlayer - 1:
                # set up interfaces
                # snow - snow
                # Upward
                self.Rbottom_coh[l] = interfaces[l + 1].specular_reflection_matrix(
                    frequency, eps_l, eps_lp1, self.mu[l], npol
                )

                # other than flat interface
                self.Rbottom_diff[l] = interfaces[l + 1].diffuse_reflection_matrix(
                    frequency, eps_l, eps_lp1, self.mu[l], self.mu[l], dphi, npol
                )

                self.Tbottom_coh[l] = interfaces[l + 1].coherent_transmission_matrix(
                    frequency, eps_l, eps_lp1, self.mu[l], npol
                )

            elif substrate is not None:
                self.Rbottom_coh[l] = substrate.specular_reflection_matrix(frequency, eps_l, self.mu[l], npol)

                self.Rbottom_diff[l] = (
                    substrate.diffuse_reflection_matrix(frequency, eps_l, self.mu[l], self.mu[l], dphi, npol)
                    if hasattr(substrate, "diffuse_reflection_matrix")
                    else smrt_matrix(0)
                )

                # sub-snow
                self.Tbottom_coh[l] = smrt_matrix(0)

            else:
                # fully transparent substrate
                self.Rbottom_coh[l] = smrt_matrix(0)
                self.Rbottom_diff[l] = smrt_matrix(0)
                self.Tbottom_coh[l] = smrt_matrix(0)
