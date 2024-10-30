# coding: utf-8

"""
Solve the radiative transfer equation using the first order and the second order iteration of the iterative solution to calculate
the backscatter. The solver calculate the zeroth, first and second order backscatter. This solver is most efficient
than 'dort' but needs to be use with caution. Single scattering albedo should be < 0.5. Muliple scattering (only double scattering)
taken into account in the first order. The advantage of this solver is the ability to investigate the
different mechanism of backscatter.

The only component of second order is:

    'double_scattering' : 2x bistatic scattering by the layer

This component is added to all component of the first order

Usage::
     # Create a model using a nonscattering medium and the rtsolver 'iterative_second'.
     m = make_model("nonscattering", "iterative_second")

Backscatter only!

"""


# Stdlib import

# other import
import numpy as np
import xarray as xr

# local import
from smrt.core.error import SMRTError, smrt_warn
from smrt.core.result import make_result
from smrt.rtsolver.dort import compute_stream
from smrt.rtsolver.iterative_first import IterativeFirst, get_np_matrix, _InterfaceProperties


class IterativeSecond(object):
    """
    Iterative RT solver. Compute the second order of the iterative solution.

    :param error_handling: If set to "exception" (the default), raise an exception in cause of error, stopping
                            the code. If set to "nan", return a nan, so the calculation can continue, but the
                            result is of course unusuable and the error message is not accessible. This is only
                            recommended for long simulations that sometimes produce an error.

    :param return_contributions: If set to "False" (the default), only the total backscatter is return. If set
                                to "True", then the five contributions ('direct_backscatter',
                                'reflection_backscatter', 'double_bounce', 'zeroth', 'double_scattering') described
                                above and the 'total' backscatter are returned.


    """

    # this specifies which dimension this solver is able to deal with.
    #  Those not in this list must be managed by the called (Model object)
    # e.g. here, frequency, time, ... are not managed
    _broadcast_capability = {"theta_inc", "polarization_inc", "theta", "polarization"}

    def __init__(
        self,
        error_handling="exception",
        return_contributions=False,
        n_max_stream=32,  # stream for integral of theta 0 to pi/2
        stream_mode="most_refringent",
        m_max=5,  # mode for integral of phi from 0 to 2pi
    ):
        self.error_handling = error_handling
        self.return_contributions = return_contributions
        self.n_max_stream = n_max_stream
        self.stream_mode = stream_mode
        self.m_max = m_max

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
        # only returns V and H but U is used in the calculation. U is removed at the end to match first order
        self.pola = ["V", "H"]
        # but U is used in the calculation, so npol =3
        self.npol = 3
        self.nlayer = snowpack.nlayer
        temperature = None
        mu0 = np.cos(sensor.theta)
        self.len_mu = len(mu0)
        self.dphi = np.pi

        # create the model
        self.emmodels = emmodels
        # interface
        self.interfaces = snowpack.interfaces
        # sensor
        self.sensor = sensor

        # get stream for integral of theta for 0 to pi/2
        streams = compute_stream(
            self.n_max_stream, effective_permittivity, substrate_permittivity, mode=self.stream_mode
        )

        # get the first order
        first_solver = IterativeFirst(return_contributions=True)
        # 2 pol for first order
        first_solver.npol = 2
        first_solver.dphi = self.dphi
        I1 = first_solver.calc_intensity(
            snowpack, emmodels, sensor, snowpack.interfaces, substrate, effective_permittivity, mu0
        )
        total_I1 = I1[0] + I1[1] + I1[2] + I1[3]

        # solve the second order iterative solution
        I2 = self.calc_intensity(snowpack, emmodels, streams, sensor, effective_permittivity, mu0)

        # add first and second, get rid of U pol for second order so it match first order
        total_I = total_I1 + I2[:, 0:2, 0:2]
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

        if self.return_contributions:
            # add total to the intensity array
            intensity = np.array([total_I, I1[0], I1[1], I1[2], I1[3], I2[:, 0:2, 0:2]])
            return make_result(
                sensor,
                intensity,
                coords=[
                    (
                        "contribution",
                        [
                            "total",
                            "direct_backscatter",
                            "reflection_backscatter",
                            "double_bounce",
                            "zeroth",
                            "double_scattering",
                        ],
                    )
                ]
                + coords,
                other_data=other_data,
            )
        else:
            return make_result(sensor, total_I, coords=coords, other_data=other_data)

    def calc_intensity(self, snowpack, emmodels, streams, sensor, effective_permittivity, mu0):
        # mode for integral of phi from 0 to 2pi
        npol = self.npol
        dphi = self.dphi
        len_mu = self.len_mu
        nlayer = snowpack.nlayer
        thickness = snowpack.layer_thicknesses

        interface_l = _InterfaceProperties(
            sensor.frequency, snowpack.interfaces, snowpack.substrate, effective_permittivity, mu0, npol, nlayer, dphi
        )

        I_i = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]).T

        # intensity in the layer
        # dense snow factor (I think) eq 22a and eq 22b in Tsang et al 2007
        dense_factor_0 = np.atleast_3d(1 / effective_permittivity[0].real) * (mu0 / interface_l.mu[0]).reshape(
            len_mu, 1, 1
        )
        I_l = get_np_matrix(interface_l.Tbottom_coh[-1], npol, len_mu) @ I_i * dense_factor_0

        # needs for loop for multiple layers
        optical_depth = 0
        intensity_up = np.zeros((len(mu0), npol, npol))
        for l in range(nlayer):
            # incident angle in layer
            mu_i = interface_l.mu[l]

            # prepare matrix of interface
            # transmission matrix of the top layer to l-1
            Ttop_coh_m = get_np_matrix(interface_l.Ttop_coh[l], npol, len_mu)

            # transmission matrix of the bottom layer to l+1
            Tbottom_coh_m = get_np_matrix(interface_l.Tbottom_coh[l], npol, len_mu)

            # stream for integral
            mus_int = streams.mu[l][::-1]
            weight = streams.weight[l][::-1]

            # extinction coef and layer optical depth
            ke = emmodels[l].ks + emmodels[l].ka
            layer_optical_depth = ke * thickness[l]
            optical_depth += layer_optical_depth

            # ##create the integral object
            # int_obj = Integral(I_l, ke, layer_optical_depth, emmodels[l], m_max, npol, dphi)

            # get intensity of double scatter

            intensity_up += Ttop_coh_m @ self.compute_double_scattering(
                emmodels[l], I_l, weight, mus_int, mu_i, ke, layer_optical_depth
            )

            # intensity transmitted down to next layer
            gamma2 = compute_gamma(layer_optical_depth, mu_i) ** 2

            if l < nlayer - 1:
                # dense snow factor (I think) eq 22a and eq 22b in Tsang et al 2007
                dense_factor_l = np.atleast_3d(effective_permittivity[l].real / effective_permittivity[l + 1].real) * (
                    mu_i / interface_l.mu[l + 1]
                ).reshape(len_mu, 1, 1)
                # intensity in the layer transmitted downward for upper layer with one way attenuation
                # one way attenuation??? sqrt of gamma2?

                # I_l = np.matmul(Tbottom_coh_m, (I_l * gamma2)) * dense_factor_l
                I_l = Tbottom_coh_m @ (I_l * gamma2) * dense_factor_l

        if snowpack.substrate is None and optical_depth < 5:
            smrt_warn(
                "The solver has detected that the snowpack is optically shallow (tau=%g) and no substrate has been set, meaning that the space "
                "under the snowpack is vaccum and that the snowpack is shallow enough to affect the signal measured at the surface."
                "This is usually not wanted. Either increase the thickness of the snowpack or set a substrate."
                " If wanted, add a transparent substrate to supress this warning" % optical_depth
            )

        return get_np_matrix(interface_l.Ttop_coh[0], npol, len_mu) @ intensity_up

    def decompose_ft_phase(self, ft_p):
        mode = np.arange(0, self.m_max)
        n_mu = np.arange(0, self.len_mu)
        # cosines
        ft_p_c = np.array(
            [
                [
                    [
                        [ft_p[0, 0, m, n], ft_p[0, 1, m, n], 0],
                        [ft_p[1, 0, m, n], ft_p[1, 1, m, n], 0],
                        [0, 0, ft_p[2, 2, m, n]],
                    ]
                    for m in mode
                ]
                for n in n_mu
            ]
        )

        # sines
        ft_p_s = np.array(
            [
                [
                    [
                        [0, 0, -ft_p[0, 2, m, n]],
                        [0, 0, -ft_p[1, 2, m, n]], 
                        [ft_p[2, 0, m, n], ft_p[2, 1, m, n], 0]
                    ]
                    for m in mode
                ]
                for n in n_mu
            ]
        )

        # sine = 0 for mode 0
        ft_p_s[:, 0, ] = np.zeros((self.len_mu, self.npol, self.npol))

        return ft_p_c, ft_p_s

    def compute_A(self, mu_i, mu_int, ke, layer_optical_depth):
        # function of mu and mu'

        gamma_i = np.atleast_3d(compute_gamma(mu_i, layer_optical_depth)).reshape(self.len_mu, 1, 1)
        gamma2_i = gamma_i**2
        mu_i = np.atleast_3d(mu_i).reshape(self.len_mu, 1, 1)

        A = (1 - gamma2_i) / (2 * ke) - (
            mu_int
            * mu_i
            * gamma_i
            * (compute_gamma(mu_int, layer_optical_depth) - gamma_i)
            / (ke**2 * (mu_int - mu_i) * (mu_int + mu_i))
        )
        return A

    def compute_B(self, mu_i, mu_int, ke, layer_optical_depth):
        gamma_i = np.atleast_3d(compute_gamma(mu_i, layer_optical_depth)).reshape(self.len_mu, 1, 1)
        mu_i = np.atleast_3d(mu_i).reshape(self.len_mu, 1, 1)

        B = (
            mu_i**2
            / (ke**2 * 2 * mu_i * (mu_int + mu_i))
            * (
                mu_i
                + gamma_i
                / (ke * (mu_int - mu_i))
                * (
                    ke * mu_i * mu_int * (gamma_i - compute_gamma(mu_int, layer_optical_depth))
                    + ke * gamma_i * (mu_i**2 - mu_int * mu_i)
                )
            )
        )
        return B

    def compute_int_phi(self, mat1, mat2):
        # approximation of the integral of phi
        # compute the summation of all the mode between two decomposed matrix
        # integral of phi (0, 2pi) using fourier decomposition Appendix 2 tsang et al 2007

        m1_c, m1_s = self.decompose_ft_phase(mat1)
        m2_c, m2_s = self.decompose_ft_phase(mat2)

        # error in fourier expansion of tsang? should be mode 0 for second matrix?
        # reshape to match shape of decompose array (len_mu, npol, npol)
        m1_0 = np.array([mat1[:, :, 0, i] for i in range(self.len_mu)])
        m2_0 = np.array([mat2[:, :, 0, i] for i in range(self.len_mu)])

        int_0 = 2 * np.pi * (m1_0 @ m2_0)
        sum_mc = 0
        # sum_ms = 0
        # summation of m=1 to m_max, skip 0
        for m in range(self.m_max)[1:]:
            sum_mc += (m1_c[:, m] @ m2_c[:, m] - m1_s[:, m] @ m2_s[:, m]) * np.cos(m * self.dphi)
            # sum_ms += (m1_c[:,m] @ m2_s[:,m]  + m1_s[:,m] @ m2_c[:,m]) * np.sin(m * self.dphi) # equal to 0

        int_phi = int_0 + np.pi * sum_mc  # + np.pi * sum_ms

        return int_phi

    def compute_double_scattering(self, emmodel, I_l, weight, mu_int, mus_i, ke, layer_optical_depth):
        # double scattering equation is from tsang et al. 2000, Scattering of Electromagnetic Waves: Theories and Applications
        # p: 309-310, eq: 8.1.81, the the double scattering is the last component (e) of the equation.
        # A and B are functions of theta, define explicitely in the book
        # integral of phi and theta
        # Compute summation of the Gauss quadrature (integral of theta) G.Picard thesis eq. 2.22
        # integral of theta from 0 to 1, for mu with change of variable
        # need weight and mu_int from streams

        # multiple incident angle

        # get negative and positive mu
        mu_i_sym = np.concatenate([-mus_i, mus_i])
        mus_int = np.concatenate([-mu_int, mu_int])

        n_stream = len(mu_int)
        n_mu_i = len(mus_i)

        phase_mu_int_mu = emmodel.ft_even_phase(mus_int, mu_i_sym, self.m_max) / (4 * np.pi)
        phase_mu_mu_int = emmodel.ft_even_phase(mu_i_sym, mus_int, self.m_max) / (4 * np.pi)

        P1 = phase_mu_mu_int[:, :, :, n_mu_i:, n_stream:]  # P(mu_i, mu_int)
        P2 = phase_mu_int_mu[:, :, :, n_stream:, 0:n_mu_i]  # P(mu_int, -mu_i)
        P3 = phase_mu_mu_int[:, :, :, n_mu_i:, 0:n_stream]  # P(mu_i, -mu_int)
        P4 = phase_mu_int_mu[:, :, :, n_stream:, n_mu_i:]  # P(mu_int, mu_i)

        P5 = phase_mu_mu_int[:, :, :, n_mu_i:, 0:n_stream]  # P(mu_i, -mu_int)
        P6 = phase_mu_int_mu[:, :, :, 0:n_stream, 0:n_mu_i]  # P(-mu_int, -mu_i)
        P7 = phase_mu_mu_int[:, :, :, n_mu_i:, n_stream:]  # P(mu_i, mu_int)
        P8 = phase_mu_int_mu[:, :, :, n_stream:, n_mu_i:]  # P(mu_int, mu_i)

        sum_a, sum_b = 0, 0
        for mu, w, i in zip(mu_int, weight, range(n_stream)):
            # bound 1 of the integral
            # integral of mu (G.Picard thesis p.72)
            # -1 coef for incident angle
            # P(mu_i, mu_int)* P(mu_int, -mu_i) + P(mu_i, -mu_int)* P(mu_int, mu_i)

            sum_a += w * (
                (
                    self.compute_A(mus_i, mu, ke, layer_optical_depth)
                    * self.compute_int_phi(P1[:, :, :, :, i], P2[:, :, :, i, :])
                )
                + (
                    self.compute_A(mus_i, mu, ke, layer_optical_depth)
                    * self.compute_int_phi(P3[:, :, :, :, i], P4[:, :, :, i, :])
                )
            )

            # P(mu_i, -mu)* P(-mu_int, -mu_i) + P(mu_i, mu_int)* P(mu_int, mu_i)
            sum_b += w * (
                (
                    self.compute_B(mus_i, mu, ke, layer_optical_depth)
                    * self.compute_int_phi(P5[:, :, :, :, i], P6[:, :, :, i, :])
                )
                + (
                    self.compute_B(mus_i, mu, ke, layer_optical_depth)
                    * self.compute_int_phi(P7[:, :, :, :, i], P8[:, :, :, i, :])
                )
            )

        I_mu = (sum_a + sum_b) @ I_l

        return I_mu


# one way attenuation (ulaby et al 2014, eq: 11.2)
def compute_gamma(mu, layer_optical_depth):
    return np.exp(-1 * layer_optical_depth / mu)


# def get_np_matrix_stream(smrt_m, npol, n_max_stream):
#     # input are smrt matrix, out numpy matrix
#     if is_equal_zero(smrt_m):
#         np_m = np.zeros((n_max_stream, npol, npol))
#         return np_m

#     if smrt_m.mtype.startswith("diagonal"):
#         np_m = np.zeros((n_max_stream, npol, npol))
#         [np.fill_diagonal(np_m[i,:,:], smrt_m.values[:,i]) for i in range(n_max_stream)]
#         return np_m

#     else:
#         return smrt_m.values.squeeze()
