

"""
Implement the interface boundary condition under IIEM (Improved IEM) formulation provided by Fung et al. 2002. 
The extended domain of validity (for large roughness or correlation length) is produced by using the transition Fresnel 
coefficients (Fung et al. 2004). This code also produces bi-static coefficients for passive sensor. Multiple scattering 
for crosspol is implemented from the original formulation in Fung 92. The integral for multiple scattering is done by 
fixed order quadrature for faster computation. A more complex implementation would be AIEM (Wu et al 2004).

This code was based on the MATLAB code published by Ulaby & Long, 2014:
https://tools.grss-ieee.org/rscl1/coderecord.php?id=469
and Robbie Mallet's python version:
https://github.com/robbiemallett/IIEM

"""

import numpy as np
#from numba import jit
from smrt.core.optional_numba import numba
import timeit


from smrt.core.fresnel import fresnel_coefficients, fresnel_reflection_matrix
from smrt.core.error import SMRTError
from smrt.core.vector3 import vector3
from smrt.core.lib import smrt_matrix, abs2, generic_ft_even_matrix, cached_roots_legendre
from smrt.core.globalconstants import C_SPEED
from smrt.interface.geometrical_optics import shadow_function

from smrt.interface.iem_fung92 import IEM_Fung92
class IIEM_Fung2002(IEM_Fung92):
        """
        A moderate rough surface model for passive and active. Multiple scattering only for crosspol backscatter since 
        it's assumed to be negligeable for co pol (passive??? to be implemented). Use with care
        """

        optional_args = {"autocorrelation_function": "exponential",
                         "warning_handling": "print",
                         "series_truncation": 10,
                         "N_integral" : 20, # number of fixed quadrature for integral
                         "shadow_correction" : True,
                         "compute_crosspol" : True,# set False to disable cross-pol calculation
                         "transition_fresnel": True,
                         }
        

        
        def specular_reflection_matrix(self, frequency, eps_1, eps_2, mu1, npol):
                """compute the reflection coefficients for an array of incidence angles (given by their cosine)
                in medium 1. Medium 2 is where the beam is transmitted.

                :param eps_1: permittivity of the medium where the incident beam is propagating.
                :param eps_2: permittivity of the other medium
                :param mu1: array of cosine of incident angles
                :param npol: number of polarization

                :return: the reflection matrix
        """
                k2 = (2 * np.pi * frequency / C_SPEED) ** 2 * abs2(eps_1)
                # Eq: 2.1.94 in Tsang 2001 Tome I
                return fresnel_reflection_matrix(eps_1, eps_2, mu1, npol) * np.exp(-4 * k2 * self.roughness_rms**2 * mu1**2)

        def transition_fresnel_coefficients(self, eps_1, eps_2, mu_i, k, k_w, n):
                """
                calculate the transition fresnel coefficients for AIEM
                """
                eps_r = eps_2.real
        
                #at 0
                Rv_0, Rh_0, _ = fresnel_coefficients(eps_1, eps_2, 1)
                # at mu
                Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)

                #sin_ i squared
                sin_i2 = 1 - mu_i**2
                Fv = 8 * abs2(Rv_0) * sin_i2 * ((mu_i + np.sqrt(eps_r - sin_i2)) / (mu_i * np.sqrt(eps_r - sin_i2)))
                Fh = 8 * abs2(Rh_0) * sin_i2 * ((mu_i + np.sqrt(eps_r - sin_i2)) / (mu_i * np.sqrt(eps_r - sin_i2)))

                Sv_0 = 1/abs2(1 + (8 * Rv_0) / (Fv * mu_i))
                Sh_0 = 1/abs2(1 + (8 * Rh_0) / (Fh * mu_i))

                rms_mu_over_factorial = np.cumprod((k.norm() * self.roughness_rms * mu_i)**(2) / n, axis = -1)

                factor_Rv0 = 2**(n+1) * Rv_0 * np.exp(-(k.norm() * self.roughness_rms * mu_i)**2) / mu_i
                factor_Rh0 = 2**(n+1) * Rh_0 * np.exp(-(k.norm() * self.roughness_rms * mu_i)**2) / mu_i

                Sv_n = np.sum(abs2(Fv)/4 * rms_mu_over_factorial * self.W_n(n, k_w), axis = -1, keepdims=True)
                Sv_d = np.sum((rms_mu_over_factorial * abs2(Fv/2 + factor_Rv0)* self.W_n(n, k_w)), axis = -1, keepdims=True)

                # Sv_n = abs2(Fv)/4 * np.sum(rms_mu_over_factorial)
                # Sv_d = np.sum((rms_mu_over_factorial * abs2(Fv/2 + factor_Rv0)))

                Sv = Sv_n / Sv_d
                sum_rms_Wn = rms_mu_over_factorial * self.W_n(n, k_w)
                sum_rms_Fh_wn =  rms_mu_over_factorial * abs2(Fh/2 + factor_Rh0)* self.W_n(n, k_w)
                Sh = np.sum(abs2(Fv)/4 * sum_rms_Wn / sum_rms_Fh_wn, axis = -1, keepdims=True)

                gamma_v = 1 - (Sv / Sv_0)
                gamma_h = 1 - (Sh / Sh_0)

                Rv_t = Rv + (Rv_0 - Rv) * gamma_v
                Rh_t = Rv + (Rh_0 - Rh) * gamma_h

                return Rv_t, Rh_t
        

        def diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol):
                """compute the reflection coefficients for an array of incident, scattered and azimuth angles
                        in medium 1. Medium 2 is where the beam is transmitted.

                :param eps_1: permittivity of the medium where the incident beam is propagating.
                :param eps_2: permittivity of the other medium
                :param mu1: array of cosine of incident angles
                :param npol: number of polarization

                :return: the reflection matrix
                """

                mu_i = np.atleast_1d(clip_mu(mu_i))[np.newaxis, np.newaxis, :, np.newaxis]
                mu_s = np.atleast_1d(clip_mu(mu_s))[np.newaxis, :, np.newaxis, np.newaxis]
                dphi = np.atleast_1d(dphi)[:, np.newaxis, np.newaxis, np.newaxis]

                # if not np.allclose(mu_s, mu_i) or not np.allclose(dphi, np.pi):
                #     raise NotImplementederror("Only the backscattering coefficient is implemented at this stage."
                #                                 "This is a very preliminary implementation")

                # if len(np.atleast_1d(dphi)) != 1:   
                #     raise NotImplementederror("Only the backscattering coefficient is implemented at this stage. ")


                #incident wavenumber
                k = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1).real, mu_i, 0)
                #scattered wavenumber
                k_s = vector3.from_angles(2 * np.pi * frequency / C_SPEED * np.sqrt(eps_1).real, mu_s, dphi)

                """
                wavenumber for roughness spectra 
                k_w calculation eqn 4.6 (Fung and Chen, 2010)
                # k * np.sqrt((sin_s * cos_phi_s - sin_i * cos_phi_i) ** 2 + (sin_s * sin_phi_s - sin_i * sin_phi_i) ** 2)) # from ulaby code
                # phi_i = 0
                # phi_s = dphi = phi_s - 0 
                # k_w with dphi assumimg phi_i = 0
                # (sin_s * cos_dphi - sin_i) ** 2 + (sin_s * sin_dphi) ** 2)
                
                """
                sin_i = np.sqrt(1 - mu_i**2)
                sin_s = np.sqrt(1 - mu_s**2)
                cos_dphi = np.cos(dphi)
                sin_dphi = np.sqrt(1 - cos_dphi**2)
                k_w = k.norm() * np.sqrt((sin_s * cos_dphi - sin_i) ** 2 + (sin_s * sin_dphi) ** 2)

                ks = np.abs(k.norm() * self.roughness_rms) 
                #kl = np.abs(k.norm() * self.corr_length)
                #self.check_validity(ks)

                # prepare the series
                N = self.series_truncation
                n = np.arange(1, N + 1, dtype=np.float64)#[:, None]
                n = np.atleast_1d(n)[np.newaxis, np.newaxis, np.newaxis, :]

                rms2 = self.roughness_rms**2
                rms2_over_fractorial = np.cumprod(rms2 / n, axis= -1)#[:, None]
                
                #transition fresnel
                if self.transition_fresnel:
                        Rv, Rh = self.transition_fresnel_coefficients(eps_1, eps_2, mu_i, k, k_w, n)
                else:
                        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)

                Ivv_n, Ihh_n = calculate_Iqp(eps_1, eps_2, k.norm(), k.z, k_s.z, Rv, Rh, n, mu_i, mu_s, dphi, rms2)
                
                # Eq 14 in wu et al. 2004
                coef = k.norm2() / 2 * np.exp(-rms2 * (k.z**2 + k_s.z**2))
                coef_n = rms2_over_fractorial * self.W_n(n, k_w)

                if self.shadow_correction:
                        # sin_i[sin_i < 1e-3] = 1e-3
                        # sin_s[sin_s < 1e-3] = 1e-3
                        rms_slope = self.roughness_rms / self.corr_length
                        # rms_slope squared because shadow function takes mean_squared_slope as input
                        s = 1 / (1 + shadow_function(rms_slope**2, mu_i / sin_i))
                        coef *= s

                sigma_vv = np.sum(coef * coef_n * abs2(Ivv_n) / (4 * np.pi * mu_i), axis =-1)
                sigma_hh = np.sum(coef * coef_n * abs2(Ihh_n) / (4 * np.pi * mu_i), axis =-1)
                reflection_coefficients = smrt_matrix.zeros((npol, npol, dphi.shape[0], mu_s.shape[1], mu_i.shape[2]))
                
                reflection_coefficients[0,0] = sigma_vv
                reflection_coefficients[1,1] = sigma_hh

                # only for backscatter
                # calculating multiple scattering contribution for cross-pol backscatter
                # a double integration function
                if self.compute_crosspol :
                        # take regular fresnel, not transitionnal... not valid for cross for now
                        Rv, Rh, _ = fresnel_coefficients(eps_1, eps_2, mu_i)
                        Rvh = (Rv - Rh) / 2
                        ks2 = ks**2
                        svh = self.double_integral(k, ks2, mu_i, eps_2, Rvh, n, self.N_integral)
                        svh = svh.reshape(1, 1, mu_i.shape[2], 1)

                        if self.shadow_correction:
                                # rms_slope squared because shadow function takes mean_squared_slope as input
                                s = 1 / (1 + shadow_function(rms_slope**2, mu_i / sin_i))

                                svh *= s
                        svh *= 1/ (4 * np.pi * mu_i)
                        svh = svh.reshape(1, 1, mu_i.shape[2])
                        reflection_coefficients[0,1] = svh
                        reflection_coefficients[1,0] = svh

                return reflection_coefficients
        
        def ft_even_diffuse_reflection_matrix(self, frequency, eps_1, eps_2, mu_s, mu_i, m_max, npol):

                def reflection_function(dphi):
                        return self.diffuse_reflection_matrix(frequency, eps_1, eps_2, mu_s, mu_i, dphi, npol=npol)

                return generic_ft_even_matrix(reflection_function, m_max, nsamples=256)
        
        def diffuse_transmission_matrix(self, frequency, eps_1, eps_2, mu_t, mu_i, dphi, npol):
                """compute the transmission coefficients for the azimuthal mode m
                and for an array of incidence angles (given by their cosine)
                in medium 1. Medium 2 is where the beam is transmitted.

                :param eps_1: permittivity of the medium where the incident beam is propagating.
                :param eps_2: permittivity of the other medium
                :param mu1: array of cosine of incident angles
                :param npol: number of polarization

                :return: the transmission matrix
                """

                return NotImplementedError("The use of the iiem is restricted to substrate only for now," \
                                           " Missing the implementation of the diffuse transmission")

        
        def W_n_2D(self, n, k, rx, ry, sin_i):
                """
                2 Dimension roughness spectra for n (multiple scattering)
                """

                kl2 = (k.norm() * self.corr_length)**2

                if self.autocorrelation_function == "gaussian":
                        w_n = 0.5 * kl2/n* np.exp(-kl2*((rx-sin_i)**2 + ry**2)/(4*n))
                        return w_n
                
                elif self.autocorrelation_function == "exponential":
                        w_n = n* kl2 / (n**2 + kl2 *((rx - sin_i)**2 + ry**2))**1.5
                        return w_n 
        
                else:
                        raise SMRTError("The autocorrelation function must be expoential or gaussian")


        def W_m_2D(self, n, k, rx, ry, sin_i):
                """
                2 Dimension roughness spectra for m (multiple scattering)
                """
                kl2 = (k.norm() * self.corr_length)**2

                if self.autocorrelation_function == "gaussian":
                        w_n = 0.5 * kl2 / n* np.exp(-kl2 * ((rx + sin_i)**2 + ry**2)/(4 * n))
                        return w_n
                
                elif self.autocorrelation_function == "exponential":
                        w_n = n* kl2 / (n**2 + kl2 *((rx + sin_i)**2 + ry**2))**1.5
                        return w_n 
        
                else:
                        raise SMRTError("The autocorrelation function must be exponential or gaussian")

        def xpol_integralfunction(self, r, dphi, k, ks2, mu_i, eps_2, Rvh, n):
                #expand dim to accomodate all variables
                #summation count
                m = n.reshape(1, 1, n.shape[-1], 1, 1)
                n = n.reshape(1, n.shape[-1], 1, 1, 1)
                #multiple angles
                mu_i = mu_i.reshape(mu_i.shape[2], 1, 1, 1, 1)
                Rvh = Rvh.reshape(Rvh.shape[2], 1, 1, 1, 1)
                #integral variables
                r = r.reshape(1, 1, 1, r.shape[0], r.shape[1])
                dphi = dphi.reshape(1, 1, 1, dphi.shape[0], dphi.shape[1])


                mu_i2 = mu_i**2
                sin_i = np.sqrt(1 - mu_i2)
                cos_dphi = np.cos(dphi)
                sin_dphi = np.sqrt(1 - cos_dphi**2)
                rx = r * cos_dphi
                ry = r * sin_dphi
                r2 = r**2

                # calculation of the field coefficients
                q = np.sqrt(1.0001 - r2)
                qt = np.sqrt(eps_2 - r2)

                a = (1 + Rvh) /q
                b = (1 - Rvh) /q
                c = (1 + Rvh) /qt
                d = (1 - Rvh) /qt

                # calculate cross-pol coefficient
                # reorganised from eqn A28 Fung et al 1992
                B3 = rx * ry / mu_i
                fvh1 = (b - c) * (1 - 3 * Rvh) - (b - c / eps_2) * (1 + Rvh)
                fvh2 = (a - d) * (1 + 3 * Rvh) - (a - d * eps_2) * (1 - Rvh)
                Fvh = abs2((fvh1 + fvh2) * B3)

                # # # calculate shadowing func for multiple scattering
                rms_slope = self.roughness_rms/self.corr_length
                sha = 1 / (1 + shadow_function(rms_slope**2, q / r))

                #calculate expressions for the surface spectra
                w_n = self.W_n_2D(n, k, rx, ry, sin_i)
                w_m = self.W_n_2D(m, k, rx, ry, sin_i)


                #--compute VH scattering coefficient
                vh_coef = np.exp(-2* ks2 * mu_i2) /(16 * np.pi)
                vhmnsum = w_n * w_m * (ks2 * mu_i2)**(n + m) / np.cumprod(n, axis =1) / np.cumprod(m, axis =2)
                #sum over axis for n and  m
                VH = np.sum(4 * vh_coef * Fvh * vhmnsum * r * sha, axis = (1,2))
                return VH
        

        def double_integral(self, k, ks2, mu_i, eps_2, Rvh, n, n_order):
                """
                Double integral function that is vectorized to handle multidimensionnal integrand (mu_i)
                Using Gauss legendre polynomials to do a fixed order quadrature 
                Modified from ChatGPT, I'm not that intelligent (Julien)
                """
                # can handle multidimensionnal integrand
                # Integration bounds
                a_r, b_r = 0.1, 1.0
                a_phi, b_phi = 0.0, np.pi

                # Get Gauss-Legendre nodes and weights
                nodes_r, weights_r = cached_roots_legendre(n_order)
                nodes_phi, weights_phi = cached_roots_legendre(n_order)

                # Rescale from [-1, 1] to [a, b]
                r = 0.5 * (nodes_r + 1) * (b_r - a_r) + a_r    
                phi = 0.5 * (nodes_phi + 1) * (b_phi - a_phi) + a_phi  
                wr = 0.5 * (b_r - a_r) * weights_r             
                wphi = 0.5 * (b_phi - a_phi) * weights_phi       

                # Create 2D meshgrid of r and phi shape: (n_order, n_order)
                R, PHI = np.meshgrid(r, phi, indexing='ij')  
                WR, WPHI = np.meshgrid(wr, wphi, indexing='ij') 

                # Evaluate integrand on the whole grid # shape (mu_i, n_order, n_order)
                integrand_vals = self.xpol_integralfunction(R, PHI, k=k, ks2=ks2, mu_i=mu_i, eps_2=eps_2, Rvh=Rvh, n=n)

                # Multiply integrand by weights and sum over phi and r
                integral_result = np.sum(integrand_vals * WR * WPHI, axis = (1,2))
                
                return integral_result
        

        # def double_integral(self, k, ks2, mu_i, eps_2, Rvh, n, n_order):
        #         #fixed order quadrature double integral
        #         # not suitable for multidimensionnal integrand, needed for multiple mu_i

        #         phi_bounds = (0, np.pi)
        #         r_bounds = (0.1, 1)
                
        #         #Create a partial function to fix the parameters for fixed_quad
        #         f_with_params = partial(self.xpol_integralfunc_vec, k=k, ks2=ks2, mu_i=mu_i, eps_2=eps_2, Rvh=Rvh, n=n)
                
        #         #Integrate with respect to r for each fixed value of phi
        #         def integrand_phi(phi):
        #                 result, _ = fixed_quad(lambda r: f_with_params(r, phi), r_bounds[0], r_bounds[1], n=n_order)
        #                 return result
                
        #         #Integrate the result with respect to phi
        #         result, _ = fixed_quad(integrand_phi, phi_bounds[0], phi_bounds[1], n=n_order)
                
        #         return result
        
        # def double_integral(self, k, ks2, mu_i, eps_2, Rvh, n, N):
        #         #precise but way too slow
               
        
        #         result_integral = dblquad(lambda phi, r : self.xpol_integralfunc_vec(r, phi, k, ks2, mu_i, eps_2, Rvh, n),
        #                                                  0.1, 1, lambda r : 0, lambda r : np.pi)[0]

        #         return result_integral

        

# check geo optics, already define there...  
def clip_mu(mu):
        # avoid large zenith angles that causes many troubles
        return np.clip(mu, 0.1, 1)

# def calculate_F(ud, is_, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi):
#         """
#         code from ulaby et al 2014 matlab code and Robbie Mallet https://github.com/robbiemallett/IIEM
#         eqn from Fung et al 2002, DOI:10.1163/156939302X01119
#         """
#         #geometry
#         sin_i = np.sqrt(1 - mu_i**2)
#         sin_s = np.sqrt(1 - mu_s**2)
#         cos_phi_i = 1.0 # np.cos(0)
#         #sin_phi_i = 0 #np.sin(0)
#         cos_dphi = np.cos(dphi)
#         sin_dphi = np.sqrt(1 - cos_dphi**2)
        
#         #clip to avoid negative sqrt
#         eps_r_sin_i2 = np.clip(eps_r - sin_i  ** 2, 0.01, eps_r)
#         sin_cosdphi_diff = sin_s * cos_dphi - sin_i * cos_phi_i
#         knorm_sin_allangle_squared = k_norm * sin_i * sin_s * sin_dphi  ** 2

#         if is_ == 1:

#                 Gqi = ud * kz
#                 Gqti = ud * k_norm * np.sqrt(eps_r_sin_i2)
#                 qi = ud * kz

#                 c11 = k_norm * cos_dphi * (k_sz - qi)
#                 c21 = mu_i * (cos_dphi * (k_norm  ** 2 * sin_i * cos_phi_i * (sin_cosdphi_diff) + Gqi * (k_norm * mu_s - qi))
#                         + k_norm  ** 2 * cos_phi_i * sin_i * sin_s * sin_dphi  ** 2)
#                 c31 = k_norm * sin_i * (sin_i * cos_phi_i * cos_dphi * (k_norm * mu_s - qi) - Gqi * (cos_dphi * (sin_cosdphi_diff) + sin_s * sin_dphi ** 2))
#                 c41 = k_norm * mu_i * (cos_dphi * mu_s * (k_norm * mu_s - qi) + k_norm * sin_s * (sin_cosdphi_diff))
#                 c51 = Gqi * (cos_dphi * mu_s * (qi - k_norm * mu_s) - k_norm * sin_s * (sin_cosdphi_diff))

#                 c12 = k_norm * cos_dphi * (k_sz - qi)
#                 c22 = mu_i * (cos_dphi * (k_norm  ** 2 * sin_i * cos_phi_i * (sin_cosdphi_diff) + Gqti * (k_norm * mu_s - qi))
#                         + k_norm  ** 2 * cos_phi_i * sin_i * sin_s * sin_dphi ** 2)
#                 c32 = k_norm * sin_i * (sin_i * cos_phi_i * cos_dphi * (k_norm * mu_s - qi) - Gqti * (cos_dphi * (sin_cosdphi_diff) - sin_s * sin_dphi ** 2))
#                 #c42 = c41
#                 c52 = Gqti * (cos_dphi * mu_s * (qi - k_norm * mu_s) - k_norm * sin_s * (sin_cosdphi_diff))


#         if is_ == 2:

#                 Gqs = ud * k_sz
#                 Gqts = ud * k_norm * np.sqrt(eps_r_sin_i2)
#                 qs = ud * k_sz

#                 c11 = k_norm * cos_dphi * (kz + qs)
#                 c21 = Gqs * (cos_dphi * (mu_i * (k_norm * mu_i + qs) - k_norm * sin_i * (sin_cosdphi_diff)) - knorm_sin_allangle_squared)
#                 c31 = k_norm * sin_s * (k_norm * mu_i * (sin_cosdphi_diff) + sin_i * (kz + qs))
#                 c41 = k_norm * mu_s * (cos_dphi * (mu_i * (kz + qs) - k_norm * sin_i * (sin_cosdphi_diff)) - knorm_sin_allangle_squared)
#                 c51 = -mu_s * (k_norm  ** 2 * sin_s * (sin_cosdphi_diff) + Gqs * cos_dphi * (kz + qs))

#                 c12 = k_norm * cos_dphi * (kz + qs)
#                 c22 = Gqts * (cos_dphi * (mu_i * (kz + qs) - k_norm * sin_i * (sin_cosdphi_diff)) - knorm_sin_allangle_squared)
#                 c32 = k_norm * sin_s * (k_norm * mu_i * (sin_cosdphi_diff) + sin_i * (kz + qs))
#                 #c42 = c41
#                 c52 = -mu_s * (k_norm  ** 2 * sin_s * (sin_cosdphi_diff) + Gqts * cos_dphi * (kz + qs))


#         q = kz
#         qt = k_norm * np.sqrt(eps_r_sin_i2)

#         Fvv = (1 + Rv) * (-(1 - Rv) * c11 / q + (1 + Rv) * c12 / qt) + \
#                 (1 - Rv) * ((1 - Rv) * c21 / q - (1 + Rv) * c22 / qt) + \
#                 (1 + Rv) * ((1 - Rv) * c31 / q - (1 + Rv) * c32 / eps_r / qt) + \
#                 (1 - Rv) * ((1 + Rv) * c41 / q - eps_r * (1 - Rv) * c41 / qt) + \
#                 (1 + Rv) * ((1 + Rv) * c51 / q - (1 - Rv) * c52 / qt)


#         Fhh = (1 + Rh) * ((1 - Rh) * c11 / q - eps_r * (1 + Rh) * c12 / qt) - \
#                 (1 - Rh) * ((1 - Rh) * c21 / q - (1 + Rh) * c22 / qt) - \
#                 (1 + Rh) * ((1 - Rh) * c31 / q - (1 + Rh) * c32 / qt) - \
#                 (1 - Rh) * ((1 + Rh) * c41 / q - (1 - Rh) * c41 / qt) - \
#                 (1 + Rh) * ((1 + Rh) * c51 / q - (1 - Rh) * c52 / qt)


#         return Fvv, Fhh

def calculate_F(ud, is_, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi):

        """Optimized version that eliminates unnecessary arrays and redundant calculations"""

        # Geometry - compute once
        sin_i = np.sqrt(1 - mu_i**2)
        sin_s = np.sqrt(1 - mu_s**2)
        cos_dphi = np.cos(dphi)
        sin_dphi = np.sqrt(1 - cos_dphi**2)

        # Shared terms
        eps_r_sin_i2 = np.maximum(eps_r - sin_i**2, 0.01)  # np.clip -> np.maximum for Numba
        sin_cosdphi_diff = sin_s * cos_dphi - sin_i
        knorm_sin_allangle_squared = k_norm * sin_i * sin_s * sin_dphi**2

        # Common terms
        q = kz
        qt = k_norm * np.sqrt(eps_r_sin_i2)

        if is_ == 1:
                # Compute G terms
                Gqi = ud * kz
                Gqti = ud * qt
                qi = ud * kz

                # Pre-compute frequently used terms
                k_norm_cos_dphi = k_norm * cos_dphi
                k_sz_minus_qi = k_sz - qi
                k_norm_mu_s_minus_qi = k_norm * mu_s - qi
                k_norm2_sin_i = k_norm**2 * sin_i
                k_norm_sin_s_sin_cosdphi_diff = k_norm * sin_s * sin_cosdphi_diff

                # Compute c coefficients inline in Fvv and Fhh calculations
                # c11 and c12 are the same
                c1x = k_norm_cos_dphi * k_sz_minus_qi

                # c21 terms
                c21_term1 = cos_dphi * (k_norm2_sin_i * sin_cosdphi_diff + Gqi * k_norm_mu_s_minus_qi)
                c21_term2 = k_norm2_sin_i * sin_s * sin_dphi**2
                c21 = mu_i * (c21_term1 + c21_term2)

                # c22 (similar to c21 but with Gqti)
                c22_term1 = cos_dphi * (k_norm2_sin_i * sin_cosdphi_diff + Gqti * k_norm_mu_s_minus_qi)
                c22 = mu_i * (c22_term1 + c21_term2)

                # Continue with other c terms...
                cos_dphi_sin_cosdphi_diff = cos_dphi * sin_cosdphi_diff
                sin_s_sin_dphi2 = sin_s * sin_dphi**2

                c31 = k_norm * sin_i * (sin_i * cos_dphi * k_norm_mu_s_minus_qi - 
                                        Gqi * (cos_dphi_sin_cosdphi_diff + sin_s_sin_dphi2))

                c32 = k_norm * sin_i * (sin_i * cos_dphi * k_norm_mu_s_minus_qi - 
                                        Gqti * (cos_dphi_sin_cosdphi_diff - sin_s_sin_dphi2))

                c41_c42 = k_norm * mu_i * (cos_dphi * mu_s * k_norm_mu_s_minus_qi + 
                                                k_norm_sin_s_sin_cosdphi_diff)

                c51 = Gqi * (cos_dphi * mu_s * (qi - k_norm * mu_s) - k_norm_sin_s_sin_cosdphi_diff)
                c52 = Gqti * (cos_dphi * mu_s * (qi - k_norm * mu_s) - k_norm_sin_s_sin_cosdphi_diff)

        else:  # is_ == 2
                Gqs = ud * k_sz
                Gqts = ud * qt
                qs = ud * k_sz

                # Pre-compute terms
                kz_plus_qs = kz + qs
                k_norm_cos_dphi = k_norm * cos_dphi
                mu_i_kz_plus_qs = mu_i * kz_plus_qs
                k_norm_sin_i_sin_cosdphi_diff = k_norm * sin_i * sin_cosdphi_diff

                c1x = k_norm_cos_dphi * kz_plus_qs

                common_term = cos_dphi * (mu_i_kz_plus_qs - k_norm_sin_i_sin_cosdphi_diff) - knorm_sin_allangle_squared
                c21 = Gqs * common_term
                c22 = Gqts * common_term

                c3x_c4x_term = k_norm * mu_i * sin_cosdphi_diff + sin_i * kz_plus_qs
                c31 = k_norm * sin_s * c3x_c4x_term
                c32 = c31  # Same calculation

                c41_c42 = k_norm * mu_s * common_term
                #c42 = c41  # Same calculation

                common_c5_term = k_norm**2 * sin_s * sin_cosdphi_diff
                c51 = -mu_s * (common_c5_term + Gqs * cos_dphi * kz_plus_qs)
                c52 = -mu_s * (common_c5_term + Gqts * cos_dphi * kz_plus_qs)

        # Calculate Fvv and Fhh directly
        one_plus_Rv = 1 + Rv
        one_minus_Rv = 1 - Rv
        one_plus_Rh = 1 + Rh
        one_minus_Rh = 1 - Rh

        Fvv = (one_plus_Rv * (-one_minus_Rv * c1x / q + one_plus_Rv * c1x / qt) + 
                one_minus_Rv * (one_minus_Rv * c21 / q - one_plus_Rv * c22 / qt) + 
                one_plus_Rv * (one_minus_Rv * c31 / q - one_plus_Rv * c32 / eps_r / qt) + 
                one_minus_Rv * (one_plus_Rv * c41_c42 / q - eps_r * one_minus_Rv * c41_c42 / qt) + 
                one_plus_Rv * (one_plus_Rv * c51 / q - one_minus_Rv * c52 / qt))

        Fhh = (one_plus_Rh * (one_minus_Rh * c1x / q - eps_r * one_plus_Rh * c1x / qt) - 
                one_minus_Rh * (one_minus_Rh * c21 / q - one_plus_Rh * c22 / qt) - 
                one_plus_Rh * (one_minus_Rh * c31 / q - one_plus_Rh * c32 / qt) - 
                one_minus_Rh * (one_plus_Rh * c41_c42 / q - one_minus_Rh * c41_c42 / qt) - 
                one_plus_Rh * (one_plus_Rh * c51 / q - one_minus_Rh * c52 / qt))

        return Fvv, Fhh


def calculate_Iqp(eps_1, eps_2, k_norm, kz, k_sz, Rv, Rh, n, mu_i, mu_s, dphi, rms2):
        """
        """
        eps_r = eps_2.real / eps_1.real

        sin_i = np.sqrt(1 - mu_i**2)
        sin_s = np.sqrt(1 - mu_s**2)

        fvv = 2 * Rv / (mu_i + mu_s) * (sin_i * sin_s - (1 + mu_i* mu_s) * np.cos(dphi)) # Eq 5 in Fung et al. 2002
        fhh = -2 * Rh / (mu_i + mu_s)* (sin_i * sin_s -(1 + mu_i* mu_s) * np.cos(dphi)) # Eq  in Fung et al. 2002
        #fvh, fhv = 0

        #Calculate the Field coefficients
        Fvv_up_i, Fhh_up_i = calculate_F(+1, 1, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)
        Fvv_up_s, Fhh_up_s = calculate_F(+1, 2, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)
        Fvv_dn_i, Fhh_dn_i = calculate_F(-1, 1, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)
        Fvv_dn_s, Fhh_dn_s = calculate_F(-1, 2, Rv, Rh, eps_r, k_norm, kz, k_sz, mu_i, mu_s, dphi)

        
        sub_kz = k_sz - kz
        add_kz = k_sz + kz
        kz2 = kz**2
        ksz2 = k_sz**2
        exp_kzkzs = np.exp(-rms2 * kz * k_sz)
        power_add_kz = add_kz**n
        power_sub_kz1 = sub_kz**(n-1)
        power_add_kz1 = add_kz**(n-1)

        # Precompute repeated exponentials
        exp_A1 = np.exp(-rms2 * (kz2 - kz * sub_kz))
        exp_A2 = np.exp(-rms2 * (kz2 + kz * sub_kz))
        exp_A3 = np.exp(-rms2 * (ksz2 - k_sz * (-sub_kz)))
        exp_A4 = np.exp(-rms2 * (ksz2 + k_sz * (-sub_kz)))


        # Calculate A using precomputed terms
        A_vv = power_sub_kz1 * Fvv_up_i * exp_A1 + \
                power_add_kz1 * Fvv_dn_i * exp_A2 + \
                power_add_kz1 * Fvv_up_s * exp_A3 + \
                power_sub_kz1 * Fvv_dn_s * exp_A4

        A_hh = power_sub_kz1 * Fhh_up_i * exp_A1 + \
                power_add_kz1 * Fhh_dn_i * exp_A2 + \
                power_add_kz1 * Fhh_up_s * exp_A3 + \
                power_sub_kz1 * Fhh_dn_s * exp_A4
        
        kirch_vv = power_add_kz * fvv * exp_kzkzs
        kirch_hh = power_add_kz * fhh * exp_kzkzs

        # kirch_vv = (k_sz + kz)**(n) * fvv * np.exp (-rms2 * kz * k_sz)
        # kirch_hh = (k_sz + kz)**(n) * fhh * np.exp (-rms2 * kz * k_sz)

        # A_vv = (k_sz - kz)**(n-1) * Fvv_up_i * np.exp(-rms2 * (kz**2 - kz * (k_sz -kz))) + \
        #         (k_sz + kz)**(n-1) * Fvv_dn_i * np.exp(-rms2 * (kz**2 + kz * (k_sz - kz))) + \
        #         (kz + k_sz)**(n-1) * Fvv_up_s * np.exp(-rms2 * (k_sz**2 - k_sz * (k_sz - kz))) + \
        #         (kz - k_sz)**(n-1) * Fvv_dn_s * np.exp(-rms2 * (k_sz**2 + k_sz * (k_sz - kz)))
        
        # A_hh = (k_sz - kz)**(n-1) * Fhh_up_i * np.exp(-rms2 * (kz**2 - kz * (k_sz -kz))) + \
        #         (k_sz + kz)**(n-1) * Fhh_dn_i * np.exp(-rms2 * (kz**2 + kz * (k_sz - kz))) + \
        #         (kz + k_sz)**(n-1) * Fhh_up_s * np.exp(-rms2 * (k_sz**2 - k_sz * (k_sz - kz))) + \
        #         (kz - k_sz)**(n-1) * Fhh_dn_s * np.exp(-rms2 * (k_sz**2 + k_sz * (k_sz - kz)))

        Ivv_n = kirch_vv + A_vv/4
        Ihh_n = kirch_hh + A_hh/4

        return Ivv_n, Ihh_n

