
import numpy as np

from smrt.inputs.make_soil import make_soil


def test_make_soil_qnh():

    make_soil('soil_qnh', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, Q=0.16, Nv=0.11, Nh=1.63, H=0.65)

def test_make_soil_qnh_params():

    make_soil('soil_qnh', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, H=0.65)

def soil_setup():
    s = make_soil('soil_qnh', 'dobson85', 275, moisture=0.9, sand=0.2, clay=0.3, drymatter=1100, Q=0.16, Nv=0.11,
                  Nh=1.63, H=0.65)

    frequency = 1.4e9
    mu1 = np.cos(np.radians(np.arange(10, 80)))

    return s, frequency, mu1


def test_soil_qnh_reflection():

    s, frequency, mu1 = soil_setup()

    npol = 2

    refl = s.specular_reflection_matrix(frequency, 1, mu1, npol)
    refl = refl.diagonal()
    print(refl)
    np.testing.assert_allclose(refl[0],  0.330368453987, atol=1e-6)
    np.testing.assert_allclose(refl[139], 0.747605452207, atol=1e-6)


def test_soil_qnh_absorption():

    s, frequency, mu1 = soil_setup()

    npol = 2

    abso = s.absorption_matrix(frequency, 1, mu1, npol, False)
    abso = abso.diagonal()
    print(abso)
    np.testing.assert_allclose(abso[0],  0.66963155, atol=1e-6)
    np.testing.assert_allclose(abso[139], 0.25239455, atol=1e-6)