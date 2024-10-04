import numpy as np
import warnings

import pytest

from smrt import make_snowpack, make_emmodel
from smrt.core.sensor import active
from smrt.core.model import Model
from smrt.core.error import SMRTWarning
from smrt.interface.transparent import Transparent
from smrt.emmodel.nonscattering import NonScattering
from smrt.rtsolver.iterative_second import IterativeSecond
from smrt.core.fresnel import snell_angle


def setup_snowpack():
    temp = 250
    return make_snowpack([100], "homogeneous", density=[300], temperature=[temp], interface=[Transparent])


def setup_snowpack_with_DH():
    return make_snowpack(
        [0.5, 1000], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
    )


def setup_2layer_snowpack():
    return make_snowpack(
        [0.5, 1000], "homogeneous", density=[250, 300], temperature=2 * [250], interface=2 * [Transparent]
    )


def setup_inf_snowpack():
    temp = 250
    return make_snowpack(
        [10000000], "exponential", corr_length=1e-4, density=[300], temperature=[temp], interface=[Transparent]
    )


def test_returned_theta():
    sp = setup_snowpack()

    theta = [30, 40]
    sensor = active(17.25e9, theta)

    m = Model(NonScattering, IterativeSecond)
    res = m.run(sensor, sp)

    res_theta = res.coords["theta_inc"]
    np.testing.assert_allclose(res_theta, theta)


def test_selectby_theta():
    sp = setup_snowpack()

    theta = [30, 40, 50, 60]
    sensor = active(17.25e9, theta)

    m = Model(NonScattering, IterativeSecond)
    res = m.run(sensor, sp)

    print(res.data.coords)
    res.sigmaVV_dB(theta=50)



def test_depth_hoar_stream_numbers():
    # Will throw error if doesn't run
    sp = setup_snowpack_with_DH()
    sensor = active(13e9, 45)
    m = Model(NonScattering, IterativeSecond)
    m.run(sensor, sp).sigmaVV()


def test_2layer_pack():
    # Will throw error if doesn't run
    sp = setup_2layer_snowpack()
    sensor = active(13e9, 45)
    m = Model(NonScattering, IterativeSecond)
    m.run(sensor, sp).sigmaVV()


def test_shallow_snowpack():
    warnings.filterwarnings("error", message=".*optically shallow.*", module=".*iterative_second")

    with pytest.raises(SMRTWarning) as e_info:
        sp = make_snowpack(
            [0.15, 0.15], "homogeneous", density=[300, 250], temperature=2 * [250], interface=2 * [Transparent]
        )
        sensor = active(17e9, 45)
        m = Model(NonScattering, IterativeSecond)
        m.run(sensor, sp).sigmaVV()


def test_normal_call():
    sp = setup_snowpack()

    sensor = active(17.25e9, 30)

    m = Model(NonScattering, "iterative_second")
    res = m.run(sensor, sp)

def test_return_contributions():
    sp = setup_snowpack()

    sensor = active(17.25e9, 30)

    m = Model(NonScattering, "iterative_second", rtsolver_options = {'return_contributions' : True})
    res = m.run(sensor, sp)
    np.testing.assert_allclose(len(res.sigmaVV().contribution), 6)
