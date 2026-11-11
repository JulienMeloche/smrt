import numpy as np
import pytest

from smrt.core.layer import Layer
from smrt.core.lib import cached_roots_legendre
from smrt.emmodel.grg_cylinder import GeneralizedRayleighGansCylinder
from smrt.inputs.sensor_list import active
from smrt.microstructure_model.independent_cylinder import IndependentCylinder


@pytest.fixture
def cylinder_emmodel():
    radius = 2e-3
    height = 10e-2
    eps = 3 + 1j * 1e-3
    freq = 1.7e9

    sensor = active(freq, theta_inc=50)

    layer = Layer(
        thickness=1,
        microstructure_model=IndependentCylinder,
        height=height,
        radius=radius,
        frac_volume=0.5,
        permittivity_model=(1, eps),
    )

    return GeneralizedRayleighGansCylinder(sensor, layer)


def test_scattering(cylinder_emmodel):
    mu, weight = cached_roots_legendre(129, 0, 1)
    coef_weight = np.tile(np.repeat(0.5 * weight, 2), 2)

    fullmu = np.concatenate((-mu, mu))
    A = cylinder_emmodel.ft_even_phase(-mu, fullmu, 0).compress(mode=0, auto_reduce_npol=True)

    A *= coef_weight[np.newaxis, :]

    ks_vv = np.sum(A, axis=1)[0::2]
    ks_hh = np.sum(A, axis=1)[1::2]
    ks = cylinder_emmodel.ks(mu, npol=2).compress().diagonal()

    assert np.allclose(ks_vv, ks[0::2], rtol=1e-2)
    assert np.allclose(ks_hh, ks[1::2], rtol=2e-2)
