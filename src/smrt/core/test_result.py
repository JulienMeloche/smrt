# coding: utf-8

import numpy as np
import pytest
import xarray as xr

from smrt.core import result

# Tests written in response to -ve intensity bug in result.py
layer_coord = ("layer", [0, 1, 2])


@pytest.fixture
def res_example():
    return result.ActiveResult(
        [
            [
                [[4.01445680e-03, 3.77746658e-03, 0.00000000e00]],
                [[3.83889082e-03, 3.85904771e-03, 0.00000000e00]],
                [[2.76453599e-20, -2.73266027e-20, 0.00000000e00]],
            ]
        ],
        coords=[
            ("theta", [35]),
            ("polarization", ["V", "H", "U"]),
            ("theta_inc", [35]),
            ("polarization_inc", ["V", "H", "U"]),
        ],
        channel_map={
            "VV": dict(polarization="V", polarization_inc="V"),
            "VH": dict(polarization="H", polarization_inc="V"),
        },
        other_data={
            "ks": xr.DataArray([1.0, 2.0, 3.0], coords=[layer_coord]),
            "ka": xr.DataArray([3.0, 2.0, 1.0], coords=[layer_coord]),
            "ke": xr.DataArray([4.0, 4.0, 4.0], coords=[layer_coord]),
            "thickness": xr.DataArray([0.1, 0.1, 0.1], coords=[layer_coord]),
        },
    )


@pytest.fixture
def res_example2():
    return result.ActiveResult(
        [
            [
                [[4e-03, 3e-03, 0], [8e-03, 6e-03, 0]],
                [[3e-03, 3.85904771e-03, 0], [6e-03, 6.85904771e-03, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ],
            [
                [[4e-03, 3e-03, 0], [8e-03, 6e-03, 0]],
                [[3e-03, 3.85904771e-03, 0], [6e-03, 6.85904771e-03, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ],
        ],
        coords=[
            ("theta", [45, 50]),
            ("polarization", ["V", "H", "U"]),
            ("theta_inc", [45, 50]),
            ("polarization_inc", ["V", "H", "U"]),
        ],
        channel_map={
            "VV": dict(polarization="V", polarization_inc="V"),
            "VH": dict(polarization="H", polarization_inc="V"),
        },
        other_data={
            "ks": xr.DataArray([2.0, 4.0, 6.0], coords=[layer_coord]),
            "ka": xr.DataArray([3.0, 2.0, 1.0], coords=[layer_coord]),
            "ke": xr.DataArray([5.0, 6.0, 7.0], coords=[layer_coord]),
            "thickness": xr.DataArray([0.1, 0.1, 0.1], coords=[layer_coord]),
        },
    )


def test_methods(res_example):
    assert hasattr(res_example, "sigma")
    assert not hasattr(res_example, "Tb")


def test_positive_sigmaVV(res_example):
    assert res_example.sigmaVV() > 0


def test_positive_sigmaVH(res_example):
    assert res_example.sigmaVH() > 0


def test_positive_sigmaHV(res_example):
    assert res_example.sigmaHV() > 0


def test_positive_sigmaHH(res_example):
    assert res_example.sigmaHH() > 0


def test_sigma_dB(res_example):
    np.testing.assert_allclose(res_example.sigmaVV_dB(), -13.8379882755357)
    np.testing.assert_allclose(res_example.sigmaHH_dB(), -14.0094546848676)
    np.testing.assert_allclose(res_example.sigmaHV_dB(), -14.102249856026)
    np.testing.assert_allclose(res_example.sigmaVH_dB(), -14.0321985560285)


def test_name_sigma(res_example2):
    print(res_example2.sigmaVV_dB().name)
    assert res_example2.sigmaVV_dB().name == "sigmaVV_dB"
    assert res_example2.sigmaVV().name == "sigmaVV"
    assert res_example2.sigmaHV_dB().name == "sigmaHV_dB"
    assert res_example2.sigmaHV().name == "sigmaHV"


def test_sigma_dB_as_dataframe(res_example):
    df = res_example.sigma_dB_as_dataframe(channel_axis="column")

    assert "VV" in df.columns
    assert "VH" in df.columns

    np.testing.assert_allclose(df["VV"], -13.8379882755357)
    np.testing.assert_allclose(df["VH"], -14.0321985560285)


def test_to_dataframe_with_channel_axis_on_column(res_example):
    df = res_example.to_dataframe(channel_axis="column")

    assert "VV" in df.columns
    assert "VH" in df.columns

    np.testing.assert_allclose(df["VV"], -13.8379882755357)
    np.testing.assert_allclose(df["VH"], -14.0321985560285)


def test_to_dataframe_without_channel_axis(res_example):
    df = res_example.to_dataframe(channel_axis=None)
    print(df)  # this test fail for old version of xarray
    np.testing.assert_allclose(df.loc[(35, "V", "V"), :], (35, -13.8379882755357))
    np.testing.assert_allclose(df.loc[(35, "H", "V"), :], (35, -14.0321985560285))


def test_return_as_series(res_example):
    series = res_example.to_series()
    print(series)

    assert "VV" in series.index
    assert "VH" in series.index

    np.testing.assert_allclose(series.loc["VV"], -13.8379882755357)
    np.testing.assert_allclose(series.loc["VH"], -14.0321985560285)


def test_concat_results(res_example, res_example2):
    allresult = result.concat_results((res_example, res_example2), coord=("dim0", [0, 1]))
    print(allresult)
    assert "dim0" in allresult.data.dims
    assert len(allresult.data["dim0"]) == 2


def test_concat_results_other_data(res_example, res_example2):
    allresult = result.concat_results((res_example, res_example2), coord=("dim0", [0, 1]))

    assert allresult.other_data["ks"].dims == ("dim0", "layer")


def test_ks(res_example):
    np.testing.assert_allclose(res_example.ks(), res_example.other_data["ks"])


def test_ka(res_example):
    np.testing.assert_allclose(res_example.ka(), res_example.other_data["ka"])


def test_single_scattering_albedo(res_example):
    ssalb = res_example.single_scattering_albedo()

    print(ssalb)
    np.testing.assert_allclose(ssalb, np.array([1 / 4, 2 / 4, 3 / 4]))


def test_optical_depth(res_example):
    tau = res_example.optical_depth()

    np.testing.assert_allclose(tau, [0.4, 0.4, 0.4])


def test_save_open_result(tmp_path, res_example):
    filename = tmp_path / "result.nc"
    res_example.save(filename)

    res = result.open_result(filename)

    assert type(res) is result.ActiveResult
    assert res.channel_map == res_example.channel_map
    xr.testing.assert_equal(res.data, res_example.data)
    np.testing.assert_allclose(res.sigmaVV_dB(), res_example.sigmaVV_dB())
    np.testing.assert_allclose(res.sigmaVH_dB(channel=None), res_example.sigmaVH_dB())
    assert res.sigma_dB(channel="VV") == res_example.sigma_dB(channel="VV")


backend_modules = {
    "netcdf4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "scipy": "scipy",
}


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
def test_save_open_result_other_data(engine, tmp_path, res_example):
    pytest.importorskip(backend_modules[engine])

    filename = tmp_path / "result.nc"
    res_example.save(filename, netcdf_engine=engine)

    res = result.open_result(filename)

    assert set(res.other_data) == set(res_example.other_data)
    for k, v in res_example.other_data.items():
        assert isinstance(res.other_data[k], xr.DataArray)
        xr.testing.assert_equal(res.other_data[k], v)

    np.testing.assert_allclose(res.optical_depth(), res_example.optical_depth())


def test_save_open_result_dataset_in_other_data(tmp_path, res_example):
    ds = xr.Dataset({"a": ("layer", [1.0, 2.0, 3.0]), "b": ("layer", [4.0, 5.0, 6.0])}, coords=dict([layer_coord]))
    res_ds = result.ActiveResult(res_example.data, other_data={"ds": ds})

    filename = tmp_path / "result.nc"
    res_ds.save(filename)
    res = result.open_result(filename)

    assert isinstance(res.other_data["ds"], xr.Dataset)
    xr.testing.assert_equal(res.other_data["ds"], ds)
    assert res.channel_map == {}


def test_save_open_result_passive(tmp_path):
    res_passive = result.PassiveResult(
        [[250.0, 260.0], [270.0, 280.0]],
        coords=[("frequency", [19e9, 37e9]), ("polarization", ["V", "H"])],
        channel_map={"19V": dict(frequency=19e9, polarization="V")},
    )

    filename = tmp_path / "result.nc"
    res_passive.save(filename)
    res = result.open_result(filename)

    assert type(res) is result.PassiveResult
    assert res.channel_map == res_passive.channel_map
    assert res.other_data == {}
    assert res.TbV(frequency=37e9) == res_passive.TbV(frequency=37e9)
    assert res.Tb(channel="19V") == 250.0
