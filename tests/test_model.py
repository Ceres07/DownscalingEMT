from __future__ import annotations

import numpy as np

from topmodel_dispatch_hybrid import (
    Covariates,
    DisaggregationConfig,
    WeatherForcing,
    disaggregate_smips,
)
from topmodel_dispatch_hybrid.geospatial import disaggregate_smips_geospatial


def test_disaggregation_preserves_each_coarse_mean() -> None:
    config = DisaggregationConfig(subgrid_factor=4)
    smips = np.array([[0.12, 0.24], [0.31, 0.39]])
    covariates = _covariates(smips.shape, config.subgrid_factor)

    result = disaggregate_smips(smips, covariates, WeatherForcing(10.0, 4.0), config)

    for row in range(smips.shape[0]):
        for col in range(smips.shape[1]):
            block = result.soil_moisture_30m[
                row * config.subgrid_factor : (row + 1) * config.subgrid_factor,
                col * config.subgrid_factor : (col + 1) * config.subgrid_factor,
            ]
            assert np.isclose(np.nanmean(block), smips[row, col])


def test_wet_state_increases_terrain_weight_and_connectivity() -> None:
    config = DisaggregationConfig(subgrid_factor=4)
    smips = np.array([[0.10, 0.36]])
    covariates = _covariates(smips.shape, config.subgrid_factor)
    weather = WeatherForcing(antecedent_precip=np.array([[0.0, 30.0]]), potential_et=2.0)

    result = disaggregate_smips(smips, covariates, weather, config)

    assert result.connectivity[0, 1] > result.connectivity[0, 0]
    assert result.weights["terrain"][0, 1] > result.weights["terrain"][0, 0]
    assert result.weights["soil_storage"][0, 0] > result.weights["soil_storage"][0, 1]


def test_geospatial_disaggregation_preserves_mean_with_real_cell_bounds() -> None:
    import xarray as xr

    smips = xr.DataArray(
        np.array([[[0.12, 0.30]]]),
        dims=("time", "y", "x"),
        coords={"time": [np.datetime64("2023-01-01")], "y": [-33.0], "x": [148.0, 148.01]},
        attrs={"units": "mm"},
    )
    y = np.linspace(-33.005, -32.995, 4)
    x = np.linspace(147.995, 148.015, 8)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    twi = xr.DataArray(xx + yy, dims=("y", "x"), coords={"y": y, "x": x})
    hli = xr.DataArray(xx - yy, dims=("y", "x"), coords={"y": y, "x": x})
    holding = xr.DataArray(np.ones_like(twi), dims=("y", "x"), coords=twi.coords)

    result = disaggregate_smips_geospatial(
        smips,
        twi=twi,
        hli=hli,
        holding_capacity=holding,
        weather=WeatherForcing(antecedent_precip=20.0, potential_et=4.0),
        config=DisaggregationConfig(wilting_point=0.0, field_capacity=0.35, saturation=0.5),
    )

    left = result.soil_moisture.where(result.soil_moisture.x < 148.005, drop=True)
    right = result.soil_moisture.where(
        (result.soil_moisture.x >= 148.005) & (result.soil_moisture.x <= 148.015),
        drop=True,
    )
    assert np.isclose(np.nanmean(left.values), 0.12)
    assert np.isclose(np.nanmean(right.values), 0.30)
    assert np.isfinite(result.soil_moisture.values).all()


def test_geospatial_disaggregation_fills_full_fine_aoi_for_multiple_smips_pixels() -> None:
    import xarray as xr

    smips = xr.DataArray(
        np.array([[[0.10, 0.20], [0.30, 0.40]]]),
        dims=("time", "y", "x"),
        coords={
            "time": [np.datetime64("2023-01-01")],
            "y": [-33.00, -33.01],
            "x": [148.00, 148.01],
        },
    )
    y = np.linspace(-33.015, -32.995, 9)
    x = np.linspace(147.995, 148.015, 9)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    twi = xr.DataArray(xx + yy, dims=("y", "x"), coords={"y": y, "x": x})
    hli = xr.DataArray(xx - yy, dims=("y", "x"), coords={"y": y, "x": x})
    holding = xr.DataArray(np.ones_like(twi), dims=("y", "x"), coords=twi.coords)

    result = disaggregate_smips_geospatial(
        smips,
        twi=twi,
        hli=hli,
        holding_capacity=holding,
        weather=WeatherForcing(antecedent_precip=5.0, potential_et=2.0),
        config=DisaggregationConfig(wilting_point=0.0, field_capacity=0.35, saturation=0.5),
    )

    assert result.soil_moisture.shape == (1, 9, 9)
    assert np.isfinite(result.soil_moisture.values).all()


def _covariates(coarse_shape: tuple[int, int], factor: int) -> Covariates:
    fine_shape = (coarse_shape[0] * factor, coarse_shape[1] * factor)
    y, x = np.indices(fine_shape)
    return Covariates(
        twi=x + y,
        hli=x - y,
        holding_capacity=np.sin(x) + np.cos(y),
        vegetation_cover=(y + 1) / (fine_shape[0] + 1),
        infiltration_capacity=(x + 1) / (fine_shape[1] + 1),
    )
