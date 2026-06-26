from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from topmodel_dispatch_hybrid.emt import EMTConfig, calibrate_emt
from topmodel_dispatch_hybrid.observations import (
    ObservationColumns,
    ObservationData,
    extract_covariates_at_points,
    load_soil_moisture_csv,
)
from topmodel_dispatch_hybrid.physical_emt import PhysicalEMTParameters, physical_emt_grid_components
from topmodel_dispatch_hybrid.smips_integration import (
    align_smips_coarse_to_terrain,
    smips_coarse_cell_labels,
    smips_to_mean_moisture,
)


def test_extract_points_and_calibrate_emt_grid() -> None:
    terrain = _terrain()
    observations = _observations_from_terrain(terrain)
    sampled = extract_covariates_at_points(observations, terrain, coordinate_crs=None)

    assert {"lfi", "eti", "inside_aoi"}.issubset(sampled.columns)
    assert sampled["inside_aoi"].all()

    result = calibrate_emt(
        sampled,
        EMTConfig(
            moisture_column="soil_moisture",
            time_column="date",
            lower_bound=0.05,
            upper_bound=0.50,
        ),
        terrain=terrain,
    )

    assert result.model.diagnostics["n_observations"] == len(sampled)
    assert result.model.diagnostics["average_spatial_nsce"] is not None
    assert np.isfinite(result.point_table["emt_prediction"]).all()
    assert result.prediction_grid is not None
    assert result.prediction_grid.shape == terrain["lfi"].shape


def test_load_soil_moisture_csv_handles_crs_suffixed_coordinate_columns(tmp_path) -> None:
    path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            " X_3577 ": [100.0, 101.0],
            "y_3577": [-35.0, -35.1],
            "VWC": [0.21, 0.24],
        }
    ).to_csv(path, index=False)

    loaded = load_soil_moisture_csv(
        path,
        ObservationColumns(x="x_3577", y="Y_3577", moisture="vwc"),
    )

    assert loaded.columns.x == "X_3577"
    assert loaded.columns.y == "y_3577"
    assert loaded.columns.moisture == "VWC"
    assert list(loaded.frame["X_3577"]) == [100.0, 101.0]


def test_emt_rejects_bounds_that_do_not_match_observation_units() -> None:
    terrain = _terrain()
    observations = _observations_from_terrain(terrain)
    sampled = extract_covariates_at_points(observations, terrain, coordinate_crs=None)
    sampled["water_mm"] = sampled["soil_moisture"] * 100.0

    try:
        calibrate_emt(
            sampled,
            EMTConfig(
                moisture_column="water_mm",
                time_column="date",
                lower_bound=0.0,
                upper_bound=0.6,
            ),
        )
    except ValueError as exc:
        assert "same units" in str(exc)
        assert "above upper_bound" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for mismatched EMT bounds")


def test_emt_predict_dataset_accepts_spatial_mean_moisture_grid() -> None:
    terrain = _terrain()
    observations = _observations_from_terrain(terrain)
    sampled = extract_covariates_at_points(observations, terrain, coordinate_crs=None)
    result = calibrate_emt(
        sampled,
        EMTConfig(
            moisture_column="soil_moisture",
            time_column="date",
            lower_bound=0.05,
            upper_bound=0.50,
        ),
        terrain=terrain,
    )
    mean_grid = xr.DataArray(
        np.linspace(0.10, 0.35, terrain.sizes["x"] * terrain.sizes["y"]).reshape(terrain["lfi"].shape),
        dims=("y", "x"),
        coords={"y": terrain.y, "x": terrain.x},
    )

    predicted = result.model.predict_dataset(terrain, mean_moisture=mean_grid)

    assert predicted.shape == terrain["lfi"].shape
    assert float(predicted.std()) > 0.0


def test_smips_relative_fullness_maps_to_emt_bounds() -> None:
    terrain = _terrain()
    observations = _observations_from_terrain(terrain)
    sampled = extract_covariates_at_points(observations, terrain, coordinate_crs=None)
    result = calibrate_emt(
        sampled,
        EMTConfig(
            moisture_column="soil_moisture",
            time_column="date",
            lower_bound=0.05,
            upper_bound=0.50,
        ),
    )
    smips = xr.DataArray(
        np.array([[0.0, 50.0], [100.0, 25.0]]),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        attrs={"layer": "SMIndexRaw"},
    )

    mean = smips_to_mean_moisture(smips, result.model)

    assert np.isclose(float(mean.min()), 0.05)
    assert np.isclose(float(mean.max()), 0.50)


def test_align_smips_coarse_to_terrain_preserves_coarse_tiles() -> None:
    terrain = xr.Dataset(coords={"y": np.linspace(0.0, 2.0, 5), "x": np.linspace(0.0, 2.0, 5)})
    smips = xr.DataArray(
        np.array([[[10.0, 20.0], [30.0, 40.0]]]),
        dims=("time", "y", "x"),
        coords={"time": [np.datetime64("2025-01-01")], "y": [0.0, 2.0], "x": [0.0, 2.0]},
    )

    aligned = align_smips_coarse_to_terrain(smips, terrain, source_crs=None)

    expected = np.array(
        [
            [10.0, 10.0, 10.0, 20.0, 20.0],
            [10.0, 10.0, 10.0, 20.0, 20.0],
            [10.0, 10.0, 10.0, 20.0, 20.0],
            [30.0, 30.0, 30.0, 40.0, 40.0],
            [30.0, 30.0, 30.0, 40.0, 40.0],
        ]
    )
    assert aligned.shape == (1, 5, 5)
    assert np.array_equal(aligned.isel(time=0).values, expected)


def test_physical_emt_process_weights_sum_to_one_with_coarse_labels() -> None:
    y = np.arange(4, dtype=float)
    x = np.arange(5, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    terrain = xr.Dataset(
        {
            "dem": (("y", "x"), 100.0 + yy + 0.2 * xx),
            "flow_acc": (("y", "x"), 1.0 + xx + yy),
            "slope": (("y", "x"), np.full((4, 5), 5.0)),
            "hli": (("y", "x"), 0.8 + 0.05 * xx),
        },
        coords={"y": y, "x": x},
    )
    smips = xr.DataArray(
        np.array([[0.15, 0.25], [0.20, 0.30]]),
        dims=("y", "x"),
        coords={"y": [0.0, 3.0], "x": [0.0, 4.0]},
    )
    labels = smips_coarse_cell_labels(smips, terrain, source_crs=None)
    theta_bar = align_smips_coarse_to_terrain(smips, terrain, source_crs=None)
    params = PhysicalEMTParameters(
        ks_v=100.0,
        ks_h=1000.0,
        porosity=0.4,
        eta_h=5.0,
        eta_v=10.0,
        beta_r=2.0,
        beta_a=2.0,
        alpha=0.26,
        z0=0.2,
        curvature_min=-0.01,
        epsilon=1.0,
        pet=4.0,
    )

    components = physical_emt_grid_components(terrain, params, theta_bar, normalization_labels=labels)
    weight_sum = (
        components["relative_w_g"].values
        + components["relative_w_l"].values
        + components["relative_w_r"].values
        + components["relative_w_a"].values
    )

    assert components["physical_emt_theta"].shape == terrain["dem"].shape
    assert np.allclose(weight_sum, 1.0)
    assert len(np.unique(labels.values)) == 4


def _terrain() -> xr.Dataset:
    y = np.arange(4, dtype=float)
    x = np.arange(5, dtype=float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    lfi = xx + 2 * yy
    eti = 1.0 - (xx / xx.max())
    return xr.Dataset(
        {
            "lfi": (("y", "x"), lfi),
            "eti": (("y", "x"), eti),
            "twi": (("y", "x"), lfi),
            "hli": (("y", "x"), 1.0 - eti),
        },
        coords={"y": y, "x": x},
    )


def _observations_from_terrain(terrain: xr.Dataset) -> ObservationData:
    rows = []
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 2.0), (4.0, 3.0)]
    for date, mean in ((pd.Timestamp("2023-01-01"), 0.16), (pd.Timestamp("2023-01-02"), 0.30)):
        rel = (mean - 0.05) / (0.50 - 0.05)
        w_lateral = rel * ((1 - rel) ** 0.75)
        w_radiative = 1 - rel
        for x, y in points:
            lfi = float(terrain["lfi"].sel(x=x, y=y))
            eti = float(terrain["eti"].sel(x=x, y=y))
            moisture = mean + 0.01 * w_lateral * lfi + 0.02 * w_radiative * eti
            rows.append({"x": x, "y": y, "date": date, "soil_moisture": moisture})
    frame = pd.DataFrame(rows)
    return ObservationData(
        frame=frame,
        columns=ObservationColumns(x="x", y="y", moisture="soil_moisture", time="date"),
    )
