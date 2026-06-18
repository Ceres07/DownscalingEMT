from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .emt import EMTCalibrationResult, EMTConfig, calibrate_emt
from .observations import (
    ObservationColumns,
    extract_covariates_at_points,
    load_soil_moisture_csv,
    merge_daily_climate,
)
from .terrain import TerrainBuildConfig, build_terrain_covariates_from_dem


@dataclass(frozen=True)
class EMTWorkflowResult:
    """Paths and in-memory outputs from an EMT calibration workflow run."""

    calibration: EMTCalibrationResult
    terrain_path: Path
    point_covariates_path: Path
    model_path: Path
    diagnostics_path: Path
    prediction_path: Path


def calibrate_emt_from_dem_and_csv(
    dem_path: str | Path,
    observations_csv: str | Path,
    output_dir: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    bbox_crs: str | None = "EPSG:4326",
    coordinate_crs: str | None = "EPSG:4326",
    latitude: float | None = None,
    observation_columns: ObservationColumns | None = None,
    climate_csv: str | Path | None = None,
    climate_date_column: str | None = None,
    rainfall_column: str | None = None,
    et_column: str | None = None,
    output_stub: str = "emt_calibration",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    ridge: float = 1e-6,
) -> EMTWorkflowResult:
    """Run the complete local EMT calibration workflow.

    Inputs are a DEM for the AOI and a CSV of point soil-moisture observations
    with coordinates. Outputs are a sampled point table, a calibrated model
    JSON, diagnostics JSON, and an EMT prediction grid.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    terrain_path = out_dir / f"{output_stub}_terrain_covariates.nc"
    point_path = out_dir / f"{output_stub}_point_covariates.csv"
    model_path = out_dir / f"{output_stub}_emt_model.json"
    diagnostics_path = out_dir / f"{output_stub}_diagnostics.json"
    prediction_path = out_dir / f"{output_stub}_emt_prediction.nc"

    terrain = build_terrain_covariates_from_dem(
        dem_path,
        TerrainBuildConfig(
            bbox=bbox,
            bbox_crs=bbox_crs,
            latitude=latitude,
            output_path=terrain_path,
        ),
    )
    observations = load_soil_moisture_csv(observations_csv, columns=observation_columns)
    points = extract_covariates_at_points(
        observations,
        terrain,
        coordinate_crs=coordinate_crs,
        drop_outside=True,
    )

    if climate_csv is not None:
        if observations.columns.time is None:
            raise ValueError("A time column is required to merge climate data")
        climate = pd.read_csv(climate_csv)
        points = merge_daily_climate(
            points,
            climate,
            observation_time_column=observations.columns.time,
            date_column=climate_date_column,
            rainfall_column=rainfall_column,
            et_column=et_column,
        )

    config = EMTConfig(
        moisture_column=observations.columns.moisture,
        time_column=observations.columns.time,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        ridge=ridge,
    )
    calibration = calibrate_emt(points, config, terrain=terrain)
    calibration.point_table.to_csv(point_path, index=False)
    calibration.model.to_json(model_path)
    Path(diagnostics_path).write_text(json.dumps(calibration.model.diagnostics, indent=2, sort_keys=True))
    if calibration.prediction_grid is not None:
        calibration.prediction_grid.to_netcdf(prediction_path)

    return EMTWorkflowResult(
        calibration=calibration,
        terrain_path=terrain_path,
        point_covariates_path=point_path,
        model_path=model_path,
        diagnostics_path=diagnostics_path,
        prediction_path=prediction_path,
    )
