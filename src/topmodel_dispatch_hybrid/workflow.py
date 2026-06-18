from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import pandas as pd
import xarray as xr

from .emt import EMTCalibrationResult, EMTConfig, calibrate_emt
from .observations import (
    ObservationColumns,
    extract_covariates_at_points,
    load_soil_moisture_csv,
    merge_daily_climate,
)
from .smips_integration import plot_emt_smips_side_by_side, predict_emt_from_smips
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
    smips_path: Path | None = None
    smips_prediction_path: Path | None = None
    smips_side_by_side_path: Path | None = None


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
    smips_path: str | Path | None = None,
    download_smips: bool = False,
    smips_layer: str = "TotalBucketRaw",
    smips_source: str = "auto",
    smips_mode: str = "auto",
    smips_start: date | str | None = None,
    smips_end: date | str | None = None,
    paddockts_path: str | Path | None = None,
    reload_smips: bool = False,
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
    smips_out_path = out_dir / f"{output_stub}_{smips_layer.lower()}_smips.nc"
    smips_prediction_path = out_dir / f"{output_stub}_emt_smips_prediction.nc"
    smips_side_by_side_path = out_dir / f"{output_stub}_emt_smips_side_by_side.png"

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

    saved_smips_path = None
    saved_smips_prediction_path = None
    saved_side_by_side_path = None
    if smips_path is not None or download_smips:
        smips = _load_or_download_smips_for_workflow(
            smips_path=smips_path,
            output_path=smips_out_path,
            download=download_smips,
            terrain=terrain,
            bbox=bbox,
            bbox_crs=bbox_crs,
            observations=observations.frame,
            time_column=observations.columns.time,
            start=smips_start,
            end=smips_end,
            layer=smips_layer,
            source=smips_source,
            output_stub=output_stub,
            paddockts_path=paddockts_path,
            reload=reload_smips,
        )
        saved_smips_path = Path(smips_path) if smips_path is not None else smips_out_path
        smips_prediction = predict_emt_from_smips(
            calibration.model,
            terrain,
            smips,
            mode=smips_mode,
            source_crs="EPSG:4326",
        )
        smips_prediction.to_netcdf(smips_prediction_path)
        plot_emt_smips_side_by_side(
            smips_prediction,
            smips,
            smips_side_by_side_path,
            smips_label=f"{smips_layer} ({smips.attrs.get('collection', smips.attrs.get('layer', 'SMIPS'))})",
        )
        saved_smips_prediction_path = smips_prediction_path
        saved_side_by_side_path = smips_side_by_side_path

    return EMTWorkflowResult(
        calibration=calibration,
        terrain_path=terrain_path,
        point_covariates_path=point_path,
        model_path=model_path,
        diagnostics_path=diagnostics_path,
        prediction_path=prediction_path,
        smips_path=saved_smips_path,
        smips_prediction_path=saved_smips_prediction_path,
        smips_side_by_side_path=saved_side_by_side_path,
    )


def _load_or_download_smips_for_workflow(
    *,
    smips_path: str | Path | None,
    output_path: Path,
    download: bool,
    terrain,
    bbox,
    bbox_crs,
    observations: pd.DataFrame,
    time_column: str | None,
    start: date | str | None,
    end: date | str | None,
    layer: str,
    source: str,
    output_stub: str,
    paddockts_path: str | Path | None,
    reload: bool,
) -> xr.DataArray:
    if smips_path is not None:
        return _read_smips_dataset(smips_path)
    if not download:
        raise ValueError("Set download_smips=True or provide smips_path")

    from .paddockts_bridge import load_or_download_smips, query_from_bbox

    bbox_wgs84 = _bbox_to_wgs84(bbox, bbox_crs, terrain)
    start_date, end_date = _resolve_smips_dates(observations, time_column, start, end)
    query = query_from_bbox(
        bbox=bbox_wgs84,
        start=start_date,
        end=end_date,
        stub=f"{output_stub}_smips",
        paddockts_path=paddockts_path,
    )
    return load_or_download_smips(
        query,
        output_path,
        layer=layer,
        source=source,
        reload=reload,
        paddockts_path=paddockts_path,
    )


def _read_smips_dataset(path: str | Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    data_vars = [name for name in ds.data_vars if name != "spatial_ref"]
    if not data_vars:
        raise ValueError(f"No SMIPS data variables found in {path}")
    da = ds[data_vars[0]].load()
    ds.close()
    return da


def _resolve_smips_dates(
    observations: pd.DataFrame,
    time_column: str | None,
    start: date | str | None,
    end: date | str | None,
) -> tuple[date, date]:
    if start is not None and end is not None:
        return _as_date(start), _as_date(end)
    if time_column is None or time_column not in observations:
        raise ValueError("SMIPS download requires smips_start/smips_end or dated observations")
    dates = pd.to_datetime(observations[time_column]).dt.date
    return _as_date(start) if start is not None else dates.min(), _as_date(end) if end is not None else dates.max()


def _as_date(value: date | str) -> date:
    return date.fromisoformat(value) if isinstance(value, str) else value


def _bbox_to_wgs84(
    bbox: tuple[float, float, float, float] | None,
    bbox_crs: str | None,
    terrain,
) -> tuple[float, float, float, float]:
    if bbox is not None:
        if not bbox_crs or str(bbox_crs).upper() in {"EPSG:4326", "4326"}:
            return tuple(float(v) for v in bbox)
        try:
            from rasterio.warp import transform_bounds

            return tuple(float(v) for v in transform_bounds(bbox_crs, "EPSG:4326", *bbox, densify_pts=21))
        except Exception as exc:
            raise ValueError("Could not transform bbox to EPSG:4326 for SMIPS download") from exc

    terrain_crs = terrain.attrs.get("crs") or "EPSG:4326"
    bounds = (
        float(terrain.x.min()),
        float(terrain.y.min()),
        float(terrain.x.max()),
        float(terrain.y.max()),
    )
    if str(terrain_crs).upper() in {"EPSG:4326", "4326"}:
        return bounds
    try:
        from rasterio.warp import transform_bounds

        return tuple(float(v) for v in transform_bounds(terrain_crs, "EPSG:4326", *bounds, densify_pts=21))
    except Exception as exc:
        raise ValueError("Could not infer an EPSG:4326 bbox from the terrain grid for SMIPS download") from exc
