from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class ObservationColumns:
    """Column names for point soil-moisture observations."""

    x: str
    y: str
    moisture: str
    time: str | None = None
    site_id: str | None = None


@dataclass(frozen=True)
class ObservationData:
    """Loaded observation table plus the columns used by the workflow."""

    frame: pd.DataFrame
    columns: ObservationColumns


DEFAULT_COVARIATE_VARIABLES = (
    "twi",
    "lfi",
    "hli",
    "eti",
    "slope",
    "aspect",
    "flow_acc",
    "holding_capacity",
    "vegetation_cover",
    "infiltration_capacity",
)


def load_soil_moisture_csv(
    path: str | Path,
    columns: ObservationColumns | None = None,
) -> ObservationData:
    """Load a point soil-moisture CSV and infer columns when not supplied."""

    frame = pd.read_csv(path)
    frame = frame.rename(columns=lambda name: str(name).strip().lstrip("\ufeff"))
    resolved = _resolve_observation_columns(frame, columns or infer_observation_columns(frame))
    out = frame.copy()
    if resolved.time is not None:
        out[resolved.time] = pd.to_datetime(out[resolved.time])
    out[resolved.x] = pd.to_numeric(out[resolved.x], errors="coerce")
    out[resolved.y] = pd.to_numeric(out[resolved.y], errors="coerce")
    out[resolved.moisture] = pd.to_numeric(out[resolved.moisture], errors="coerce")
    out = out.dropna(subset=[resolved.x, resolved.y, resolved.moisture]).reset_index(drop=True)
    return ObservationData(out, resolved)


def infer_observation_columns(frame: pd.DataFrame) -> ObservationColumns:
    """Infer coordinate, moisture, and optional time columns from a CSV."""

    names = {_normalise_column(name): name for name in frame.columns}
    x = _first_present(names, ("x", "lon", "long", "longitude", "easting"), required=False)
    y = _first_present(names, ("y", "lat", "latitude", "northing"), required=False)
    if x is None:
        x = _first_prefixed(names, "x")
    if y is None:
        y = _first_prefixed(names, "y")
    if x is None:
        raise ValueError(_missing_column_message("x coordinate", frame.columns, ("x", "lon", "longitude", "easting", "x_3577")))
    if y is None:
        raise ValueError(_missing_column_message("y coordinate", frame.columns, ("y", "lat", "latitude", "northing", "y_3577")))
    moisture = _first_present(
        names,
        (
            "soil_moisture",
            "soilmoisture",
            "moisture",
            "theta",
            "theta_v",
            "vwc",
            "vwc_percent",
            "sm",
        ),
    )
    time = _first_present(
        names,
        ("date", "time", "datetime", "timestamp", "yyyy_mm_dd", "yyyyMMdd"),
        required=False,
    )
    site_id = _first_present(names, ("site", "site_id", "station", "point_id", "id"), required=False)
    return ObservationColumns(x=x, y=y, moisture=moisture, time=time, site_id=site_id)


def extract_covariates_at_points(
    observations: ObservationData | pd.DataFrame,
    terrain: xr.Dataset,
    columns: ObservationColumns | None = None,
    variables: Iterable[str] = DEFAULT_COVARIATE_VARIABLES,
    coordinate_crs: str | None = "EPSG:4326",
    drop_outside: bool = True,
) -> pd.DataFrame:
    """Sample terrain/covariate values at each observation coordinate."""

    if isinstance(observations, ObservationData):
        frame = observations.frame
        columns = observations.columns
    elif columns is not None:
        frame = observations
    else:
        raise ValueError("columns must be supplied when observations is a raw DataFrame")

    x_values = frame[columns.x].to_numpy(dtype=float)
    y_values = frame[columns.y].to_numpy(dtype=float)
    x_values, y_values = _transform_points_if_needed(
        x_values,
        y_values,
        source_crs=coordinate_crs,
        target_crs=terrain.attrs.get("crs") or None,
    )

    out = frame.copy()
    out["_terrain_x"] = x_values
    out["_terrain_y"] = y_values
    out["inside_aoi"] = _inside_grid(x_values, y_values, terrain)

    point_dim = "observation"
    x_da = xr.DataArray(x_values, dims=point_dim)
    y_da = xr.DataArray(y_values, dims=point_dim)
    for variable in variables:
        if variable not in terrain:
            continue
        sampled = terrain[variable].sel(x=x_da, y=y_da, method="nearest")
        out[variable] = np.asarray(sampled.values, dtype=float)

    if drop_outside:
        out = out[out["inside_aoi"]].reset_index(drop=True)

    return out


def merge_daily_climate(
    observations: pd.DataFrame,
    climate: pd.DataFrame,
    observation_time_column: str,
    date_column: str | None = None,
    rainfall_column: str | None = None,
    et_column: str | None = None,
    antecedent_days: int = 7,
) -> pd.DataFrame:
    """Attach daily climate forcing to observations by nearest calendar day."""

    obs = observations.copy()
    climate_norm = normalise_climate_table(
        climate,
        date_column=date_column,
        rainfall_column=rainfall_column,
        et_column=et_column,
        antecedent_days=antecedent_days,
    )
    obs["_date_key"] = pd.to_datetime(obs[observation_time_column]).dt.normalize()
    merged = obs.merge(climate_norm, how="left", left_on="_date_key", right_on="date")
    return merged.drop(columns=["_date_key"])


def normalise_climate_table(
    climate: pd.DataFrame,
    date_column: str | None = None,
    rainfall_column: str | None = None,
    et_column: str | None = None,
    antecedent_days: int = 7,
) -> pd.DataFrame:
    """Return EMT-ready climate columns from SILO/OzWALD-style tables."""

    names = {_normalise_column(name): name for name in climate.columns}
    date_col = date_column or _first_present(names, ("date", "time", "yyyy_mm_dd", "yyyyMMdd"))
    rain_col = rainfall_column or _first_present(
        names,
        ("daily_rain", "rain", "rainfall", "precip", "precipitation", "pg"),
        required=False,
    )
    pet_col = et_column or _first_present(
        names,
        ("potential_et", "pet", "et0", "et_short_crop", "evap_pan"),
        required=False,
    )

    out = pd.DataFrame({"date": pd.to_datetime(climate[date_col]).dt.normalize()})
    if rain_col is not None:
        rain = pd.to_numeric(climate[rain_col], errors="coerce").fillna(0.0)
        out["rainfall"] = rain
        out["antecedent_precip"] = rain.rolling(antecedent_days, min_periods=1).sum()
    if pet_col is not None:
        out["potential_et"] = pd.to_numeric(climate[pet_col], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def _normalise_column(name: str) -> str:
    normalised = str(name).strip().lstrip("\ufeff").lower()
    return normalised.replace("-", "_").replace(" ", "_")


def _first_present(
    names: dict[str, str],
    candidates: tuple[str, ...],
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        key = _normalise_column(candidate)
        if key in names:
            return names[key]
    if required:
        raise ValueError(f"Could not infer required column. Tried: {', '.join(candidates)}")
    return None


def _first_prefixed(names: dict[str, str], prefix: str) -> str | None:
    prefix = _normalise_column(prefix)
    matches = sorted(
        original
        for normalised, original in names.items()
        if normalised == prefix or normalised.startswith(f"{prefix}_")
    )
    return matches[0] if matches else None


def _resolve_observation_columns(
    frame: pd.DataFrame,
    columns: ObservationColumns,
) -> ObservationColumns:
    return ObservationColumns(
        x=_resolve_column_name(frame, columns.x, "x coordinate"),
        y=_resolve_column_name(frame, columns.y, "y coordinate"),
        moisture=_resolve_column_name(frame, columns.moisture, "soil moisture"),
        time=(
            _resolve_column_name(frame, columns.time, "time")
            if columns.time is not None
            else None
        ),
        site_id=(
            _resolve_column_name(frame, columns.site_id, "site id")
            if columns.site_id is not None
            else None
        ),
    )


def _resolve_column_name(frame: pd.DataFrame, requested: str, role: str) -> str:
    if requested in frame.columns:
        return requested

    names = {_normalise_column(name): name for name in frame.columns}
    normalised = _normalise_column(requested)
    if normalised in names:
        return names[normalised]

    raise ValueError(_missing_column_message(role, frame.columns, (requested,)))


def _missing_column_message(
    role: str,
    columns: pd.Index,
    tried: tuple[str, ...],
) -> str:
    available = ", ".join(str(column) for column in columns)
    tried_text = ", ".join(tried)
    return (
        f"Could not find the {role} column. Tried: {tried_text}. "
        f"Available CSV columns are: {available}"
    )


def _transform_points_if_needed(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: str | None,
    target_crs: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not source_crs or not target_crs or str(source_crs) == str(target_crs):
        return x, y
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform

        src = CRS.from_user_input(source_crs)
        dst = CRS.from_user_input(target_crs)
        if src == dst:
            return x, y
        tx, ty = transform(src, dst, x.tolist(), y.tolist())
        return np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)
    except Exception:
        return x, y


def _inside_grid(x: np.ndarray, y: np.ndarray, terrain: xr.Dataset) -> np.ndarray:
    xmin = float(np.nanmin(terrain.x.values))
    xmax = float(np.nanmax(terrain.x.values))
    ymin = float(np.nanmin(terrain.y.values))
    ymax = float(np.nanmax(terrain.y.values))
    if terrain.sizes.get("x", 0) > 1:
        x_pad = abs(float(terrain.x.values[1] - terrain.x.values[0])) / 2.0
    else:
        x_pad = 0.0
    if terrain.sizes.get("y", 0) > 1:
        y_pad = abs(float(terrain.y.values[1] - terrain.y.values[0])) / 2.0
    else:
        y_pad = 0.0
    return (x >= xmin - x_pad) & (x <= xmax + x_pad) & (y >= ymin - y_pad) & (y <= ymax + y_pad)
