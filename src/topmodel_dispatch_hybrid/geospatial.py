from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .model import (
    Covariates,
    DisaggregationConfig,
    WeatherForcing,
    _clip_preserve_mean,
    _hydrological_state,
    _normalised_anomaly,
    _redistribution_amplitude,
    _relative_wetness_score,
    _state_dependent_weights,
)


@dataclass(frozen=True)
class GeospatialDisaggregationResult:
    soil_moisture: xr.DataArray
    relative_wetness_score: xr.DataArray
    connectivity: xr.DataArray
    weights: xr.Dataset


def disaggregate_smips_geospatial(
    smips: xr.DataArray,
    twi: xr.DataArray,
    hli: xr.DataArray,
    holding_capacity: xr.DataArray,
    weather: WeatherForcing | None = None,
    vegetation_cover: xr.DataArray | None = None,
    infiltration_capacity: xr.DataArray | None = None,
    config: DisaggregationConfig | None = None,
    bbox: list[float] | tuple[float, float, float, float] | None = None,
) -> GeospatialDisaggregationResult:
    """Disaggregate SMIPS to a fine x/y grid using nearest coarse-pixel labels.

    This adapter is for real rasters where 1 km / 30 m is not an integer.
    It assigns every fine-grid cell to its nearest SMIPS x/y center, then
    preserves each coarse SMIPS pixel mean over the assigned fine cells.
    """

    config = config or DisaggregationConfig()
    weather = weather or WeatherForcing()
    smips = _normalise_smips_coords(smips)

    if "time" not in smips.dims:
        smips = smips.expand_dims(time=[np.datetime64("NaT")])

    fine = xr.Dataset(
        {
            "twi": twi,
            "hli": hli,
            "holding_capacity": holding_capacity,
        }
    )
    if vegetation_cover is not None:
        fine["vegetation_cover"] = vegetation_cover
    if infiltration_capacity is not None:
        fine["infiltration_capacity"] = infiltration_capacity
    fine = fine.interp(x=twi.x, y=twi.y, method="nearest")
    if bbox is not None:
        fine = fine.sel(
            x=slice(min(bbox[0], bbox[2]), max(bbox[0], bbox[2])),
            y=slice(max(bbox[1], bbox[3]), min(bbox[1], bbox[3]))
            if fine.y.values[0] > fine.y.values[-1]
            else slice(min(bbox[1], bbox[3]), max(bbox[1], bbox[3])),
        )

    out = xr.full_like(
        xr.DataArray(
            np.full((smips.sizes["time"], fine.sizes["y"], fine.sizes["x"]), np.nan),
            dims=("time", "y", "x"),
            coords={"time": smips.time, "y": fine.y, "x": fine.x},
        ),
        np.nan,
    )
    score_out = xr.full_like(out, np.nan)
    connectivity = xr.full_like(smips, np.nan)
    weight_arrays = {
        name: xr.full_like(smips, np.nan)
        for name in (
            "terrain",
            "soil_storage",
            "vegetation",
            "evaporative_exposure",
            "infiltration",
        )
    }

    precip = _weather_data_array(weather.antecedent_precip, smips, "antecedent_precip")
    pet = _weather_data_array(weather.potential_et, smips, "potential_et")
    x_labels = _nearest_index_labels(fine.x.values.astype(float), smips.x.values.astype(float))
    y_labels = _nearest_index_labels(fine.y.values.astype(float), smips.y.values.astype(float))

    for iy in range(smips.sizes["y"]):
        fine_y = xr.DataArray(y_labels == iy, dims=("y",), coords={"y": fine.y})
        for ix in range(smips.sizes["x"]):
            fine_x = xr.DataArray(x_labels == ix, dims=("x",), coords={"x": fine.x})
            tile = fine.where(fine_y & fine_x, drop=True)
            if tile.sizes.get("x", 0) == 0 or tile.sizes.get("y", 0) == 0:
                continue

            for it in range(smips.sizes["time"]):
                mean_sm = float(smips.isel(time=it, y=iy, x=ix))
                if not np.isfinite(mean_sm):
                    continue

                state = _hydrological_state(
                    mean_sm,
                    float(precip.isel(time=it, y=iy, x=ix)),
                    float(pet.isel(time=it, y=iy, x=ix)),
                    config,
                )
                weights = _state_dependent_weights(state, config)
                connectivity[it, iy, ix] = state["connectivity"]
                for name, value in weights.items():
                    weight_arrays[name][it, iy, ix] = value

                covariates = Covariates(
                    twi=tile["twi"].values,
                    hli=tile["hli"].values,
                    holding_capacity=tile["holding_capacity"].values,
                    vegetation_cover=(
                        tile["vegetation_cover"].values
                        if "vegetation_cover" in tile
                        else None
                    ),
                    infiltration_capacity=(
                        tile["infiltration_capacity"].values
                        if "infiltration_capacity" in tile
                        else None
                    ),
                )
                block = (slice(0, tile.sizes["y"]), slice(0, tile.sizes["x"]))
                score = _relative_wetness_score(covariates, block, weights)
                anomaly = _normalised_anomaly(score)
                amplitude = _redistribution_amplitude(mean_sm, state, config)
                values = _clip_preserve_mean(
                    mean_sm + amplitude * anomaly,
                    target_mean=mean_sm,
                    lower=config.wilting_point,
                    upper=config.saturation,
                    tolerance=config.preserve_mean_tolerance,
                )

                out.loc[dict(time=smips.time[it], y=tile.y, x=tile.x)] = values
                score_out.loc[dict(time=smips.time[it], y=tile.y, x=tile.x)] = score

    out.name = "soil_moisture_downscaled"
    out.attrs.update(
        units=smips.attrs.get("units", "mm"),
        long_name="State-dependent TOPMODEL/DISPATCH SMIPS redistribution",
    )
    score_out.name = "relative_wetness_score"
    connectivity.name = "connectivity"

    return GeospatialDisaggregationResult(
        soil_moisture=out,
        relative_wetness_score=score_out,
        connectivity=connectivity,
        weights=xr.Dataset(weight_arrays),
    )


def _normalise_smips_coords(smips: xr.DataArray) -> xr.DataArray:
    if "xc" in smips.coords and "yc" in smips.coords:
        smips = smips.assign_coords(
            x=smips["xc"].isel(y=0).values,
            y=smips["yc"].isel(x=0).values,
        )
    if "longitude" in smips.coords and "latitude" in smips.coords:
        smips = smips.rename(longitude="x", latitude="y")
    if "x" not in smips.coords or "y" not in smips.coords:
        raise ValueError("SMIPS must have x/y, longitude/latitude, or xc/yc coordinates")
    return smips


def _weather_data_array(value: np.ndarray | xr.DataArray | float, like: xr.DataArray, name: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        arr = value
        if "time" in arr.dims:
            arr = arr.interp(time=like.time, method="nearest")
        if {"x", "y"}.issubset(arr.dims):
            arr = arr.interp(x=like.x, y=like.y, method="nearest")
        else:
            arr = arr.broadcast_like(like)
        return arr.astype(float)

    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        return xr.full_like(like, float(raw), dtype=float)
    if raw.shape == (like.sizes["time"],):
        data = np.broadcast_to(raw[:, None, None], like.shape)
        return xr.DataArray(data, dims=like.dims, coords=like.coords, name=name)
    if raw.shape == like.shape:
        return xr.DataArray(raw, dims=like.dims, coords=like.coords, name=name)
    raise ValueError(f"{name} must be scalar, a time vector, a matching DataArray, or shape {like.shape}")


def _nearest_index_labels(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    if coarse.size == 0:
        raise ValueError("coarse coordinate array is empty")
    distances = np.abs(fine[:, None] - coarse[None, :])
    return np.argmin(distances, axis=1)
