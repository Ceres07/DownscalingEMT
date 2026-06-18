from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class EMTConfig:
    """Configuration for an Equilibrium Moisture from Topography surrogate.

    Coleman and Niemann's EMT model uses spatial-average moisture as the
    changing catchment state and combines a lateral-flow index (LFI) with an
    ET index (ETI). This implementation keeps that structure and calibrates a
    compact anomaly model against point observations.
    """

    moisture_column: str
    time_column: str | None = None
    mean_moisture_column: str | None = None
    lfi_column: str = "lfi"
    eti_column: str = "eti"
    storage_column: str = "holding_capacity"
    vegetation_column: str = "vegetation_cover"
    infiltration_column: str = "infiltration_capacity"
    lower_bound: float | None = None
    upper_bound: float | None = None
    ridge: float = 1e-6
    lateral_wetness_power: float = 1.0
    lateral_saturation_power: float = 0.75
    radiative_dryness_power: float = 1.0
    clip_predictions: bool = True


@dataclass(frozen=True)
class EMTModel:
    """Calibrated EMT anomaly model."""

    config: EMTConfig
    coefficients: dict[str, float]
    feature_stats: dict[str, tuple[float, float]]
    feature_names: tuple[str, ...]
    mean_bounds: tuple[float, float]
    default_mean_moisture: float
    diagnostics: dict[str, float | int | str | None]

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict soil moisture for a point/grid table."""

        prepared = _prepare_mean_moisture(frame, self.config, self.default_mean_moisture)
        features = _build_feature_frame(prepared, self.config, self.feature_stats, self.mean_bounds)
        design = _design_matrix(features, self.feature_names)
        coef = np.array([self.coefficients[name] for name in ("intercept", *self.feature_names)])
        predicted = prepared["_emt_mean_moisture"].to_numpy(dtype=float) + design @ coef
        if self.config.clip_predictions:
            lower, upper = self._prediction_bounds()
            predicted = np.clip(predicted, lower, upper)
        return predicted

    def predict_dataset(
        self,
        terrain: xr.Dataset,
        mean_moisture: float | pd.DataFrame | xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Predict a full-grid EMT soil-moisture field."""

        if mean_moisture is None:
            mean_moisture = self.default_mean_moisture

        grid_frame = _terrain_to_frame(terrain, self.config)
        ny = terrain.sizes["y"]
        nx = terrain.sizes["x"]
        mean_col = self.config.mean_moisture_column or "spatial_mean_soil_moisture"

        if isinstance(mean_moisture, xr.DataArray):
            mean_grid = _align_mean_grid(mean_moisture, terrain)
            if "time" in mean_grid.dims:
                arrays = []
                for it in range(mean_grid.sizes["time"]):
                    table = grid_frame.copy()
                    table[mean_col] = np.asarray(mean_grid.isel(time=it).values, dtype=float).ravel()
                    arrays.append(self.predict_frame(table).reshape(ny, nx))
                return xr.DataArray(
                    np.stack(arrays).astype(np.float32),
                    dims=("time", "y", "x"),
                    coords={"time": mean_grid.time, "y": terrain.y, "x": terrain.x},
                    name="emt_soil_moisture",
                    attrs={"method": "Calibrated EMT anomaly model with spatial mean-moisture forcing"},
                )

            table = grid_frame.copy()
            table[mean_col] = np.asarray(mean_grid.values, dtype=float).ravel()
            predicted = self.predict_frame(table).reshape(ny, nx)
            return xr.DataArray(
                predicted.astype(np.float32),
                dims=("y", "x"),
                coords={"y": terrain.y, "x": terrain.x},
                name="emt_soil_moisture",
                attrs={"method": "Calibrated EMT anomaly model with spatial mean-moisture forcing"},
            )

        if isinstance(mean_moisture, pd.DataFrame):
            date_col = _find_date_column(mean_moisture)
            if mean_col not in mean_moisture:
                raise ValueError(f"mean_moisture DataFrame is missing {mean_col!r}")
            arrays = []
            times = []
            for _, row in mean_moisture.iterrows():
                table = grid_frame.copy()
                table[mean_col] = float(row[mean_col])
                arrays.append(self.predict_frame(table).reshape(ny, nx))
                times.append(pd.Timestamp(row[date_col]))
            return xr.DataArray(
                np.stack(arrays).astype(np.float32),
                dims=("time", "y", "x"),
                coords={"time": times, "y": terrain.y, "x": terrain.x},
                name="emt_soil_moisture",
                attrs={"method": "Calibrated EMT anomaly model"},
            )

        table = grid_frame.copy()
        table[mean_col] = float(mean_moisture)
        predicted = self.predict_frame(table).reshape(ny, nx)
        return xr.DataArray(
            predicted.astype(np.float32),
            dims=("y", "x"),
            coords={"y": terrain.y, "x": terrain.x},
            name="emt_soil_moisture",
            attrs={"method": "Calibrated EMT anomaly model"},
        )

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "coefficients": self.coefficients,
            "feature_stats": self.feature_stats,
            "feature_names": list(self.feature_names),
            "mean_bounds": list(self.mean_bounds),
            "default_mean_moisture": self.default_mean_moisture,
            "diagnostics": self.diagnostics,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, payload: dict) -> "EMTModel":
        return cls(
            config=EMTConfig(**payload["config"]),
            coefficients={str(k): float(v) for k, v in payload["coefficients"].items()},
            feature_stats={
                str(k): (float(v[0]), float(v[1])) for k, v in payload["feature_stats"].items()
            },
            feature_names=tuple(payload["feature_names"]),
            mean_bounds=(float(payload["mean_bounds"][0]), float(payload["mean_bounds"][1])),
            default_mean_moisture=float(payload["default_mean_moisture"]),
            diagnostics=payload.get("diagnostics", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "EMTModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def _prediction_bounds(self) -> tuple[float, float]:
        lower = self.config.lower_bound if self.config.lower_bound is not None else self.mean_bounds[0]
        upper = self.config.upper_bound if self.config.upper_bound is not None else self.mean_bounds[1]
        if lower >= upper:
            upper = lower + 1.0
        return lower, upper


@dataclass(frozen=True)
class EMTCalibrationResult:
    """Result bundle returned by ``calibrate_emt``."""

    model: EMTModel
    point_table: pd.DataFrame
    prediction_grid: xr.DataArray | None


def calibrate_emt(
    point_table: pd.DataFrame,
    config: EMTConfig,
    terrain: xr.Dataset | None = None,
) -> EMTCalibrationResult:
    """Calibrate an EMT anomaly model from sampled point observations."""

    prepared = _prepare_mean_moisture(point_table, config)
    _validate_configured_bounds(prepared, config)
    mean_bounds = _mean_bounds(prepared["_emt_mean_moisture"], config)
    feature_stats = _fit_feature_stats(prepared, config)
    features = _build_feature_frame(prepared, config, feature_stats, mean_bounds)
    feature_names = tuple(features.columns)
    design = _design_matrix(features, feature_names)

    observed = prepared[config.moisture_column].to_numpy(dtype=float)
    target = observed - prepared["_emt_mean_moisture"].to_numpy(dtype=float)
    valid = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    if int(valid.sum()) < 2:
        raise ValueError("At least two valid observations are required for EMT calibration")

    coef = _ridge_solve(design[valid], target[valid], ridge=config.ridge)
    coefficient_names = ("intercept", *feature_names)
    coefficients = {name: float(value) for name, value in zip(coefficient_names, coef)}

    provisional = EMTModel(
        config=config,
        coefficients=coefficients,
        feature_stats=feature_stats,
        feature_names=feature_names,
        mean_bounds=mean_bounds,
        default_mean_moisture=float(np.nanmean(prepared["_emt_mean_moisture"])),
        diagnostics={},
    )
    predicted = provisional.predict_frame(prepared)
    diagnostics = _diagnostics(prepared, observed, predicted, config)
    model = EMTModel(
        config=config,
        coefficients=coefficients,
        feature_stats=feature_stats,
        feature_names=feature_names,
        mean_bounds=mean_bounds,
        default_mean_moisture=provisional.default_mean_moisture,
        diagnostics=diagnostics,
    )

    output = prepared.copy()
    output["emt_prediction"] = predicted
    output["emt_residual"] = output[config.moisture_column] - output["emt_prediction"]
    prediction_grid = model.predict_dataset(terrain) if terrain is not None else None
    return EMTCalibrationResult(model=model, point_table=output, prediction_grid=prediction_grid)


def _prepare_mean_moisture(
    frame: pd.DataFrame,
    config: EMTConfig,
    default: float | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    mean_col = config.mean_moisture_column
    if mean_col and mean_col in out:
        out["_emt_mean_moisture"] = pd.to_numeric(out[mean_col], errors="coerce")
    elif config.time_column and config.time_column in out:
        time_key = pd.to_datetime(out[config.time_column]).dt.normalize()
        out["_emt_mean_moisture"] = out.groupby(time_key)[config.moisture_column].transform("mean")
    elif default is not None:
        out["_emt_mean_moisture"] = float(default)
    else:
        out["_emt_mean_moisture"] = float(pd.to_numeric(out[config.moisture_column], errors="coerce").mean())
    return out


def _fit_feature_stats(frame: pd.DataFrame, config: EMTConfig) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in _raw_feature_columns(frame, config):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        stats[column] = (mean, std)
    return stats


def _build_feature_frame(
    frame: pd.DataFrame,
    config: EMTConfig,
    stats: dict[str, tuple[float, float]],
    mean_bounds: tuple[float, float],
) -> pd.DataFrame:
    rel = _relative_mean(frame["_emt_mean_moisture"].to_numpy(dtype=float), mean_bounds)
    w_lateral = (rel**config.lateral_wetness_power) * ((1.0 - rel) ** config.lateral_saturation_power)
    w_radiative = (1.0 - rel) ** config.radiative_dryness_power

    columns: dict[str, np.ndarray] = {}
    if config.lfi_column in stats:
        lfi = _z(frame[config.lfi_column], stats[config.lfi_column])
        columns["lateral_lfi"] = w_lateral * lfi
    if config.eti_column in stats:
        eti = _z(frame[config.eti_column], stats[config.eti_column])
        columns["radiative_eti"] = w_radiative * eti
    if config.storage_column in stats:
        columns["soil_storage"] = _z(frame[config.storage_column], stats[config.storage_column])
    if config.vegetation_column in stats:
        veg = _z(frame[config.vegetation_column], stats[config.vegetation_column])
        columns["vegetation_radiative"] = w_radiative * veg
    if config.infiltration_column in stats:
        columns["infiltration_lateral"] = w_lateral * _z(
            frame[config.infiltration_column],
            stats[config.infiltration_column],
        )
    if "antecedent_precip" in frame and config.lfi_column in stats:
        pulse = 1.0 - np.exp(-np.maximum(frame["antecedent_precip"].to_numpy(dtype=float), 0.0) / 20.0)
        columns["rainfall_lfi"] = pulse * _z(frame[config.lfi_column], stats[config.lfi_column])
    if "potential_et" in frame and config.eti_column in stats:
        pet = np.maximum(frame["potential_et"].to_numpy(dtype=float), 0.0)
        pet_scale = pet / (pet + 5.0)
        columns["pet_eti"] = pet_scale * _z(frame[config.eti_column], stats[config.eti_column])

    if not columns:
        raise ValueError("No EMT covariate columns are available. Expected LFI/ETI or optional covariates.")
    return pd.DataFrame(columns, index=frame.index)


def _raw_feature_columns(frame: pd.DataFrame, config: EMTConfig) -> tuple[str, ...]:
    candidates = (
        config.lfi_column,
        config.eti_column,
        config.storage_column,
        config.vegetation_column,
        config.infiltration_column,
    )
    return tuple(column for column in candidates if column in frame)


def _design_matrix(features: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    aligned = pd.DataFrame(index=features.index)
    for name in feature_names:
        if name in features:
            aligned[name] = features[name]
        else:
            aligned[name] = 0.0
    values = aligned.to_numpy(dtype=float)
    return np.column_stack([np.ones(len(features), dtype=float), values])


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float) * max(ridge, 0.0)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _mean_bounds(values: pd.Series, config: EMTConfig) -> tuple[float, float]:
    lower = config.lower_bound if config.lower_bound is not None else float(np.nanmin(values))
    upper = config.upper_bound if config.upper_bound is not None else float(np.nanmax(values))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        center = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0
        lower = center - 0.5
        upper = center + 0.5
    return float(lower), float(upper)


def _validate_configured_bounds(frame: pd.DataFrame, config: EMTConfig) -> None:
    if config.lower_bound is None and config.upper_bound is None:
        return

    observed = pd.to_numeric(frame[config.moisture_column], errors="coerce").to_numpy(dtype=float)
    mean_moisture = frame["_emt_mean_moisture"].to_numpy(dtype=float)
    observed_min = float(np.nanmin(observed))
    observed_max = float(np.nanmax(observed))
    mean_min = float(np.nanmin(mean_moisture))
    mean_max = float(np.nanmax(mean_moisture))

    messages = []
    if config.lower_bound is not None and observed_min < config.lower_bound:
        messages.append(
            f"min observed {config.moisture_column} is {observed_min:.6g}, below lower_bound {config.lower_bound:.6g}"
        )
    if config.upper_bound is not None and observed_max > config.upper_bound:
        messages.append(
            f"max observed {config.moisture_column} is {observed_max:.6g}, above upper_bound {config.upper_bound:.6g}"
        )
    if config.lower_bound is not None and mean_min < config.lower_bound:
        messages.append(
            f"min spatial mean is {mean_min:.6g}, below lower_bound {config.lower_bound:.6g}"
        )
    if config.upper_bound is not None and mean_max > config.upper_bound:
        messages.append(
            f"max spatial mean is {mean_max:.6g}, above upper_bound {config.upper_bound:.6g}"
        )

    if messages:
        detail = "; ".join(messages)
        raise ValueError(
            "EMT moisture bounds must use the same units as the moisture column. "
            f"{detail}. For example, use bounds like 0-30 for millimetres, 0-60 for percent, "
            "or convert volumetric fraction observations to 0-0.6 before using those bounds."
        )


def _relative_mean(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    scale = upper - lower
    if scale <= 0:
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - lower) / scale, 0.0, 1.0)


def _z(values: pd.Series, stats: tuple[float, float]) -> np.ndarray:
    mean, std = stats
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return (arr - mean) / std


def _diagnostics(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    config: EMTConfig,
) -> dict[str, float | int | str | None]:
    residual = observed - predicted
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = observed[valid]
    pred = predicted[valid]
    res = residual[valid]
    sse = float(np.sum((obs - pred) ** 2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = 1.0 - sse / sst if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean(res**2)))
    mae = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))

    spatial_nsce = np.nan
    if config.time_column and config.time_column in frame:
        scores = []
        grouped = frame.assign(_obs=observed, _pred=predicted).groupby(
            pd.to_datetime(frame[config.time_column]).dt.normalize()
        )
        for _, group in grouped:
            if len(group) < 2:
                continue
            group_obs = group["_obs"].to_numpy(dtype=float)
            group_pred = group["_pred"].to_numpy(dtype=float)
            denom = np.sum((group_obs - np.mean(group_obs)) ** 2)
            if denom > 0:
                scores.append(1.0 - float(np.sum((group_obs - group_pred) ** 2)) / float(denom))
        if scores:
            spatial_nsce = float(np.mean(scores))

    return {
        "n_observations": int(valid.sum()),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "nse": float(nse) if np.isfinite(nse) else None,
        "average_spatial_nsce": spatial_nsce if np.isfinite(spatial_nsce) else None,
        "moisture_column": config.moisture_column,
        "time_column": config.time_column,
    }


def _terrain_to_frame(terrain: xr.Dataset, config: EMTConfig) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for column in _raw_feature_columns_from_dataset(terrain, config):
        data[column] = np.asarray(terrain[column].values, dtype=float).ravel()
    return pd.DataFrame(data)


def _raw_feature_columns_from_dataset(terrain: xr.Dataset, config: EMTConfig) -> tuple[str, ...]:
    candidates = (
        config.lfi_column,
        config.eti_column,
        config.storage_column,
        config.vegetation_column,
        config.infiltration_column,
    )
    return tuple(column for column in candidates if column in terrain)


def _find_date_column(frame: pd.DataFrame) -> str:
    for column in ("date", "time", "YYYY-MM-DD", "datetime"):
        if column in frame:
            return column
    raise ValueError("Could not find a date/time column in mean_moisture DataFrame")


def _align_mean_grid(mean_moisture: xr.DataArray, terrain: xr.Dataset) -> xr.DataArray:
    if {"x", "y"}.issubset(mean_moisture.dims):
        if not np.array_equal(mean_moisture.x.values, terrain.x.values) or not np.array_equal(
            mean_moisture.y.values,
            terrain.y.values,
        ):
            return mean_moisture.interp(x=terrain.x, y=terrain.y, method="nearest")
        return mean_moisture

    if mean_moisture.size == 1:
        return xr.full_like(terrain[next(iter(terrain.data_vars))], float(mean_moisture), dtype=float)

    raise ValueError("mean_moisture DataArray must have x/y dimensions or contain a scalar value")
