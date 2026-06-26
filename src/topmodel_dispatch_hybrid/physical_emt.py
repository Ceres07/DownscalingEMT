from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class PhysicalEMTParameters:
    """Coleman-Niemann EMT parameter set.

    Units follow Coleman and Niemann (2013) where possible. The implementation
    uses a terrain-derived LFI proxy, so some process rates are effective
    catchment parameters rather than independently identifiable measurements.
    """

    ks_v: float
    ks_h: float
    porosity: float
    eta_h: float
    eta_v: float
    beta_r: float
    beta_a: float
    alpha: float
    z0: float
    curvature_min: float
    epsilon: float
    pet: float


@dataclass(frozen=True)
class PhysicalEMTCalibration:
    parameters: PhysicalEMTParameters
    diagnostics: dict[str, float | int | str | None]
    water_depth_mm: float
    date_metrics: pd.DataFrame

    def to_json(self, path: str | Path) -> None:
        payload = {
            "parameters": asdict(self.parameters),
            "diagnostics": self.diagnostics,
            "water_depth_mm": self.water_depth_mm,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


PAPER_PARAMETER_ROWS = (
    ("Ks,v", "50-50,000 mm day^-1"),
    ("Ks,h", "50-50,000 mm day^-1; constrained Ks,h >= Ks,v"),
    ("porosity", "0.25-0.70 m3 m^-3"),
    ("eta_h", "1-25"),
    ("eta_v", "4-25"),
    ("beta_r", "0.2-5.0; constrained beta_r <= eta_h"),
    ("beta_a", "0.2-5.0"),
    ("alpha", "0.26 fixed"),
    ("z0", "calibrated effective soil thickness, m"),
    ("curvature_min", "-1e6 to catchment minimum m^-1"),
    ("epsilon", "1-3"),
)


def calibrate_physical_emt(
    point_table: pd.DataFrame,
    terrain: xr.Dataset,
    *,
    theta_column: str = "Soil_moisture",
    theta_is_percent: bool = True,
    water_column: str = "Water_mm",
    time_column: str = "Date",
    pet: float | None = None,
    seed: int = 42,
    maxiter: int = 200,
    mean_sample_size: int | None = 20000,
) -> PhysicalEMTCalibration:
    """Calibrate the physical EMT parameter set by maximizing average NSCE."""

    from scipy.optimize import differential_evolution

    data = _prepare_point_data(
        point_table,
        terrain,
        theta_column=theta_column,
        theta_is_percent=theta_is_percent,
        water_column=water_column,
        time_column=time_column,
        mean_sample_size=mean_sample_size,
    )
    pet_value = float(pet) if pet is not None else _estimate_pet(point_table)

    def objective(vector: np.ndarray) -> float:
        params = _unpack_parameters(vector, pet=pet_value)
        penalty = _parameter_penalty(params, min_curvature=data.min_curvature)
        if penalty > 0:
            return penalty
        predicted = predict_physical_emt_points(data, params)
        metrics = score_predictions(data.frame, data.theta_obs, predicted, time_column=time_column)
        nsce = metrics["average_spatial_nsce"]
        if nsce is None or not np.isfinite(float(nsce)):
            return 1e6
        return 1.0 - float(nsce)

    bounds = [
        (np.log10(50.0), np.log10(50000.0)),
        (np.log10(50.0), np.log10(50000.0)),
        (0.25, 0.70),
        (1.0, 25.0),
        (4.0, 25.0),
        (0.2, 5.0),
        (0.2, 5.0),
        (np.log10(0.05), np.log10(2.0)),
        (-6.0, 6.0),
        (1.0, 3.0),
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=8,
        tol=1e-5,
        polish=True,
        updating="immediate",
        workers=1,
    )
    params = _unpack_parameters(result.x, pet=pet_value)
    predicted = predict_physical_emt_points(data, params)
    metrics = score_predictions(data.frame, data.theta_obs, predicted, time_column=time_column)
    date_metrics = score_predictions_by_date(data.frame, data.theta_obs, predicted, time_column=time_column)
    metrics.update(
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        optimizer_objective=float(result.fun),
        theta_column=theta_column,
        water_column=water_column,
        time_column=time_column,
    )
    return PhysicalEMTCalibration(
        parameters=params,
        diagnostics=metrics,
        water_depth_mm=data.water_depth_mm,
        date_metrics=date_metrics,
    )


def predict_physical_emt_points(data: "_PreparedPhysicalData", params: PhysicalEMTParameters) -> np.ndarray:
    ip_mean = _finite_mean(data.mean_hli)
    lfi = physical_lfi(
        flow_acc=data.flow_acc,
        slope_degrees=data.slope,
        curvature=data.curvature,
        eta_h=params.eta_h,
        epsilon=params.epsilon,
        curvature_min=params.curvature_min,
        specific_area_scale=data.specific_area_scale,
    )
    lfi_mean = _finite_mean(
        physical_lfi(
            flow_acc=data.mean_flow_acc,
            slope_degrees=data.mean_slope,
            curvature=data.mean_curvature,
            eta_h=params.eta_h,
            epsilon=params.epsilon,
            curvature_min=params.curvature_min,
            specific_area_scale=data.specific_area_scale,
        )
    )
    eti = physical_eti(data.hli, beta_r=params.beta_r, ip_mean=ip_mean)
    eti_mean = _finite_mean(physical_eti(data.mean_hli, beta_r=params.beta_r, ip_mean=ip_mean))
    return physical_emt_prediction(
        theta_bar=data.theta_bar,
        lfi=lfi,
        eti=eti,
        params=params,
        lfi_mean=lfi_mean,
        eti_mean=eti_mean,
    )


def predict_physical_emt_grid(
    terrain: xr.Dataset,
    params: PhysicalEMTParameters,
    theta_bar: xr.DataArray,
    normalization_labels: xr.DataArray | None = None,
) -> xr.DataArray:
    """Predict physical EMT theta over a terrain grid."""

    return physical_emt_grid_components(
        terrain,
        params,
        theta_bar,
        normalization_labels=normalization_labels,
    )["physical_emt_theta"]


def physical_emt_grid_components(
    terrain: xr.Dataset,
    params: PhysicalEMTParameters,
    theta_bar: xr.DataArray,
    normalization_labels: xr.DataArray | None = None,
) -> xr.Dataset:
    """Evaluate paper EMT equation 7/19 and process weights over a terrain grid."""

    theta_bar = _align_theta_bar(theta_bar, terrain)
    curvature = terrain_curvature(terrain)
    specific_area_scale = _specific_area_scale(terrain)
    lfi = physical_lfi(
        flow_acc=terrain["flow_acc"].values,
        slope_degrees=terrain["slope"].values,
        curvature=curvature.values,
        eta_h=params.eta_h,
        epsilon=params.epsilon,
        curvature_min=params.curvature_min,
        specific_area_scale=specific_area_scale,
    )
    ip_mean = _finite_mean(terrain["hli"].values)
    eti = physical_eti(terrain["hli"].values, beta_r=params.beta_r, ip_mean=ip_mean)
    lfi_mean, eti_mean = _index_means_for_grid(lfi, eti, normalization_labels)
    lfi_da = xr.DataArray(lfi, dims=("y", "x"), coords={"y": terrain.y, "x": terrain.x})
    eti_da = xr.DataArray(eti, dims=("y", "x"), coords={"y": terrain.y, "x": terrain.x})

    component_names = (
        "theta",
        "theta_g",
        "theta_l",
        "theta_r",
        "theta_a",
        "w_g",
        "w_l",
        "w_r",
        "w_a",
        "relative_w_g",
        "relative_w_l",
        "relative_w_r",
        "relative_w_a",
    )
    if "time" in theta_bar.dims:
        arrays: dict[str, list[np.ndarray]] = {name: [] for name in component_names}
        for it in range(theta_bar.sizes["time"]):
            components = physical_emt_components(
                theta_bar=theta_bar.isel(time=it).values,
                lfi=lfi_da.values,
                eti=eti_da.values,
                params=params,
                lfi_mean=lfi_mean,
                eti_mean=eti_mean,
            )
            for name in component_names:
                arrays[name].append(np.asarray(components[name], dtype=np.float32))
        data_vars = {
            _component_var_name(name): (
                ("time", "y", "x"),
                np.stack(values).astype(np.float32),
                _component_attrs(name),
            )
            for name, values in arrays.items()
        }
        ds = xr.Dataset(
            data_vars,
            coords={"time": theta_bar.time, "y": terrain.y, "x": terrain.x},
            attrs={"method": "Coleman-Niemann physical EMT equation 7/19"},
        )
    else:
        components = physical_emt_components(
            theta_bar=theta_bar.values,
            lfi=lfi_da.values,
            eti=eti_da.values,
            params=params,
            lfi_mean=lfi_mean,
            eti_mean=eti_mean,
        )
        ds = xr.Dataset(
            {
                _component_var_name(name): (
                    ("y", "x"),
                    np.asarray(components[name], dtype=np.float32),
                    _component_attrs(name),
                )
                for name in component_names
            },
            coords={"y": terrain.y, "x": terrain.x},
            attrs={"method": "Coleman-Niemann physical EMT equation 7/19"},
        )

    ds["lfi_index"] = lfi_da.astype(np.float32)
    ds["eti_index"] = eti_da.astype(np.float32)
    ds["lfi_mean"] = xr.DataArray(
        np.asarray(np.broadcast_to(lfi_mean, lfi.shape), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": terrain.y, "x": terrain.x},
    )
    ds["eti_mean"] = xr.DataArray(
        np.asarray(np.broadcast_to(eti_mean, eti.shape), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": terrain.y, "x": terrain.x},
    )
    if normalization_labels is not None:
        ds["normalization_label"] = normalization_labels.astype(np.int32)
    return ds


def physical_emt_prediction(
    theta_bar: np.ndarray | float,
    lfi: np.ndarray,
    eti: np.ndarray,
    params: PhysicalEMTParameters,
    lfi_mean: np.ndarray | float | None = None,
    eti_mean: np.ndarray | float | None = None,
) -> np.ndarray:
    """Explicit physical EMT estimate from weighted process-limit estimates."""

    return physical_emt_components(theta_bar, lfi, eti, params, lfi_mean=lfi_mean, eti_mean=eti_mean)["theta"]


def physical_emt_components(
    theta_bar: np.ndarray | float,
    lfi: np.ndarray,
    eti: np.ndarray,
    params: PhysicalEMTParameters,
    lfi_mean: np.ndarray | float | None = None,
    eti_mean: np.ndarray | float | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate equation 7/19 and return process estimates and weights.

    ``theta_bar`` is the spatial-average moisture state. ``lfi_mean`` is
    Coleman and Niemann's Lambda, and ``eti_mean`` is Pi. For SMIPS
    downscaling these can be supplied as coarse-tile mean grids so each SMIPS
    tile is mass-normalized independently.
    """

    theta_bar_arr = np.asarray(theta_bar, dtype=float)
    theta_bar_arr = np.clip(theta_bar_arr, 1e-6, params.porosity * 0.999)
    lfi_arr = np.asarray(lfi, dtype=float)
    eti_arr = np.asarray(eti, dtype=float)
    lambda_arr = _safe_positive_mean(lfi_arr if lfi_mean is None else lfi_mean)
    pi_arr = _safe_positive_mean(eti_arr if eti_mean is None else eti_mean)
    theta_g = theta_bar_arr
    theta_l = theta_bar_arr * lfi_arr / lambda_arr
    theta_r = theta_bar_arr * eti_arr / pi_arr
    theta_a = theta_bar_arr
    rel = np.clip(theta_bar_arr / params.porosity, 1e-6, 0.999)

    w_g = params.ks_v * rel**params.eta_v
    w_l = params.z0 * params.ks_h / (lambda_arr**params.eta_h) * rel**params.eta_h
    w_r = params.pet / ((1.0 + params.alpha) * (pi_arr**params.beta_r)) * rel**params.beta_r
    w_a = params.pet * params.alpha / (1.0 + params.alpha) * rel**params.beta_a
    denom = w_g + w_l + w_r + w_a
    pred = (w_g * theta_g + w_l * theta_l + w_r * theta_r + w_a * theta_a) / denom
    pred = np.clip(pred, 0.0, params.porosity)
    return {
        "theta": pred,
        "theta_g": theta_g,
        "theta_l": theta_l,
        "theta_r": theta_r,
        "theta_a": theta_a,
        "w_g": w_g,
        "w_l": w_l,
        "w_r": w_r,
        "w_a": w_a,
        "relative_w_g": w_g / denom,
        "relative_w_l": w_l / denom,
        "relative_w_r": w_r / denom,
        "relative_w_a": w_a / denom,
        "lfi_mean": lambda_arr,
        "eti_mean": pi_arr,
    }


def physical_lfi(
    flow_acc: np.ndarray,
    slope_degrees: np.ndarray,
    curvature: np.ndarray,
    eta_h: float,
    epsilon: float,
    curvature_min: float,
    specific_area_scale: float = 1.0,
) -> np.ndarray:
    slope = np.tan(np.radians(np.asarray(slope_degrees, dtype=float)))
    slope = np.clip(slope, 1e-4, None)
    area = np.clip(np.asarray(flow_acc, dtype=float) * float(specific_area_scale), 1.0, None)
    curv = np.asarray(curvature, dtype=float)
    denom = curvature_min - curv
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature_factor = curvature_min / denom
    curvature_factor = np.where(curvature_factor > 0, curvature_factor, np.nan)
    curvature_factor = np.clip(curvature_factor, 1e-3, 1e3)
    raw = (area / (slope**epsilon) * curvature_factor) ** (1.0 / eta_h)
    return np.where(np.isfinite(raw), raw, np.nan)


def physical_eti(hli: np.ndarray, beta_r: float, ip_mean: float | None = None) -> np.ndarray:
    ip = np.asarray(hli, dtype=float)
    scale = _finite_mean(ip) if ip_mean is None else float(ip_mean)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    ip = ip / scale
    ip = np.clip(ip, 1e-6, None)
    raw = (1.0 / ip) ** (1.0 / beta_r)
    return np.where(np.isfinite(raw), raw, np.nan)


def terrain_curvature(terrain: xr.Dataset) -> xr.DataArray:
    dem = terrain["dem"].astype(float)
    x = terrain.x.values.astype(float)
    y = terrain.y.values.astype(float)
    dx = abs(float(np.nanmedian(np.diff(x)))) if x.size > 1 else 1.0
    dy = abs(float(np.nanmedian(np.diff(y)))) if y.size > 1 else 1.0
    z = dem.values
    dzdy, dzdx = np.gradient(z, dy, dx)
    d2zdx2 = np.gradient(dzdx, dx, axis=1)
    d2zdy2 = np.gradient(dzdy, dy, axis=0)
    # Positive values represent convergent/topographic hollow locations.
    curvature = -(d2zdx2 + d2zdy2)
    return xr.DataArray(
        curvature.astype(np.float32),
        dims=("y", "x"),
        coords={"y": terrain.y, "x": terrain.x},
        name="curvature",
        attrs={"units": "m-1"},
    )


def score_predictions(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    time_column: str,
) -> dict[str, float | int | None]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = observed[valid]
    pred = predicted[valid]
    residual = obs - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    date_metrics = score_predictions_by_date(frame, observed, predicted, time_column=time_column)
    return {
        "n_observations": int(valid.sum()),
        "rmse_theta": float(np.sqrt(np.mean(residual**2))),
        "mae_theta": float(np.mean(np.abs(residual))),
        "bias_theta": float(np.mean(residual)),
        "nse": 1.0 - sse / sst if sst > 0 else None,
        "average_spatial_nsce": (
            float(date_metrics["spatial_nsce"].mean(skipna=True))
            if len(date_metrics)
            else None
        ),
    }


def score_predictions_by_date(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    time_column: str,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[time_column]).dt.normalize(),
            "observed": observed,
            "predicted": predicted,
        }
    )
    rows = []
    for date_value, group in data.groupby("date"):
        obs = group["observed"].to_numpy(dtype=float)
        pred = group["predicted"].to_numpy(dtype=float)
        valid = np.isfinite(obs) & np.isfinite(pred)
        obs = obs[valid]
        pred = pred[valid]
        residual = obs - pred
        sse = float(np.sum(residual**2))
        sst = float(np.sum((obs - np.mean(obs)) ** 2))
        rows.append(
            {
                "date": date_value.date().isoformat(),
                "n": int(valid.sum()),
                "rmse_theta": float(np.sqrt(np.mean(residual**2))) if valid.any() else np.nan,
                "nse": 1.0 - sse / sst if sst > 0 else np.nan,
                "spatial_nsce": 1.0 - sse / sst if sst > 0 else np.nan,
                "observed_mean_theta": float(np.mean(obs)) if valid.any() else np.nan,
                "predicted_mean_theta": float(np.mean(pred)) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_table1_style_parameter_set(
    calibration: PhysicalEMTCalibration,
    path: str | Path,
    catchment_name: str = "Esdale",
) -> Path:
    values = _parameter_table_values(calibration.parameters)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("| Parameter | Allowed Range | " + catchment_name + " |\n")
        handle.write("|---|---:|---:|\n")
        for parameter, allowed in PAPER_PARAMETER_ROWS:
            value = values[parameter]
            handle.write(f"| {parameter} | {allowed} | {_format_parameter(value)} |\n")
    return path


def write_table1_style_parameter_csv(
    calibration: PhysicalEMTCalibration,
    path: str | Path,
    catchment_name: str = "Esdale",
) -> Path:
    values = _parameter_table_values(calibration.parameters)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"Parameter": p, "Allowed Range": a, catchment_name: values[p]} for p, a in PAPER_PARAMETER_ROWS]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@dataclass(frozen=True)
class _PreparedPhysicalData:
    frame: pd.DataFrame
    theta_obs: np.ndarray
    theta_bar: np.ndarray
    flow_acc: np.ndarray
    slope: np.ndarray
    hli: np.ndarray
    curvature: np.ndarray
    mean_flow_acc: np.ndarray
    mean_slope: np.ndarray
    mean_hli: np.ndarray
    mean_curvature: np.ndarray
    specific_area_scale: float
    min_curvature: float
    water_depth_mm: float


def _prepare_point_data(
    point_table: pd.DataFrame,
    terrain: xr.Dataset,
    *,
    theta_column: str,
    theta_is_percent: bool,
    water_column: str,
    time_column: str,
    mean_sample_size: int | None,
) -> _PreparedPhysicalData:
    frame = point_table.copy()
    theta = pd.to_numeric(frame[theta_column], errors="coerce").to_numpy(dtype=float)
    if theta_is_percent:
        theta = theta / 100.0
    date_key = pd.to_datetime(frame[time_column]).dt.normalize()
    frame["_physical_theta"] = theta
    theta_bar = frame.groupby(date_key)["_physical_theta"].transform("mean").to_numpy(dtype=float)
    water = pd.to_numeric(frame[water_column], errors="coerce").to_numpy(dtype=float)
    depth = float(np.nanmedian(water / theta))

    curvature = terrain_curvature(terrain)
    mean_sample = _terrain_mean_sample(terrain, curvature, max_cells=mean_sample_size)
    x = xr.DataArray(frame["_terrain_x"].to_numpy(dtype=float), dims="observation")
    y = xr.DataArray(frame["_terrain_y"].to_numpy(dtype=float), dims="observation")
    sampled_curvature = curvature.sel(x=x, y=y, method="nearest").values
    return _PreparedPhysicalData(
        frame=frame,
        theta_obs=theta,
        theta_bar=theta_bar,
        flow_acc=frame["flow_acc"].to_numpy(dtype=float),
        slope=frame["slope"].to_numpy(dtype=float),
        hli=frame["hli"].to_numpy(dtype=float),
        curvature=np.asarray(sampled_curvature, dtype=float),
        mean_flow_acc=mean_sample["flow_acc"],
        mean_slope=mean_sample["slope"],
        mean_hli=mean_sample["hli"],
        mean_curvature=mean_sample["curvature"],
        specific_area_scale=_specific_area_scale(terrain),
        min_curvature=float(np.nanmin(curvature.values)),
        water_depth_mm=depth,
    )


def _unpack_parameters(vector: np.ndarray, pet: float) -> PhysicalEMTParameters:
    return PhysicalEMTParameters(
        ks_v=float(10.0 ** vector[0]),
        ks_h=float(10.0 ** vector[1]),
        porosity=float(vector[2]),
        eta_h=float(vector[3]),
        eta_v=float(vector[4]),
        beta_r=float(vector[5]),
        beta_a=float(vector[6]),
        alpha=0.26,
        z0=float(10.0 ** vector[7]),
        curvature_min=-float(10.0 ** vector[8]),
        epsilon=float(vector[9]),
        pet=float(pet),
    )


def _parameter_penalty(params: PhysicalEMTParameters, *, min_curvature: float | None = None) -> float:
    penalty = 0.0
    if params.ks_h < params.ks_v:
        penalty += 1e5 + (params.ks_v - params.ks_h)
    if params.beta_r > params.eta_h:
        penalty += 1e5 + (params.beta_r - params.eta_h)
    if min_curvature is not None and np.isfinite(min_curvature) and params.curvature_min >= min_curvature:
        penalty += 1e5 + (params.curvature_min - min_curvature + 1e-9) * 1e8
    return penalty


def _estimate_pet(point_table: pd.DataFrame) -> float:
    for column in ("Cumulative_ET_period", "potential_et", "et_short_crop", "evap_pan"):
        if column in point_table:
            values = pd.to_numeric(point_table[column], errors="coerce")
            if values.notna().any():
                value = float(values[values > 0].median())
                if np.isfinite(value) and value > 0:
                    return value
    return 4.0


def _align_theta_bar(theta_bar: xr.DataArray, terrain: xr.Dataset) -> xr.DataArray:
    if {"x", "y"}.issubset(theta_bar.dims):
        if not np.array_equal(theta_bar.x.values, terrain.x.values) or not np.array_equal(
            theta_bar.y.values,
            terrain.y.values,
        ):
            return theta_bar.interp(x=terrain.x, y=terrain.y, method="nearest")
    return theta_bar


def _component_var_name(name: str) -> str:
    if name == "theta":
        return "physical_emt_theta"
    return name


def _component_attrs(name: str) -> dict[str, str]:
    if name.startswith("theta") or name == "theta":
        return {"units": "m3 m-3"}
    if name.startswith("relative_w"):
        return {"units": "1", "long_name": "relative process weight"}
    if name.startswith("w_"):
        return {"units": "model process weight"}
    return {}


def _index_means_for_grid(
    lfi: np.ndarray,
    eti: np.ndarray,
    normalization_labels: xr.DataArray | None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    if normalization_labels is None:
        return _finite_mean(lfi), _finite_mean(eti)
    labels = np.asarray(normalization_labels.values, dtype=np.int64)
    return _mean_by_label(lfi, labels), _mean_by_label(eti, labels)


def _mean_by_label(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    labs = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(vals) & (labs >= 0)
    if not valid.any():
        return np.ones_like(vals, dtype=float)
    max_label = int(np.nanmax(labs[valid]))
    sums = np.bincount(labs[valid].ravel(), weights=vals[valid].ravel(), minlength=max_label + 1)
    counts = np.bincount(labs[valid].ravel(), minlength=max_label + 1)
    means = sums / np.maximum(counts, 1)
    global_mean = _finite_mean(vals)
    means = np.where(counts > 0, means, global_mean)
    out = np.full(vals.shape, global_mean, dtype=float)
    out[valid] = means[labs[valid]]
    return _safe_positive_mean(out)


def _terrain_mean_sample(
    terrain: xr.Dataset,
    curvature: xr.DataArray,
    *,
    max_cells: int | None,
) -> dict[str, np.ndarray]:
    total = terrain.sizes["x"] * terrain.sizes["y"]
    if max_cells is None or max_cells <= 0 or total <= max_cells:
        stride = 1
    else:
        stride = int(np.ceil(np.sqrt(total / max_cells)))
    slicer = (slice(None, None, stride), slice(None, None, stride))
    return {
        "flow_acc": np.asarray(terrain["flow_acc"].values[slicer], dtype=float).ravel(),
        "slope": np.asarray(terrain["slope"].values[slicer], dtype=float).ravel(),
        "hli": np.asarray(terrain["hli"].values[slicer], dtype=float).ravel(),
        "curvature": np.asarray(curvature.values[slicer], dtype=float).ravel(),
    }


def _specific_area_scale(terrain: xr.Dataset) -> float:
    x = terrain.x.values.astype(float)
    y = terrain.y.values.astype(float)
    dx = abs(float(np.nanmedian(np.diff(x)))) if x.size > 1 else 1.0
    dy = abs(float(np.nanmedian(np.diff(y)))) if y.size > 1 else dx
    return float(np.sqrt(dx * dy))


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(arr))
    if not np.isfinite(mean):
        return 1.0
    return mean


def _safe_positive_mean(values: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        value = float(arr)
        return value if np.isfinite(value) and value > 1e-12 else 1.0
    return np.where(np.isfinite(arr) & (arr > 1e-12), arr, 1.0)


def _parameter_table_values(params: PhysicalEMTParameters) -> dict[str, float]:
    return {
        "Ks,v": params.ks_v,
        "Ks,h": params.ks_h,
        "porosity": params.porosity,
        "eta_h": params.eta_h,
        "eta_v": params.eta_v,
        "beta_r": params.beta_r,
        "beta_a": params.beta_a,
        "alpha": params.alpha,
        "z0": params.z0,
        "curvature_min": params.curvature_min,
        "epsilon": params.epsilon,
    }


def _format_parameter(value: float) -> str:
    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.4e}"
    return f"{value:.6g}"
