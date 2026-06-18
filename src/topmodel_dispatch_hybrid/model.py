from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray | float


@dataclass(frozen=True)
class Covariates:
    """Fine-grid covariates used as conditional wetness priors."""

    twi: np.ndarray
    hli: np.ndarray
    holding_capacity: np.ndarray
    vegetation_cover: np.ndarray | None = None
    infiltration_capacity: np.ndarray | None = None


@dataclass(frozen=True)
class WeatherForcing:
    """Coarse-grid or scalar weather forcing used to infer hydrological state."""

    antecedent_precip: ArrayLike = 0.0
    potential_et: ArrayLike = 0.0


@dataclass(frozen=True)
class DisaggregationConfig:
    """Rule-based parameters for regression-less SMIPS redistribution."""

    subgrid_factor: int = 33
    wilting_point: float = 0.08
    field_capacity: float = 0.35
    saturation: float = 0.50
    connectivity_threshold: float = 0.58
    connectivity_steepness: float = 9.0
    precip_scale: float = 20.0
    et_scale: float = 5.0
    redistribution_strength: float = 0.28
    min_terrain_weight: float = 0.05
    max_anomaly_fraction: float = 0.45
    preserve_mean_tolerance: float = 1e-8


@dataclass(frozen=True)
class DisaggregationResult:
    soil_moisture_30m: np.ndarray
    relative_wetness_score: np.ndarray
    connectivity: np.ndarray
    weights: dict[str, np.ndarray]


def disaggregate_smips(
    smips_1km: np.ndarray,
    covariates: Covariates,
    weather: WeatherForcing | None = None,
    config: DisaggregationConfig | None = None,
) -> DisaggregationResult:
    """Disaggregate coarse SMIPS soil moisture with state-dependent priors.

    The output preserves each coarse-pixel mean. Fine-grid covariates determine
    only the bounded anomaly around that coarse mean.
    """

    weather = weather or WeatherForcing()
    config = config or DisaggregationConfig()

    smips = np.asarray(smips_1km, dtype=float)
    if smips.ndim != 2:
        raise ValueError("smips_1km must be a 2D array")

    _validate_shapes(smips, covariates, config.subgrid_factor)

    fine_shape = covariates.twi.shape
    out = np.full(fine_shape, np.nan, dtype=float)
    score_out = np.full(fine_shape, np.nan, dtype=float)
    connectivity = np.full(smips.shape, np.nan, dtype=float)

    weight_maps = {
        "terrain": np.full(smips.shape, np.nan, dtype=float),
        "soil_storage": np.full(smips.shape, np.nan, dtype=float),
        "vegetation": np.full(smips.shape, np.nan, dtype=float),
        "evaporative_exposure": np.full(smips.shape, np.nan, dtype=float),
        "infiltration": np.full(smips.shape, np.nan, dtype=float),
    }

    precip = _coarse_array(weather.antecedent_precip, smips.shape, "antecedent_precip")
    pet = _coarse_array(weather.potential_et, smips.shape, "potential_et")

    for row in range(smips.shape[0]):
        for col in range(smips.shape[1]):
            block = _block_slice(row, col, config.subgrid_factor)
            mean_sm = smips[row, col]
            if not np.isfinite(mean_sm):
                continue

            state = _hydrological_state(mean_sm, precip[row, col], pet[row, col], config)
            weights = _state_dependent_weights(state, config)
            for name, value in weights.items():
                weight_maps[name][row, col] = value
            connectivity[row, col] = state["connectivity"]

            score = _relative_wetness_score(covariates, block, weights)
            anomaly = _normalised_anomaly(score)
            amplitude = _redistribution_amplitude(mean_sm, state, config)
            values = mean_sm + amplitude * anomaly
            values = _clip_preserve_mean(
                values,
                target_mean=mean_sm,
                lower=config.wilting_point,
                upper=config.saturation,
                tolerance=config.preserve_mean_tolerance,
            )

            out[block] = values
            score_out[block] = score

    return DisaggregationResult(
        soil_moisture_30m=out,
        relative_wetness_score=score_out,
        connectivity=connectivity,
        weights=weight_maps,
    )


def _validate_shapes(smips: np.ndarray, covariates: Covariates, factor: int) -> None:
    expected = (smips.shape[0] * factor, smips.shape[1] * factor)
    for name in (
        "twi",
        "hli",
        "holding_capacity",
        "vegetation_cover",
        "infiltration_capacity",
    ):
        value = getattr(covariates, name)
        if value is not None and np.asarray(value).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")


def _block_slice(row: int, col: int, factor: int) -> tuple[slice, slice]:
    return (
        slice(row * factor, (row + 1) * factor),
        slice(col * factor, (col + 1) * factor),
    )


def _coarse_array(value: ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    return arr


def _hydrological_state(
    mean_sm: float,
    antecedent_precip: float,
    potential_et: float,
    config: DisaggregationConfig,
) -> dict[str, float]:
    wetness = np.clip(
        (mean_sm - config.wilting_point) / (config.field_capacity - config.wilting_point),
        0.0,
        1.0,
    )
    rainfall_pulse = 1.0 - np.exp(-max(antecedent_precip, 0.0) / config.precip_scale)
    drydown = max(potential_et, 0.0) / (max(potential_et, 0.0) + config.et_scale)
    activation = max(wetness, rainfall_pulse)
    connectivity = _sigmoid(
        config.connectivity_steepness * (activation - config.connectivity_threshold)
    )
    return {
        "wetness": float(wetness),
        "rainfall_pulse": float(rainfall_pulse),
        "drydown": float(drydown),
        "connectivity": float(connectivity),
    }


def _state_dependent_weights(
    state: dict[str, float],
    config: DisaggregationConfig,
) -> dict[str, float]:
    connected = state["connectivity"]
    drydown = state["drydown"]
    pulse = state["rainfall_pulse"]

    return {
        "terrain": config.min_terrain_weight + 0.70 * connected,
        "soil_storage": 0.45 + 0.35 * (1.0 - connected),
        "vegetation": 0.15 + 0.35 * (1.0 - connected) + 0.10 * drydown,
        "evaporative_exposure": 0.15 + 0.45 * drydown * (1.0 - 0.5 * connected),
        "infiltration": 0.10 + 0.35 * pulse * (1.0 - connected),
    }


def _relative_wetness_score(
    covariates: Covariates,
    block: tuple[slice, slice],
    weights: dict[str, float],
) -> np.ndarray:
    twi = _standardise(covariates.twi[block])
    hli = _standardise(covariates.hli[block])
    storage = _standardise(covariates.holding_capacity[block])
    cover = _optional_standardise(covariates.vegetation_cover, block)
    infiltration = _optional_standardise(covariates.infiltration_capacity, block)

    return (
        weights["terrain"] * twi
        + weights["soil_storage"] * storage
        + weights["vegetation"] * cover
        + weights["infiltration"] * infiltration
        - weights["evaporative_exposure"] * hli
    )


def _optional_standardise(
    array: np.ndarray | None,
    block: tuple[slice, slice],
) -> np.ndarray:
    if array is None:
        return np.zeros((block[0].stop - block[0].start, block[1].stop - block[1].start))
    return _standardise(array[block])


def _standardise(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std < 1e-12:
        return np.zeros_like(arr, dtype=float)
    out = (arr - mean) / std
    return np.where(finite, out, 0.0)


def _normalised_anomaly(score: np.ndarray) -> np.ndarray:
    centered = score - np.nanmean(score)
    scale = np.nanmax(np.abs(centered))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(score, dtype=float)
    return centered / scale


def _redistribution_amplitude(
    mean_sm: float,
    state: dict[str, float],
    config: DisaggregationConfig,
) -> float:
    lower_room = max(mean_sm - config.wilting_point, 0.0)
    upper_room = max(config.saturation - mean_sm, 0.0)
    room = min(lower_room, upper_room)
    state_multiplier = 0.45 + 0.55 * state["connectivity"]
    return min(
        config.redistribution_strength * state_multiplier,
        config.max_anomaly_fraction * room,
    )


def _clip_preserve_mean(
    values: np.ndarray,
    target_mean: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> np.ndarray:
    clipped = np.clip(values, lower, upper)
    for _ in range(20):
        residual = target_mean - float(np.nanmean(clipped))
        if abs(residual) <= tolerance:
            break
        if residual > 0:
            adjustable = clipped < upper - tolerance
        else:
            adjustable = clipped > lower + tolerance
        if not np.any(adjustable):
            break
        clipped[adjustable] += residual / adjustable.mean()
        clipped = np.clip(clipped, lower, upper)
    return clipped


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

