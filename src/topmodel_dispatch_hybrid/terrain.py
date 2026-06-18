from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TerrainBuildConfig:
    """Options for deriving terrain indices from an AOI DEM."""

    bbox: tuple[float, float, float, float] | None = None
    bbox_crs: str | None = "EPSG:4326"
    latitude: float | None = None
    output_path: str | Path | None = None


def build_terrain_covariates_from_dem(
    dem_path: str | Path,
    config: TerrainBuildConfig | None = None,
) -> xr.Dataset:
    """Build DEM-derived covariates used by the EMT and redistribution models.

    The returned dataset includes ``dem``, ``slope``, ``aspect``,
    ``flow_acc``, ``twi`` and ``hli`` on the DEM grid. ``bbox`` is interpreted
    in ``bbox_crs`` and transformed to the DEM CRS when rasterio can do so.
    """

    config = config or TerrainBuildConfig()
    dem, transform, crs = _read_dem(dem_path, config.bbox, config.bbox_crs)
    dem_filled = _fill_nan_with_median(dem)

    cell_x = abs(float(transform.a))
    cell_y = abs(float(transform.e))
    slope = calculate_slope(dem_filled, cell_x=cell_x, cell_y=cell_y)
    aspect = calculate_aspect(dem_filled, cell_x=cell_x, cell_y=cell_y)
    flow_acc = calculate_d8_flow_accumulation(dem_filled)
    twi = calculate_twi(flow_acc, slope)
    lfi = twi.copy()
    latitude = config.latitude if config.latitude is not None else _latitude_from_transform(transform, dem.shape)
    hli = calculate_hli(slope, aspect, latitude)
    eti = 1.0 - hli

    y, x = _coords_from_transform(transform, dem.shape)
    ds = xr.Dataset(
        data_vars={
            "dem": (("y", "x"), dem.astype(np.float32), {"units": "m"}),
            "slope": (("y", "x"), slope.astype(np.float32), {"units": "degrees"}),
            "aspect": (("y", "x"), aspect.astype(np.float32), {"units": "degrees"}),
            "flow_acc": (("y", "x"), flow_acc.astype(np.float32), {"units": "cells"}),
            "twi": (("y", "x"), twi.astype(np.float32)),
            "lfi": (("y", "x"), lfi.astype(np.float32), {"long_name": "Lateral flow index proxy"}),
            "hli": (("y", "x"), hli.astype(np.float32)),
            "eti": (("y", "x"), eti.astype(np.float32), {"long_name": "ET index proxy"}),
        },
        coords={"y": y, "x": x},
        attrs={
            "crs": str(crs) if crs is not None else "",
            "source_dem": str(dem_path),
            "method": "Local DEM-derived terrain covariates",
        },
    )

    if config.output_path is not None:
        Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(config.output_path)

    return ds


def calculate_slope(dem: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Calculate slope angle in degrees from a DEM array."""

    gradient_y, gradient_x = np.gradient(dem, cell_y, cell_x)
    slope = np.degrees(np.arctan(np.sqrt(gradient_x**2 + gradient_y**2)))
    return np.where(np.isfinite(dem), slope, np.nan)


def calculate_aspect(dem: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Calculate aspect in degrees, where 0/360 is north and 90 is east."""

    gradient_y, gradient_x = np.gradient(dem, cell_y, cell_x)
    aspect = np.degrees(np.arctan2(-gradient_x, gradient_y))
    aspect = np.where(aspect < 0, aspect + 360.0, aspect)
    return np.where(np.isfinite(dem), aspect, np.nan)


def calculate_d8_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Approximate D8 flow accumulation with a single downslope receiver.

    This lightweight implementation is intended for paddock/AOI-scale DEMs.
    It follows the steepest lower neighbour and accumulates cells in descending
    elevation order, so no external hydrology library is required.
    """

    filled = _fill_nan_with_median(dem)
    ny, nx = filled.shape
    receiver = np.full((ny, nx, 2), -1, dtype=np.int32)
    offsets = (
        (-1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, np.sqrt(2.0)),
    )

    for row in range(ny):
        for col in range(nx):
            if not np.isfinite(dem[row, col]):
                continue
            best_drop = 0.0
            best = (-1, -1)
            here = filled[row, col]
            for dy, dx, distance in offsets:
                rr = row + dy
                cc = col + dx
                if rr < 0 or rr >= ny or cc < 0 or cc >= nx or not np.isfinite(dem[rr, cc]):
                    continue
                drop = (here - filled[rr, cc]) / distance
                if drop > best_drop:
                    best_drop = drop
                    best = (rr, cc)
            receiver[row, col] = best

    acc = np.where(np.isfinite(dem), 1.0, np.nan)
    order = np.argsort(filled.ravel())[::-1]
    for flat in order:
        row, col = divmod(int(flat), nx)
        if not np.isfinite(acc[row, col]):
            continue
        rr, cc = receiver[row, col]
        if rr >= 0:
            acc[rr, cc] += acc[row, col]
    return acc


def calculate_twi(flow_acc: np.ndarray, slope_degrees: np.ndarray) -> np.ndarray:
    """Calculate topographic wetness index as ln(flow accumulation / tan slope)."""

    tan_slope = np.tan(np.radians(slope_degrees))
    tan_slope = np.where(tan_slope <= 1e-6, 1e-6, tan_slope)
    ratio = np.asarray(flow_acc, dtype=float) / tan_slope
    ratio = np.where(ratio <= 0, np.nan, ratio)
    twi = np.log(ratio)
    return np.where(np.isfinite(twi), twi, np.nan)


def calculate_hli(slope_degrees: np.ndarray, aspect_degrees: np.ndarray, latitude: float) -> np.ndarray:
    """Calculate the McCune-Keon heat load index used as an exposure proxy."""

    slope_rad = np.radians(slope_degrees)
    aspect = np.asarray(aspect_degrees, dtype=float)
    lat_rad = np.radians(latitude)
    folded_aspect = np.abs(180.0 - np.abs(aspect - 225.0))
    folded_rad = np.radians(folded_aspect)
    hli = np.exp(
        -1.467
        + 1.582 * np.cos(lat_rad) * np.cos(slope_rad)
        - 1.5 * np.cos(folded_rad) * np.sin(slope_rad) * np.sin(lat_rad)
        - 0.262 * np.sin(lat_rad) * np.sin(slope_rad)
        + 0.607 * np.sin(folded_rad) * np.sin(slope_rad)
    )
    return np.clip(hli, 0.0, 1.0)


def _read_dem(
    dem_path: str | Path,
    bbox: tuple[float, float, float, float] | None,
    bbox_crs: str | None,
):
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds
    except ImportError as exc:
        raise ImportError("build_terrain_covariates_from_dem requires rasterio") from exc

    with rasterio.open(dem_path) as src:
        window = None
        if bbox is not None:
            bounds = bbox
            if bbox_crs and src.crs and str(src.crs) != str(bbox_crs):
                bounds = transform_bounds(bbox_crs, src.crs, *bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()

        data = src.read(1, window=window, masked=True).astype("float32")
        transform = src.window_transform(window) if window is not None else src.transform
        crs = src.crs

    dem = np.asarray(data.filled(np.nan), dtype=np.float32)
    if dem.size == 0:
        raise ValueError("DEM read produced an empty AOI")
    return dem, transform, crs


def _fill_nan_with_median(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if np.isfinite(arr).any():
        fill = float(np.nanmedian(arr))
    else:
        fill = 0.0
    arr[~np.isfinite(arr)] = fill
    return arr


def _coords_from_transform(transform, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = np.arange(nx, dtype=float) * transform.a + transform.c + transform.a / 2.0
    y = np.arange(ny, dtype=float) * transform.e + transform.f + transform.e / 2.0
    return y, x


def _latitude_from_transform(transform, shape: tuple[int, int]) -> float:
    y, _ = _coords_from_transform(transform, shape)
    if y.size == 0:
        return 0.0
    return float(np.nanmean(y))
