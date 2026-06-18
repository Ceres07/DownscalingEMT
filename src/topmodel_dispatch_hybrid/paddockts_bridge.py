from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
import xarray as xr


DEFAULT_PADDOCKTS_PATHS = (
    Path("/Volumes/Dmitry_work/borevitz_projects/paddock-ts-local"),
    Path("/Users/dmitrygrishin/borevitz_projects/paddock-ts-local"),
)


def configure_paddockts(path: str | Path | None = None) -> Path:
    """Put a local PaddockTS checkout on ``sys.path`` and return its path."""

    candidates = [Path(path)] if path is not None else list(DEFAULT_PADDOCKTS_PATHS)
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError("Could not find paddock-ts-local. Pass paddockts_path explicitly.")


def query_from_bbox(
    bbox: tuple[float, float, float, float] | list[float],
    start: date,
    end: date,
    stub: str,
    paddockts_path: str | Path | None = None,
):
    """Create a PaddockTS Query for a bbox/date range."""

    configure_paddockts(paddockts_path)
    from PaddockTS.query import Query

    return Query(bbox=list(bbox), start=start, end=end, stub=stub)


def query_from_lat_lon(
    lat: float,
    lon: float,
    buffer_km: float,
    start: date,
    end: date,
    stub: str,
    paddockts_path: str | Path | None = None,
):
    """Create a PaddockTS Query from a point and square buffer."""

    configure_paddockts(paddockts_path)
    from PaddockTS.query import Query

    return Query.from_lat_lon(lat=lat, lon=lon, buffer_km=buffer_km, start=start, end=end, stub=stub)


def load_or_download_terrain_covariates(
    query,
    output_path: str | Path,
    *,
    reload: bool = False,
    paddockts_path: str | Path | None = None,
) -> xr.Dataset:
    """Download a Copernicus DEM through PaddockTS and compute EMT terrain indices."""

    configure_paddockts(paddockts_path)
    output_path = Path(output_path)
    if output_path.exists() and not reload:
        return xr.open_dataset(output_path).load()

    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import (
        download_terrain,
        get_filename,
    )
    from PaddockTS.Environmental.TerrainTiles.utils import (
        calculate_aspect,
        calculate_hli,
        calculate_slope,
        calculate_twi,
        pysheds_accumulation,
    )

    terrain_tif = get_filename(query)
    if not Path(terrain_tif).exists() or reload:
        download_terrain(query)

    _, dem_filled, _, acc = pysheds_accumulation(terrain_tif)
    slope = calculate_slope(terrain_tif)
    aspect = calculate_aspect(terrain_tif)
    twi = calculate_twi(acc, slope)
    hli = calculate_hli(slope, aspect, query.centre_lat)

    with rasterio.open(terrain_tif) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    ny, nx = dem.shape
    x = np.arange(nx) * transform.a + transform.c + transform.a / 2.0
    y = np.arange(ny) * transform.e + transform.f + transform.e / 2.0
    ds = xr.Dataset(
        {
            "dem": (("y", "x"), dem.astype(np.float32)),
            "dem_filled": (("y", "x"), np.asarray(dem_filled, dtype=np.float32)),
            "slope": (("y", "x"), slope.astype(np.float32)),
            "aspect": (("y", "x"), aspect.astype(np.float32)),
            "flow_acc": (("y", "x"), np.asarray(acc, dtype=np.float32)),
            "twi": (("y", "x"), twi.astype(np.float32)),
            "lfi": (("y", "x"), twi.astype(np.float32)),
            "hli": (("y", "x"), hli.astype(np.float32)),
            "eti": (("y", "x"), (1.0 - hli).astype(np.float32)),
        },
        coords={"y": y, "x": x},
        attrs={"crs": str(crs), "source": "PaddockTS TerrainTiles"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    return ds


def load_or_download_silo(
    query,
    output_path: str | Path,
    *,
    reload: bool = False,
    paddockts_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download SILO climate through PaddockTS and cache it as CSV."""

    configure_paddockts(paddockts_path)
    output_path = Path(output_path)
    if output_path.exists() and not reload:
        return pd.read_csv(output_path, parse_dates=["YYYY-MM-DD"])

    from PaddockTS.Environmental.SILO.download_silo import download_silo

    climate = download_silo(query)
    climate["YYYY-MM-DD"] = pd.to_datetime(climate["YYYY-MM-DD"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    climate.to_csv(output_path, index=False)
    return climate
