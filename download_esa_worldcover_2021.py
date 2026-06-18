#!/usr/bin/env python3
"""
Download ESA WorldCover 2021 v200 land-cover COG tiles for a lon/lat point
and square buffer, clipping the result to the requested area.

Dataset:
  ESA WorldCover 10 m 2021 v200
  https://esa-worldcover.org/en/data-access

Notes:
- Map tiles are public Cloud-Optimized GeoTIFFs in EPSG:4326.
- Tiles are named from the lower-left corner of each 3 x 3 degree tile.
- The output is a categorical uint8 GeoTIFF. Class value 0 is no data.

Example:
  python download_esa_worldcover_2021.py \
    --lon 131.012945 --lat -13.147158 --buffer-m 5000 \
    --out worldcover_2021.tif
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Tuple

import rasterio
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.windows import WindowError, from_bounds, intersection


BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
TILE_SIZE_DEG = 3
EPSG_WGS84 = 4326
NODATA = 0

CLASS_NAMES = {
    0: "No data",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


def _format_lat(lat_origin: int) -> str:
    return f"S{abs(lat_origin):02d}" if lat_origin < 0 else f"N{lat_origin:02d}"


def _format_lon(lon_origin: int) -> str:
    return f"W{abs(lon_origin):03d}" if lon_origin < 0 else f"E{lon_origin:03d}"


def _tile_origin(value: float) -> int:
    return int(math.floor(value / TILE_SIZE_DEG) * TILE_SIZE_DEG)


def _tile_origins(min_value: float, max_value: float) -> range:
    """Return 3-degree lower-left tile origins intersecting [min, max]."""
    if max_value <= min_value:
        raise ValueError("Bounding box must have positive width/height.")

    epsilon = 1e-12
    start = _tile_origin(min_value)
    end = _tile_origin(max_value - epsilon)
    return range(start, end + TILE_SIZE_DEG, TILE_SIZE_DEG)


def make_bbox_lonlat(lon: float, lat: float, buffer_m: float) -> BBox:
    """Create a lon/lat bbox from a point and square buffer in metres."""
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("Latitude must be between -90 and 90.")
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive.")

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    )
    to_wgs84 = Transformer.from_crs(local_crs, EPSG_WGS84, always_xy=True)
    corners_m = [
        (-buffer_m, -buffer_m),
        (-buffer_m, buffer_m),
        (buffer_m, -buffer_m),
        (buffer_m, buffer_m),
    ]
    corners_lonlat = [to_wgs84.transform(x, y) for x, y in corners_m]
    lons = [xy[0] for xy in corners_lonlat]
    lats = [xy[1] for xy in corners_lonlat]

    west, east = max(-180.0, min(lons)), min(180.0, max(lons))
    south, north = max(-90.0, min(lats)), min(90.0, max(lats))
    if east - west > 180.0:
        raise ValueError("Buffers crossing the antimeridian are not supported by this simple downloader.")
    return BBox(west=west, south=south, east=east, north=north)


def get_worldcover_urls(bbox: BBox) -> List[str]:
    """Return ESA WorldCover 2021 v200 Map COG URLs intersecting the bbox."""
    urls = []
    for lat_origin, lon_origin in product(
        _tile_origins(bbox.south, bbox.north),
        _tile_origins(bbox.west, bbox.east),
    ):
        tile = f"{_format_lat(lat_origin)}{_format_lon(lon_origin)}"
        filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        urls.append(f"{BASE_URL}/{filename}")
    return urls


def url_exists(url: str, timeout: int = 30) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def _read_cog_window(url: str, bbox: BBox):
    with rasterio.open(url) as src:
        full_window = from_bounds(*bbox.to_tuple(), transform=src.transform)
        tile_window = from_bounds(*src.bounds, transform=src.transform)
        window = intersection(full_window, tile_window).round_offsets().round_lengths()

        data = src.read(1, window=window)
        transform = src.window_transform(window)
        meta = src.meta.copy()

    meta.update(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        transform=transform,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    return data, meta


def download_worldcover(
    lon: float,
    lat: float,
    buffer_m: float,
    out_tif: str,
    skip_missing: bool = True,
) -> str:
    """Download, clip, and merge ESA WorldCover tiles for a point buffer."""
    bbox = make_bbox_lonlat(lon=lon, lat=lat, buffer_m=buffer_m)
    urls = get_worldcover_urls(bbox)
    existing_urls = []

    for url in urls:
        if url_exists(url):
            existing_urls.append(url)
        elif skip_missing:
            print(f"Skipping missing tile: {url}", file=sys.stderr)
        else:
            raise FileNotFoundError(f"ESA WorldCover tile not found: {url}")

    if not existing_urls:
        raise FileNotFoundError("No ESA WorldCover tiles found for the requested area.")

    os.makedirs(os.path.dirname(os.path.abspath(out_tif)) or ".", exist_ok=True)

    memfiles = []
    datasets = []
    try:
        for url in existing_urls:
            try:
                data, meta = _read_cog_window(url, bbox)
            except WindowError:
                continue

            memfile = MemoryFile()
            with memfile.open(**meta) as dst:
                dst.write(data, 1)
                dst.update_tags(1, **{str(k): v for k, v in CLASS_NAMES.items()})

            memfiles.append(memfile)
            datasets.append(memfile.open())

        if not datasets:
            raise RuntimeError("Tiles were found, but none overlapped the requested bbox.")

        if len(datasets) == 1:
            data = datasets[0].read(1)
            meta = datasets[0].meta.copy()
        else:
            merged, transform = merge(datasets, bounds=bbox.to_tuple(), nodata=NODATA)
            data = merged[0]
            meta = datasets[0].meta.copy()
            meta.update(
                height=merged.shape[1],
                width=merged.shape[2],
                transform=transform,
            )

        meta.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            crs=f"EPSG:{EPSG_WGS84}",
            nodata=NODATA,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(data, 1)
            dst.update_tags(
                dataset="ESA WorldCover 10 m 2021 v200",
                source="https://esa-worldcover.org/en/data-access",
                bbox_wgs84=",".join(f"{value:.10f}" for value in bbox.to_tuple()),
            )
            dst.update_tags(1, **{str(k): v for k, v in CLASS_NAMES.items()})
    finally:
        for ds in datasets:
            ds.close()
        for memfile in memfiles:
            memfile.close()

    print(f"Saved: {out_tif}")
    print(f"BBox EPSG:4326: {bbox.to_tuple()}")
    print("Tiles:")
    for url in existing_urls:
        print(f"  {url}")
    return out_tif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ESA WorldCover 2021 v200 tiles for a lon/lat point and buffer."
    )
    parser.add_argument("--lon", type=float, required=True, help="Longitude in decimal degrees.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude in decimal degrees.")
    parser.add_argument("--buffer-m", type=float, required=True, help="Square buffer radius in metres.")
    parser.add_argument("--out", required=True, help="Output GeoTIFF path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any intersecting 3-degree tile is missing instead of skipping missing tiles.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_worldcover(
        lon=args.lon,
        lat=args.lat,
        buffer_m=args.buffer_m,
        out_tif=args.out,
        skip_missing=not args.strict,
    )
