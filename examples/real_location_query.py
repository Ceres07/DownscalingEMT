#!/usr/bin/env python
"""Run the hybrid SMIPS redistribution for a real location and time range."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from topmodel_dispatch_hybrid import DisaggregationConfig, WeatherForcing
from topmodel_dispatch_hybrid.geospatial import disaggregate_smips_geospatial


DEFAULT_PADDOCKTS = Path("/Users/dmitrygrishin/borevitz_projects/paddock-ts-local")
SMIPS_COG_START = date(2015, 11, 20)
SMIPS_COG_DELAY_DAYS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=-33.51)
    parser.add_argument("--lon", type=float, default=148.37)
    parser.add_argument("--buffer-km", type=float, default=1.0)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 1, 7))
    parser.add_argument("--stub", default="hybrid_smips_real_location")
    parser.add_argument(
        "--smips-layer",
        default="TotalBucketRaw",
        choices=[
            "TotalBucketRaw",
            "SMIndexRaw",
            "totalbucket",
            "SMindex",
            "bucket1",
            "bucket2",
            "deepD",
            "runoff",
        ],
        help="SMIPS layer/collection to download. TotalBucketRaw maps to the modern totalbucket COG collection.",
    )
    parser.add_argument("--paddockts-path", type=Path, default=DEFAULT_PADDOCKTS)
    parser.add_argument("--with-soils", action="store_true", help="Download SLGA Clay/Sand and derive holding capacity.")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--smips-source", choices=["auto", "cog", "wms"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paddockts_path.exists():
        sys.path.insert(0, str(args.paddockts_path))

    from PaddockTS.Environmental.SILO.download_silo import download_silo
    from PaddockTS.Environmental.SMIPS.download_smips import smips_cube
    from PaddockTS.Environmental.TerrainTiles.download_terrain_tiles import (
        download_terrain,
        get_filename as terrain_filename,
    )
    from PaddockTS.Environmental.TerrainTiles.utils import (
        calculate_aspect,
        calculate_hli,
        calculate_slope,
        calculate_twi,
        pysheds_accumulation,
    )
    from PaddockTS.query import Query

    query = Query.from_lat_lon(
        lat=args.lat,
        lon=args.lon,
        buffer_km=args.buffer_km,
        start=args.start,
        end=args.end,
        stub=args.stub,
    )
    out_dir = Path(query.out_dir) / "topmodel_dispatch_hybrid"
    tmp_dir = Path(query.tmp_dir) / "topmodel_dispatch_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Query: {query.stub}")
    print(f"  bbox: {query.bbox}")
    print(f"  period: {query.start} to {query.end}")
    print(f"  output: {out_dir}")
    validate_smips_dates(query.start, query.end)

    terrain = load_or_build_terrain(
        query,
        tmp_dir / f"{query.stub}_terrain_twi_hli.nc",
        reload=args.reload,
        download_terrain=download_terrain,
        terrain_filename=terrain_filename,
        pysheds_accumulation=pysheds_accumulation,
        calculate_slope=calculate_slope,
        calculate_aspect=calculate_aspect,
        calculate_twi=calculate_twi,
        calculate_hli=calculate_hli,
    )
    smips_layer_stub = args.smips_layer.lower()
    smips_key = _smips_request_key(query.bbox, query.start, query.end, args.smips_layer)
    smips = load_or_fetch_smips(
        query,
        tmp_dir / f"{query.stub}_{smips_layer_stub}_smips_{smips_key}.nc",
        args.reload,
        smips_cube,
        args.smips_layer,
        args.smips_source,
    )
    smips_summary = summarise_smips(smips)
    silo_key = _silo_request_key(query.bbox, query.start, query.end)
    climate = load_or_fetch_silo(query, tmp_dir / f"{query.stub}_silo_{silo_key}.csv", args.reload, download_silo)
    holding_capacity = load_holding_capacity(query, terrain, tmp_dir, args.with_soils)

    climate_indexed = climate.set_index("YYYY-MM-DD").sort_index()
    smips_dates = pd.to_datetime(smips.time.values)
    precip = climate_indexed["daily_rain"].reindex(smips_dates, method="nearest").to_numpy()
    pet_col = "et_short_crop" if "et_short_crop" in climate else "evap_pan"
    pet = climate_indexed[pet_col].reindex(smips_dates, method="nearest").to_numpy()

    result = disaggregate_smips_geospatial(
        smips=smips,
        twi=terrain["twi"],
        hli=terrain["hli"],
        holding_capacity=holding_capacity,
        weather=WeatherForcing(antecedent_precip=precip, potential_et=pet),
        config=DisaggregationConfig(
            wilting_point=0.0,
            field_capacity=120.0,
            saturation=250.0,
            redistribution_strength=40.0,
            max_anomaly_fraction=0.45,
        ),
        bbox=query.bbox,
    )

    ds_out = xr.Dataset(
        {
            "soil_moisture_downscaled": result.soil_moisture,
            "relative_wetness_score": result.relative_wetness_score,
            "connectivity": result.connectivity,
            "twi": terrain["twi"],
            "hli": terrain["hli"],
            "holding_capacity": holding_capacity,
        },
        attrs={
            "bbox": str(query.bbox),
            "start": query.start.isoformat(),
            "end": query.end.isoformat(),
            "method": "State-dependent TOPMODEL/DISPATCH-style redistribution constrained by SMIPS means.",
        },
    )
    nc_path = out_dir / f"{query.stub}_hybrid_smips_downscaled.nc"
    ds_out.to_netcdf(nc_path)
    print(f"Saved: {nc_path}")

    diagnostics_csv = out_dir / f"{query.stub}_smips_diagnostics.csv"
    smips_summary.to_csv(diagnostics_csv, index=False)
    spatial_png = out_dir / f"{query.stub}_soil_moisture_snapshot.png"
    climate_png = out_dir / f"{query.stub}_rainfall_temperature.png"
    summary_png = out_dir / f"{query.stub}_summary.png"
    plot_soil_moisture_snapshot(terrain, smips, result.soil_moisture, result.connectivity, spatial_png)
    plot_climate(climate, climate_png)
    plot_summary(result.soil_moisture, smips, result.connectivity, climate, summary_png)
    print(f"Saved: {diagnostics_csv}")
    print(f"Saved: {spatial_png}")
    print(f"Saved: {climate_png}")
    print(f"Saved: {summary_png}")


def load_or_build_terrain(query, path: Path, reload: bool, **funcs) -> xr.Dataset:
    if path.exists() and not reload:
        print(f"Using cached terrain: {path}")
        return xr.open_dataset(path).load()

    terrain_tif = funcs["terrain_filename"](query)
    if not Path(terrain_tif).exists():
        print("Downloading terrain DEM...")
        funcs["download_terrain"](query)

    _, dem_filled, _, acc = funcs["pysheds_accumulation"](terrain_tif)
    slope = funcs["calculate_slope"](terrain_tif)
    aspect = funcs["calculate_aspect"](terrain_tif)
    twi = funcs["calculate_twi"](acc, slope)
    hli = funcs["calculate_hli"](slope, aspect, query.centre_lat)

    with rasterio.open(terrain_tif) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    ny, nx = dem.shape
    x = np.arange(nx) * transform.a + transform.c + transform.a / 2
    y = np.arange(ny) * transform.e + transform.f + transform.e / 2
    ds = xr.Dataset(
        {
            "dem": (("y", "x"), dem),
            "dem_filled": (("y", "x"), np.asarray(dem_filled, dtype=np.float32)),
            "slope": (("y", "x"), slope.astype(np.float32)),
            "aspect": (("y", "x"), aspect.astype(np.float32)),
            "twi": (("y", "x"), twi.astype(np.float32)),
            "hli": (("y", "x"), hli.astype(np.float32)),
        },
        coords={"y": y, "x": x},
        attrs={"crs": str(crs), "source": "PaddockTS TerrainTiles"},
    )
    ds.to_netcdf(path)
    print(f"Saved: {path}")
    return ds


def load_or_fetch_smips(query, path: Path, reload: bool, smips_cube, layer: str, source: str) -> xr.DataArray:
    if path.exists() and not reload:
        print(f"Using cached SMIPS: {path}")
        return xr.open_dataset(path)["smips"].load()
    print(f"Downloading SMIPS layer {layer} from {source} source...")
    smips = smips_cube(
        query.start,
        query.end,
        tuple(query.bbox),
        layer=layer,
        source=source,
        skip_missing=True,
    ).rename("smips").astype(np.float32)
    smips = _normalise_smips(smips)
    smips.attrs.setdefault("units", "mm")
    smips.attrs["layer"] = layer
    smips.to_dataset().to_netcdf(path)
    print(f"Saved: {path}")
    return smips


def validate_smips_dates(start: date, end: date) -> None:
    latest = (pd.Timestamp.today().normalize() - pd.Timedelta(days=SMIPS_COG_DELAY_DAYS)).date()
    if start < SMIPS_COG_START or end > latest:
        raise ValueError(
            "SMIPS COG daily coverage is available from "
            f"{SMIPS_COG_START.isoformat()} to approximately {latest.isoformat()}; "
            f"requested {start.isoformat()} to {end.isoformat()}. "
            "Use a date range inside that window or supply a different coarse soil-moisture source. "
            "The older WMS endpoint only advertises coverage to 2023-03-01."
        )


def summarise_smips(smips: xr.DataArray) -> pd.DataFrame:
    smips = _normalise_smips(smips)
    if "time" not in smips.dims:
        smips = smips.expand_dims(time=[np.datetime64("NaT")])

    spatial_dims = [dim for dim in ("y", "x") if dim in smips.dims]
    summary = pd.DataFrame(
        {
            "date": pd.to_datetime(smips.time.values),
            "valid_pixels": smips.count(spatial_dims).values.astype(int),
            "mean": smips.mean(spatial_dims, skipna=True).values,
            "min": smips.min(spatial_dims, skipna=True).values,
            "max": smips.max(spatial_dims, skipna=True).values,
            "std": smips.std(spatial_dims, skipna=True).values,
            "fingerprint": [
                _array_fingerprint(smips.isel(time=i).values)
                for i in range(smips.sizes.get("time", 1))
            ],
        }
    )

    coarse_shape = " x ".join(str(smips.sizes[dim]) for dim in spatial_dims)
    print("SMIPS diagnostics:")
    print(f"  layer: {smips.attrs.get('layer', smips.attrs.get('long_name', 'unknown'))}")
    print(f"  timesteps returned: {smips.sizes.get('time', 1)}")
    print(f"  coarse grid shape: {coarse_shape} ({int(summary['valid_pixels'].max())} valid pixels max)")
    print(f"  dates: {summary['date'].min().date()} to {summary['date'].max().date()}")
    print(f"  daily mean range: {summary['mean'].min():.4g} to {summary['mean'].max():.4g}")
    print(f"  daily min/max range: {summary['min'].min():.4g} to {summary['max'].max():.4g}")
    print(f"  daily spatial std range: {summary['std'].min():.4g} to {summary['std'].max():.4g}")
    print(f"  unique raster fingerprints: {summary['fingerprint'].nunique()}")
    if np.isclose(summary["mean"].min(), summary["mean"].max(), equal_nan=True):
        print("  note: SMIPS domain mean is effectively flat for this AOI/time window.")
    if summary["fingerprint"].nunique() == 1 and len(summary) > 1:
        print("  warning: every downloaded SMIPS raster is byte-identical after NaN normalization.")
    return summary


def _array_fingerprint(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, np.float32(-999999.0))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _smips_request_key(
    bbox: list[float] | tuple[float, float, float, float],
    start: date,
    end: date,
    layer: str,
) -> str:
    payload = f"{tuple(float(v) for v in bbox)}|{start}|{end}|{layer}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _silo_request_key(
    bbox: list[float] | tuple[float, float, float, float],
    start: date,
    end: date,
) -> str:
    payload = f"{tuple(float(v) for v in bbox)}|{start}|{end}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_or_fetch_silo(query, path: Path, reload: bool, download_silo) -> pd.DataFrame:
    if path.exists() and not reload:
        print(f"Using cached SILO: {path}")
        return _normalise_silo_dates(pd.read_csv(path))
    print("Downloading SILO rainfall and temperature...")
    climate = _normalise_silo_dates(download_silo(query))
    climate.to_csv(path, index=False)
    print(f"Saved: {path}")
    return climate


def _normalise_silo_dates(climate: pd.DataFrame) -> pd.DataFrame:
    climate = climate.copy()
    if "YYYY-MM-DD" not in climate:
        raise ValueError("SILO climate table is missing the YYYY-MM-DD column")
    climate["YYYY-MM-DD"] = pd.to_datetime(climate["YYYY-MM-DD"])
    return climate.sort_values("YYYY-MM-DD").reset_index(drop=True)


def load_holding_capacity(query, terrain: xr.Dataset, tmp_dir: Path, with_soils: bool) -> xr.DataArray:
    if not with_soils:
        print("Using neutral holding capacity. Pass --with-soils to derive it from SLGA Clay/Sand.")
        return xr.ones_like(terrain["twi"], dtype=float).rename("holding_capacity")

    try:
        from PaddockTS.Environmental.SLGASoils.download_slgasoils import download_slga_soils, get_filename

        download_slga_soils(query, vars=["Clay", "Sand"], depths=["5-15cm"])
        clay = read_soil_tif(get_filename(query, "Clay", "5-15cm"), terrain)
        sand = read_soil_tif(get_filename(query, "Sand", "5-15cm"), terrain)
        holding = (0.6 * _scale01(clay) + 0.4 * (1.0 - _scale01(sand))).rename("holding_capacity")
        holding.to_netcdf(tmp_dir / f"{query.stub}_holding_capacity.nc")
        return holding
    except Exception as exc:
        print(f"Could not load SLGA soils ({exc}). Falling back to neutral holding capacity.")
        return xr.ones_like(terrain["twi"], dtype=float).rename("holding_capacity")


def read_soil_tif(path: str, terrain: xr.Dataset) -> xr.DataArray:
    import rioxarray

    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    if "x" not in da.coords or "y" not in da.coords:
        raise ValueError(f"Soil raster lacks x/y coordinates: {path}")
    return da.interp(x=terrain.x, y=terrain.y, method="nearest")


def plot_soil_moisture_snapshot(
    terrain: xr.Dataset,
    smips: xr.DataArray,
    downscaled: xr.DataArray,
    connectivity: xr.DataArray,
    path: Path,
) -> None:
    idx = len(downscaled.time) // 2
    time_label = str(downscaled.time.values[idx])[:10]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    terrain["twi"].plot(ax=axes[0, 0], cmap="Blues", cbar_kwargs={"label": "TWI"})
    axes[0, 0].set_title("TWI")

    terrain["hli"].plot(ax=axes[0, 1], cmap="YlOrRd", cbar_kwargs={"label": "HLI"})
    axes[0, 1].set_title("Heat Load Index")

    smips.isel(time=idx).plot(ax=axes[1, 0], cmap="YlGnBu", cbar_kwargs={"label": smips.attrs.get("units", "mm")})
    axes[1, 0].set_title(f"SMIPS coarse ({time_label})")

    downscaled.isel(time=idx).plot(ax=axes[1, 1], cmap="YlGnBu", cbar_kwargs={"label": downscaled.attrs.get("units", "mm")})
    conn = float(connectivity.isel(time=idx).mean(skipna=True))
    axes[1, 1].set_title(f"Downscaled ({time_label}); mean connectivity {conn:.2f}")

    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_climate(climate: pd.DataFrame, path: Path) -> None:
    dates = pd.to_datetime(climate["YYYY-MM-DD"])
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(dates, climate["daily_rain"], color="#2878b5", alpha=0.75, label="Rainfall")
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()
    ax2.plot(dates, climate["max_temp"], color="#c23b22", linewidth=1.8, label="Max temp")
    ax2.plot(dates, climate["min_temp"], color="#5b8cc0", linewidth=1.4, label="Min temp")
    ax2.set_ylabel("Temperature (deg C)")
    ax1.set_title("Rainfall and Temperature")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    downscaled: xr.DataArray,
    smips: xr.DataArray,
    connectivity: xr.DataArray,
    climate: pd.DataFrame,
    path: Path,
) -> None:
    dates = pd.to_datetime(downscaled.time.values)
    sm_mean = downscaled.mean(("x", "y"), skipna=True).to_pandas()
    sm_std = downscaled.std(("x", "y"), skipna=True).to_pandas()
    smips_mean = smips.mean(("x", "y"), skipna=True).to_pandas()
    connectivity_mean = connectivity.mean(("x", "y"), skipna=True).to_pandas()

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(dates, sm_mean.values, color="#1f6f50", linewidth=2, label="Downscaled mean")
    axes[0].plot(pd.to_datetime(smips.time.values), smips_mean.values, color="#333333", linestyle="--", label="SMIPS mean")
    axes[0].set_ylabel(smips.attrs.get("units", "mm"))
    axes[0].set_title("Soil Moisture Mean (Mass-Preserved)")
    axes[0].legend()

    axes[1].plot(dates, sm_std.values, color="#6a4c93", linewidth=1.8, label="Downscaled spatial std")
    axes[1].plot(dates, connectivity_mean.values, color="#d17a22", linewidth=1.8, label="Mean connectivity")
    axes[1].set_ylabel("Std / connectivity")
    axes[1].set_title("Redistribution Dynamics")
    axes[1].legend()

    axes[2].bar(pd.to_datetime(climate["YYYY-MM-DD"]), climate["daily_rain"], color="#2878b5")
    axes[2].set_ylabel("Rainfall (mm/day)")

    axes[3].plot(pd.to_datetime(climate["YYYY-MM-DD"]), climate["max_temp"], color="#c23b22", label="Max")
    axes[3].plot(pd.to_datetime(climate["YYYY-MM-DD"]), climate["min_temp"], color="#5b8cc0", label="Min")
    axes[3].set_ylabel("Temperature (deg C)")
    axes[3].legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalise_smips(smips: xr.DataArray) -> xr.DataArray:
    if "xc" in smips.coords and "yc" in smips.coords:
        smips = smips.assign_coords(x=smips["xc"].isel(y=0).values, y=smips["yc"].isel(x=0).values)
    return smips


def _scale01(da: xr.DataArray) -> xr.DataArray:
    lo = da.quantile(0.02, skipna=True)
    hi = da.quantile(0.98, skipna=True)
    return ((da - lo) / (hi - lo)).clip(0, 1)


if __name__ == "__main__":
    main()
