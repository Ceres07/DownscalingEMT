#!/usr/bin/env python
"""Calibrate physical EMT parameters and export per-date maps/metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from topmodel_dispatch_hybrid.physical_emt import (
    calibrate_physical_emt,
    predict_physical_emt_grid,
    score_predictions_by_date,
    write_table1_style_parameter_csv,
    write_table1_style_parameter_set,
)
from topmodel_dispatch_hybrid.smips_integration import align_smips_coarse_to_terrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--terrain", required=True, type=Path)
    parser.add_argument("--smips", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stub", default="physical_emt")
    parser.add_argument("--theta-column", default="Soil_moisture")
    parser.add_argument("--water-column", default="Water_mm")
    parser.add_argument("--time-column", default="Date")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.points)
    terrain = xr.open_dataset(args.terrain).load()
    smips = _read_first_data_array(args.smips)

    calibration = calibrate_physical_emt(
        points,
        terrain,
        theta_column=args.theta_column,
        theta_is_percent=True,
        water_column=args.water_column,
        time_column=args.time_column,
        seed=args.seed,
        maxiter=args.maxiter,
    )
    calibration.to_json(args.out_dir / f"{args.stub}_physical_emt_parameters.json")
    write_table1_style_parameter_csv(
        calibration,
        args.out_dir / f"{args.stub}_physical_emt_table1_parameters.csv",
        catchment_name=args.stub,
    )
    write_table1_style_parameter_set(
        calibration,
        args.out_dir / f"{args.stub}_physical_emt_table1_parameters.md",
        catchment_name=args.stub,
    )
    calibration.date_metrics.to_csv(args.out_dir / f"{args.stub}_physical_emt_calibration_metrics_by_date.csv", index=False)

    point_dates = pd.to_datetime(points[args.time_column]).dt.normalize().drop_duplicates().sort_values()
    smips_dates = smips.sel(time=point_dates.to_numpy(), method="nearest")
    smips_aligned_mm = align_smips_coarse_to_terrain(smips_dates, terrain, source_crs="EPSG:4326").rename(
        "smips_totalbucket_mm"
    )
    theta_bar = (smips_aligned_mm / calibration.water_depth_mm).clip(0.0, calibration.parameters.porosity * 0.999)
    theta_bar = theta_bar.rename("smips_theta_bar")
    predicted_theta = predict_physical_emt_grid(terrain, calibration.parameters, theta_bar=theta_bar)
    predicted_water = (predicted_theta * calibration.water_depth_mm).rename("physical_emt_water_mm")
    predicted_water.attrs["units"] = "mm"

    sampled = _sample_predictions_at_points(points, predicted_theta, args.time_column)
    observed_theta = pd.to_numeric(points[args.theta_column], errors="coerce").to_numpy(dtype=float) / 100.0
    map_metrics = score_predictions_by_date(points, observed_theta, sampled, time_column=args.time_column)
    map_metrics["rmse_water_mm"] = map_metrics["rmse_theta"] * calibration.water_depth_mm
    map_metrics.to_csv(args.out_dir / f"{args.stub}_physical_emt_smips_map_metrics_by_date.csv", index=False)

    ds_out = xr.Dataset(
        {
            "physical_emt_theta": predicted_theta,
            "physical_emt_water_mm": predicted_water,
            "smips_totalbucket_mm_aligned": smips_aligned_mm,
            "smips_theta_bar": theta_bar,
        },
        attrs={
            "method": "Physical EMT driven by nearest coarse SMIPS total bucket tile",
            "water_depth_mm": calibration.water_depth_mm,
        },
    )
    nc_path = args.out_dir / f"{args.stub}_physical_emt_smips_maps.nc"
    ds_out.to_netcdf(nc_path)

    png_dir = args.out_dir / f"{args.stub}_physical_emt_maps"
    png_dir.mkdir(parents=True, exist_ok=True)
    for index, date_value in enumerate(pd.to_datetime(predicted_water.time.values)):
        metrics = map_metrics[map_metrics["date"] == date_value.date().isoformat()].iloc[0]
        _plot_date_map(
            predicted_water.isel(time=index),
            smips_aligned_mm.isel(time=index),
            png_dir / f"{date_value.date().isoformat()}_{args.stub}_physical_emt.png",
            rmse=float(metrics["rmse_water_mm"]),
            nse=float(metrics["nse"]),
        )

    print(f"Saved parameters: {args.out_dir / f'{args.stub}_physical_emt_parameters.json'}")
    print(f"Saved Table 1-style parameters: {args.out_dir / f'{args.stub}_physical_emt_table1_parameters.md'}")
    print(f"Saved per-date metrics: {args.out_dir / f'{args.stub}_physical_emt_smips_map_metrics_by_date.csv'}")
    print(f"Saved maps: {nc_path}")
    print(f"Saved PNG directory: {png_dir}")


def _read_first_data_array(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    data_vars = [name for name in ds.data_vars if name != "spatial_ref"]
    if not data_vars:
        raise ValueError(f"No data variables found in {path}")
    da = ds[data_vars[0]].load()
    ds.close()
    return da


def _sample_predictions_at_points(points: pd.DataFrame, predicted: xr.DataArray, time_column: str) -> np.ndarray:
    values = np.full(len(points), np.nan, dtype=float)
    x = xr.DataArray(points["_terrain_x"].to_numpy(dtype=float), dims="observation")
    y = xr.DataArray(points["_terrain_y"].to_numpy(dtype=float), dims="observation")
    date_key = pd.to_datetime(points[time_column]).dt.normalize()
    for time_value in pd.to_datetime(predicted.time.values):
        mask = date_key == time_value.normalize()
        if not mask.any():
            continue
        sampled = predicted.sel(time=time_value).sel(x=x[mask.to_numpy()], y=y[mask.to_numpy()], method="nearest")
        values[np.where(mask.to_numpy())[0]] = sampled.values
    return values


def _plot_date_map(
    emt_water,
    smips_water,
    path: Path,
    *,
    rmse: float,
    nse: float,
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    date_text = str(emt_water.time.values)[:10] if "time" in emt_water.coords else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    emt_water.plot(ax=axes[0], cmap="YlGnBu", cbar_kwargs={"label": "Water (mm)"})
    axes[0].set_title(f"Physical EMT water {date_text}\nRMSE={rmse:.2f} mm, NSE={nse:.3f}")
    smips_water.plot(ax=axes[1], cmap="YlGnBu", cbar_kwargs={"label": "SMIPS total bucket (mm)"})
    axes[1].set_title("Nearest coarse SMIPS tile forcing")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
