#!/usr/bin/env python
"""Calibrate an EMT soil-moisture model from an AOI DEM and point CSV."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from topmodel_dispatch_hybrid.observations import ObservationColumns
from topmodel_dispatch_hybrid.workflow import calibrate_emt_from_dem_and_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", required=True, type=Path, help="AOI DEM GeoTIFF path.")
    parser.add_argument("--observations", required=True, type=Path, help="Soil moisture point CSV.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stub", default="emt_calibration")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--bbox-crs", default="EPSG:4326")
    parser.add_argument("--coordinate-crs", default="EPSG:4326")
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--x-column")
    parser.add_argument("--y-column")
    parser.add_argument("--moisture-column")
    parser.add_argument("--time-column")
    parser.add_argument("--site-id-column")
    parser.add_argument("--climate-csv", type=Path)
    parser.add_argument("--climate-date-column")
    parser.add_argument("--rainfall-column")
    parser.add_argument("--et-column")
    parser.add_argument("--lower-bound", type=float)
    parser.add_argument("--upper-bound", type=float)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--smips-path", type=Path, help="Existing SMIPS NetCDF to use as EMT forcing.")
    parser.add_argument("--download-smips", action="store_true", help="Download SMIPS for the AOI/date range.")
    parser.add_argument(
        "--smips-layer",
        default="TotalBucketRaw",
        choices=["TotalBucketRaw", "SMIndexRaw", "totalbucket", "SMindex", "bucket1", "bucket2"],
    )
    parser.add_argument("--smips-source", choices=["auto", "cog", "wms"], default="auto")
    parser.add_argument(
        "--smips-mode",
        choices=["auto", "totalbucket", "relative_fullness"],
        default="auto",
        help="How to convert SMIPS values into EMT mean-moisture forcing.",
    )
    parser.add_argument("--smips-start", type=date.fromisoformat)
    parser.add_argument("--smips-end", type=date.fromisoformat)
    parser.add_argument("--paddockts-path", type=Path)
    parser.add_argument("--reload-smips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = None
    if args.x_column or args.y_column or args.moisture_column:
        if not (args.x_column and args.y_column and args.moisture_column):
            raise ValueError("--x-column, --y-column, and --moisture-column must be supplied together")
        columns = ObservationColumns(
            x=args.x_column,
            y=args.y_column,
            moisture=args.moisture_column,
            time=args.time_column,
            site_id=args.site_id_column,
        )

    result = calibrate_emt_from_dem_and_csv(
        dem_path=args.dem,
        observations_csv=args.observations,
        output_dir=args.out_dir,
        bbox=tuple(args.bbox) if args.bbox else None,
        bbox_crs=args.bbox_crs,
        coordinate_crs=args.coordinate_crs,
        latitude=args.latitude,
        observation_columns=columns,
        climate_csv=args.climate_csv,
        climate_date_column=args.climate_date_column,
        rainfall_column=args.rainfall_column,
        et_column=args.et_column,
        output_stub=args.stub,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        ridge=args.ridge,
        smips_path=args.smips_path,
        download_smips=args.download_smips,
        smips_layer=args.smips_layer,
        smips_source=args.smips_source,
        smips_mode=args.smips_mode,
        smips_start=args.smips_start,
        smips_end=args.smips_end,
        paddockts_path=args.paddockts_path,
        reload_smips=args.reload_smips,
    )

    diagnostics = result.calibration.model.diagnostics
    print(f"Saved point covariates: {result.point_covariates_path}")
    print(f"Saved EMT model: {result.model_path}")
    print(f"Saved prediction grid: {result.prediction_path}")
    if result.smips_prediction_path is not None:
        print(f"Saved SMIPS-driven EMT grid: {result.smips_prediction_path}")
    if result.smips_side_by_side_path is not None:
        print(f"Saved EMT/SMIPS side-by-side PNG: {result.smips_side_by_side_path}")
    print(
        "Diagnostics: "
        f"n={diagnostics['n_observations']}, "
        f"rmse={diagnostics['rmse']:.4g}, "
        f"nse={diagnostics['nse']}"
    )


if __name__ == "__main__":
    main()
