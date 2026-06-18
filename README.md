# TOPMODEL / DISPATCH Style SMIPS Redistribution

This is a small scaffold for disaggregating coarse SMIPS soil moisture pixels to an
approximately 30 m grid without fitting a regression between coarse and fine
scales.

The model treats each coarse SMIPS pixel as a mass constraint:

```text
mean(30 m soil moisture inside 1 km pixel) = SMIPS 1 km soil moisture
```

Fine-scale covariates only redistribute that moisture as a bounded relative
wetness pattern. The redistribution weights are dynamic: terrain controls become
stronger under wet/connected states, while soil storage, vegetation cover, and
evaporative exposure dominate during dry/disconnected states.

## Concept

The scaffold combines four ideas:

- **Bucket constraint**: SMIPS defines the coarse water state.
- **TOPMODEL prior**: TWI matters most when wetness/connectivity is active.
- **DISPATCH prior**: heat/radiation exposure dries the surface during dry-down.
- **Soil storage prior**: texture and holding capacity control persistence.

No trained regression coefficients are used. The default parameter values are
heuristic and should be treated as priors, not calibrated truth.

## Quick Start

```bash
python examples/synthetic_run.py
```

The example creates a 2 x 2 SMIPS grid and disaggregates each coarse cell into a
33 x 33 fine grid. In production you would usually replace this integer block
mapping with raster overlay weights because 1000 m / 30 m is not exactly an
integer.

## Inputs

Required fine-grid covariates:

- `twi`: topographic wetness index or equivalent convergence prior.
- `hli`: heat load, radiation, aspect exposure, or another evaporative exposure
  prior where higher values mean drier exposure.
- `holding_capacity`: plant-available or total water holding capacity prior.

Optional fine-grid covariates:

- `vegetation_cover`: fractional cover, NDVI-like cover, or biomass proxy.
- `infiltration_capacity`: relative infiltration/storage intake prior.

Weather/state inputs:

- `antecedent_precip`: recent rainfall or antecedent precipitation index.
- `potential_et`: evaporative demand.

## Hydrological States

For each coarse SMIPS pixel the model estimates:

- relative wetness
- rainfall pulse strength
- dry-down strength
- connectivity

Connectivity is a smooth threshold function of wetness and antecedent rainfall.
It determines whether TWI acts as a strong lateral redistribution prior or only a
weak background covariate.

## Files

- `src/topmodel_dispatch_hybrid/model.py`: core model.
- `src/topmodel_dispatch_hybrid/geospatial.py`: xarray adapter for real
  coarse/fine raster grids where the scale factor is not exactly integer.
- `examples/synthetic_run.py`: minimal synthetic run.
- `examples/real_location_query.py`: real AOI/date runner using PaddockTS,
  SMIPS, TerrainTiles, SILO, and optionally SLGA soils.
- `tests/test_model.py`: mass-preservation and state-response checks.

## Real Location Run

The real-data runner uses the sibling `paddock-ts-local` project if it is present
at `/Users/dmitrygrishin/borevitz_projects/paddock-ts-local`.

```bash
PYTHONPATH=src:/Users/dmitrygrishin/borevitz_projects/paddock-ts-local \
python examples/real_location_query.py \
  --lat -33.51 \
  --lon 148.37 \
  --buffer-km 1 \
  --start 2023-01-01 \
  --end 2023-01-07 \
  --stub cowra_hybrid_test
```

Outputs are written under the PaddockTS output directory, usually:

```text
~/Documents/PaddockTS-Outputs/<stub>/topmodel_dispatch_hybrid/
```

The runner saves:

- `<stub>_hybrid_smips_downscaled.nc`: NetCDF with downscaled soil moisture,
  relative wetness score, connectivity, TWI, HLI, and holding capacity.
- `<stub>_soil_moisture_snapshot.png`: TWI, HLI, coarse SMIPS, and downscaled
  soil moisture for the middle date.
- `<stub>_rainfall_temperature.png`: SILO daily rainfall and min/max
  temperature.
- `<stub>_summary.png`: mean downscaled soil moisture, SMIPS mean, rainfall,
  and temperature through time.

To include real SLGA soil texture-derived holding capacity, add:

```bash
--with-soils
```

That requires a TERN API key in `~/.config/PaddockTS.json`. SILO rainfall and
temperature require a configured SILO email in the same file or in the PaddockTS
config.

## EMT Calibration From DEM + Point Soil Moisture

The package now includes an Equilibrium Moisture from Topography (EMT) calibration
workflow based on Coleman and Niemann (2013), *Controls on topographic dependence
and temporal instability in catchment-scale soil moisture patterns*.

The workflow:

- reads an AOI DEM and derives slope, aspect, flow accumulation, TWI, an LFI proxy,
  HLI, and an ETI proxy;
- reads a point soil-moisture CSV with coordinates and optional dates;
- samples the terrain indices at each observation point;
- uses spatial-average soil moisture as the EMT wetness-state input;
- calibrates observed anomalies around that spatial mean using LFI and ETI terms;
- reports RMSE, NSE, and the paper-style average spatial NSCE when dated point
  groups are available;
- writes a calibrated model JSON, point covariate CSV, diagnostics JSON, and
  predicted EMT grid NetCDF.

```bash
PYTHONPATH=src python examples/emt_calibration_workflow.py \
  --dem /path/to/aoi_dem.tif \
  --observations /path/to/soil_moisture_points.csv \
  --out-dir outputs/emt_test \
  --stub paddock_emt \
  --x-column longitude \
  --y-column latitude \
  --moisture-column soil_moisture \
  --time-column date \
  --lower-bound 0.0 \
  --upper-bound 0.6
```

Optional SILO/OzWALD-style climate tables can be joined by date:

```bash
PYTHONPATH=src python examples/emt_calibration_workflow.py \
  --dem /path/to/aoi_dem.tif \
  --observations /path/to/soil_moisture_points.csv \
  --climate-csv /path/to/silo.csv \
  --out-dir outputs/emt_test
```

For downloaded covariates, `src/topmodel_dispatch_hybrid/paddockts_bridge.py`
provides thin adapters around the local `paddock-ts-local` checkout under
`/Volumes/Dmitry_work/borevitz_projects/paddock-ts-local`. Use those adapters to
create a PaddockTS query, download terrain covariates, and download SILO climate
before running the EMT calibration.

### Optional SMIPS Forcing

The base EMT calibration does not require SMIPS. It calibrates a fine-scale
topographic pattern from point observations and uses the point-observation
spatial mean as the wetness-state input.

To use SMIPS as the wetness-state driver, pass either an existing SMIPS NetCDF
or ask the workflow to download one through PaddockTS. The workflow then writes:

- `<stub>_emt_prediction.nc`: EMT using the observation-derived mean state.
- `<stub>_emt_smips_prediction.nc`: EMT using SMIPS as spatial/time-varying
  mean-moisture forcing.
- `<stub>_emt_smips_side_by_side.png`: EMT prediction and the driving SMIPS tile
  plotted side by side.

For SMIPS total bucket in millimetres:

```bash
PYTHONPATH=src python examples/emt_calibration_workflow.py \
  --dem /path/to/aoi_dem.tif \
  --observations /path/to/soil_moisture_points.csv \
  --out-dir outputs/emt_test \
  --moisture-column Water_mm \
  --time-column Date \
  --lower-bound 0 \
  --upper-bound 30 \
  --download-smips \
  --smips-layer TotalBucketRaw \
  --smips-mode totalbucket
```

For SMIPS relative fullness / SMIndex:

```bash
PYTHONPATH=src python examples/emt_calibration_workflow.py \
  --dem /path/to/aoi_dem.tif \
  --observations /path/to/soil_moisture_points.csv \
  --out-dir outputs/emt_test \
  --moisture-column Water_mm \
  --time-column Date \
  --lower-bound 0 \
  --upper-bound 30 \
  --download-smips \
  --smips-layer SMIndexRaw \
  --smips-mode relative_fullness
```

`TotalBucketRaw` is treated as moisture in the same units as the calibrated EMT
target. `SMIndexRaw` is treated as relative fullness and mapped into the EMT
lower/upper bounds. Dates are inferred from the observation time column unless
`--smips-start` and `--smips-end` are supplied.
