from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from .emt import EMTModel


def smips_to_mean_moisture(
    smips: xr.DataArray,
    model: EMTModel,
    mode: str = "auto",
) -> xr.DataArray:
    """Convert a SMIPS layer to EMT mean-moisture forcing.

    ``mode='totalbucket'`` treats SMIPS values as moisture in the same units as
    the calibrated EMT target, usually millimetres. ``mode='relative_fullness'``
    treats SMIPS values as 0-1 or 0-100 relative fullness and maps them into the
    calibrated EMT lower/upper bounds. ``mode='auto'`` selects relative
    fullness for SMIndex layers and total bucket otherwise.
    """

    mode = _resolve_smips_mode(smips, mode)
    values = smips.astype(float)
    if mode == "totalbucket":
        out = values
    elif mode == "relative_fullness":
        lower, upper = _model_bounds(model)
        fullness = values / 100.0 if float(values.max(skipna=True)) > 1.5 else values
        out = lower + fullness.clip(0.0, 1.0) * (upper - lower)
    else:
        raise ValueError("mode must be one of 'auto', 'totalbucket', or 'relative_fullness'")

    out = out.rename("smips_mean_moisture")
    out.attrs.update(
        source_layer=smips.attrs.get("layer", smips.name or ""),
        smips_mode=mode,
        units=model.config.moisture_column,
    )
    return out


def align_smips_to_terrain(
    smips: xr.DataArray,
    terrain: xr.Dataset,
    source_crs: str | None = "EPSG:4326",
) -> xr.DataArray:
    """Align a SMIPS grid to the terrain grid used by EMT."""

    smips = _normalise_xy(smips)
    target_crs = terrain.attrs.get("crs") or None
    if source_crs and target_crs and str(source_crs) != str(target_crs):
        try:
            import rioxarray  # noqa: F401

            template = terrain[next(iter(terrain.data_vars))]
            template = template.rio.write_crs(target_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
            source = smips.rio.write_crs(source_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
            aligned = source.rio.reproject_match(template)
            aligned.attrs.update(smips.attrs)
            return aligned
        except ImportError as exc:
            raise ImportError(
                "rioxarray is required to align SMIPS to a terrain grid with a different CRS"
            ) from exc
        except Exception as exc:
            raise ValueError("Could not reproject SMIPS onto the terrain grid") from exc
    return smips.interp(x=terrain.x, y=terrain.y, method="nearest")


def predict_emt_from_smips(
    model: EMTModel,
    terrain: xr.Dataset,
    smips: xr.DataArray,
    mode: str = "auto",
    source_crs: str | None = "EPSG:4326",
) -> xr.DataArray:
    """Predict EMT moisture using SMIPS as a spatial/time wetness-state driver."""

    mean_moisture = smips_to_mean_moisture(smips, model, mode=mode)
    aligned = align_smips_to_terrain(mean_moisture, terrain, source_crs=source_crs)
    out = model.predict_dataset(terrain, mean_moisture=aligned)
    out.attrs.update(
        smips_mode=aligned.attrs.get("smips_mode", mode),
        smips_layer=smips.attrs.get("layer", smips.name or ""),
        method="Calibrated EMT pattern driven by SMIPS mean-moisture forcing",
    )
    return out


def plot_emt_smips_side_by_side(
    emt: xr.DataArray,
    smips: xr.DataArray,
    path: str | Path,
    *,
    time_index: int = 0,
    smips_label: str | None = None,
) -> Path:
    """Save a side-by-side PNG of EMT prediction and the driving SMIPS tile."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emt_2d = emt.isel(time=time_index) if "time" in emt.dims else emt
    smips_2d = smips.isel(time=time_index) if "time" in smips.dims else smips
    title_date = ""
    if "time" in emt.dims:
        title_date = f" ({str(emt.time.values[time_index])[:10]})"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    emt_2d.plot(ax=axes[0], cmap="YlGnBu", cbar_kwargs={"label": emt.attrs.get("units", "EMT moisture")})
    axes[0].set_title(f"EMT prediction{title_date}")
    smips_2d.plot(
        ax=axes[1],
        cmap="YlGnBu",
        cbar_kwargs={"label": smips_label or smips.attrs.get("layer", smips.name or "SMIPS")},
    )
    axes[1].set_title(f"SMIPS tile{title_date}")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _resolve_smips_mode(smips: xr.DataArray, mode: str) -> str:
    if mode != "auto":
        return mode
    layer = " ".join(
        str(value)
        for value in (
            smips.name,
            smips.attrs.get("layer", ""),
            smips.attrs.get("collection", ""),
        )
    ).lower()
    if "smindex" in layer or "smi" in layer or "index" in layer:
        return "relative_fullness"
    return "totalbucket"


def _model_bounds(model: EMTModel) -> tuple[float, float]:
    lower = model.config.lower_bound if model.config.lower_bound is not None else model.mean_bounds[0]
    upper = model.config.upper_bound if model.config.upper_bound is not None else model.mean_bounds[1]
    if lower >= upper:
        raise ValueError("EMT model bounds are invalid; cannot map SMIPS relative fullness")
    return lower, upper


def _normalise_xy(da: xr.DataArray) -> xr.DataArray:
    out = da
    if "longitude" in out.coords and "latitude" in out.coords:
        out = out.rename(longitude="x", latitude="y")
    if "xc" in out.coords and "yc" in out.coords:
        out = out.assign_coords(x=out["xc"].isel(y=0).values, y=out["yc"].isel(x=0).values)
    if "x" not in out.coords or "y" not in out.coords:
        raise ValueError("SMIPS data must have x/y, longitude/latitude, or xc/yc coordinates")
    return out
