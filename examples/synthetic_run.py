from __future__ import annotations

import numpy as np

from topmodel_dispatch_hybrid import (
    Covariates,
    DisaggregationConfig,
    WeatherForcing,
    disaggregate_smips,
)


def main() -> None:
    config = DisaggregationConfig(subgrid_factor=33)
    smips = np.array(
        [
            [0.12, 0.22],
            [0.30, 0.38],
        ]
    )

    fine_shape = tuple(dim * config.subgrid_factor for dim in smips.shape)
    y, x = np.indices(fine_shape)

    valley = -np.abs(x - fine_shape[1] * 0.45)
    northness = 1.0 - y / fine_shape[0]

    covariates = Covariates(
        twi=valley + 0.2 * np.sin(y / 5.0),
        hli=x / fine_shape[1] + 0.4 * northness,
        holding_capacity=0.5 + 0.3 * np.sin(x / 12.0) + 0.2 * np.cos(y / 15.0),
        vegetation_cover=np.clip(0.35 + 0.25 * northness + 0.1 * np.sin(x / 8.0), 0, 1),
        infiltration_capacity=0.6 + 0.2 * np.cos((x + y) / 11.0),
    )
    weather = WeatherForcing(
        antecedent_precip=np.array([[0.0, 4.0], [18.0, 35.0]]),
        potential_et=5.0,
    )

    result = disaggregate_smips(smips, covariates, weather, config)

    print("Coarse SMIPS:")
    print(smips)
    print("\nRecovered block means:")
    print(block_means(result.soil_moisture_30m, config.subgrid_factor))
    print("\nConnectivity:")
    print(np.round(result.connectivity, 3))
    print("\n30 m output shape:", result.soil_moisture_30m.shape)


def block_means(values: np.ndarray, factor: int) -> np.ndarray:
    coarse_rows = values.shape[0] // factor
    coarse_cols = values.shape[1] // factor
    means = np.zeros((coarse_rows, coarse_cols))
    for row in range(coarse_rows):
        for col in range(coarse_cols):
            block = values[
                row * factor : (row + 1) * factor,
                col * factor : (col + 1) * factor,
            ]
            means[row, col] = np.nanmean(block)
    return means


if __name__ == "__main__":
    main()

