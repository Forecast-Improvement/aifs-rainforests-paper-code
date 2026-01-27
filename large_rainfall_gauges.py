"""Find large rainfall observations that are unlikely to be measurement errors"""

import xarray as xr
import pandas as pd
import polars as pl
from train_models import get_merged_set, val, test

target_months = test
obs_path = "/path/to/obs/precipitation_accumulation-PT24H.parquet"
pred_path = "/path/to/extracted_sites/"
obs = pd.read_parquet(obs_path)
RECORD_RAINFALL = 0.96
pred_path = "/path/to/extracted_sites/"


def get_high_rainfall_gauges(obs):
    """does the following:
    1. groups by day
    2. gets the top 10 rainfall amounts for each day/group
    3. returns the median rainfall row for that top 10 (this avoids erronerous max rainfall counts)
    """
    median_rows = obs.groupby("time", group_keys=False).apply(
        lambda group: (
            group.nlargest(10, "lwe_thickness_of_precipitation_amount")
            .sort_values("lwe_thickness_of_precipitation_amount")
            .iloc[[4]]
        )
    )
    median_rows = median_rows.reset_index()
    median_rows = median_rows[
        median_rows["lwe_thickness_of_precipitation_amount"] < RECORD_RAINFALL
    ]
    median_rows = median_rows[
        ["time", "spot_index", "lwe_thickness_of_precipitation_amount"]
    ]
    return median_rows


def get_all_rainfall_gauages(obs):
    rows = obs.reset_index()
    rows = rows[["time", "spot_index", "lwe_thickness_of_precipitation_amount"]]
    return rows


def save_gauges(obs_gauges, name):

    gauges = pl.from_pandas(obs_gauges)
    gauges = gauges.with_columns(pl.col("time").cast(pl.Datetime("ns")))

    gauges_HRES = get_merged_set(target_months, gauges, ["tp"], pred_path + "HRES")
    gauges_AIFS = get_merged_set(target_months, gauges, ["tp"], pred_path + "AIFS")

    gauges_HRES.to_parquet(f"{pred_path}{name}_HRES.parquet", index=False)
    gauges_AIFS.to_parquet(f"{pred_path}{name}_AIFS.parquet", index=False)


all_gauges = get_all_rainfall_gauages(obs)
save_gauges(all_gauges, "all_test")


large_gauges = get_high_rainfall_gauges(obs)
save_gauges(large_gauges, "large_gauages_test")
