import xarray as xr
import pandas as pd
from train_models import get_merged_set

obs_path = "/path/to/obs/precipitation_accumulation-PT24H.parquet"
obs = pd.read_parquet(obs_path).reset_index()
print(obs.head())

RECORD_RAINFALL = 0.96


# this dense number does the following:
# 1. groups by day
# 2. gets the top 10 rainfall amounts for each day/group
# 3. returns the median rainfall row for that top 10 (this avoids erronerous max rainfall counts)
median_rows = obs.groupby("time").apply(
    lambda group: (
        group.nlargest(10, "lwe_thickness_of_precipitation_amount")
        .sort_values("lwe_thickness_of_precipitation_amount")
        .iloc[4]
    )
)
median_rows = median_rows[
    median_rows["lwe_thickness_of_precipitation_amount"] < RECORD_RAINFALL
]
print(median_rows)
