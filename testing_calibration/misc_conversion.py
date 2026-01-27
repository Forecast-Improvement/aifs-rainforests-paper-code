import xarray as xr
import pandas as pd
import scores

from scores.probability import crps_cdf
import matplotlib.pyplot as plt

AIFS_path = "/path/to/model_output/AIFS/calibrated_months/calibrated_precip_2024_06.nc"
HRES_path = "/path/to/model_output/HRES/calibrated_months/calibrated_precip_2024_06.nc"
aifs_val = xr.load_dataset(AIFS_path)
hres_val = xr.load_dataset(HRES_path)


aifs_pd = aifs_val.to_dataframe().reset_index()
obs_path = "/path/to/obs/precipitation_accumulation-PT24H.parquet"
obs = pd.read_parquet(obs_path).reset_index()
print("size", obs.size)


# merge together predictions and obs
# so we can get the forecast_period and forecast_reference_time
merged_set = aifs_pd.merge(
    obs, right_on=["time", "spot_index"], left_on=["time", "site_id"]
)
merged_set = merged_set[
    [
        "forecast_period",
        "forecast_reference_time",
        "site_id",
        "lwe_thickness_of_precipitation_amount",
    ]
]
merged_set = merged_set.set_index(
    ["forecast_period", "forecast_reference_time", "site_id"]
)
# print(merged_set.size)
merged_set_dedup = merged_set[~merged_set.index.duplicated()]
# print("size", merged_set_dedup.size)
merged_set_x = merged_set_dedup.to_xarray()
# print(merged_set_x)


# -- initial workaround, issue was that the time index ends up being non-unique --
# obs.rename(columns={"spot_index":"site_id"}, inplace=True)
# obs = obs.set_index(['time','site_id'])
# obs_x = obs.to_xarray()


da = aifs_val["probability_of_lwe_thickness_of_precipitation_amount_above_threshold"]
obs_x = obs_x["lwe_thickness_of_precipitation_amount"]

print(da)
print(obs_x)

print("Generating crps")


crps = crps_cdf(da, obs_x, threshold_dim="threshold")
print(crps.total.values.round(3))
