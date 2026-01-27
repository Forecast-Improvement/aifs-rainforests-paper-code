"""Aggegating and processing AIFS forecast data"""

import xarray as xr
import datetime as dt
import pandas as pd
import numpy as np
from improver.utilities.save import save_netcdf

grib_loc = "/path/to/aifs_ekr_grib"


def process_aifs_to_netcdf(var_filename, start_date, end_date, std_names):
    """Process AIFS data and save as netCDF files for IMPROVER ingestion"""
    for month_start in pd.date_range(start_date, end_date, freq="MS"):
        month_str = month_start.strftime("%Y_%m")
        path = f"{grib_loc}/{var_filename}_{month_str}.grib"
        grid_data = xr.load_dataset(path, engine="cfgrib")
        # limit to australia region
        grid_data = grid_data.sel(latitude=slice(5, -60), longitude=slice(75, 180))
        # ensure units are correct
        if "tp" in std_names or "cp" in std_names:
            for var in ["tp", "cp"]:
                if grid_data[var].units == "kg m**-2":
                    with xr.set_options(keep_attrs=True):
                        grid_data[var] = grid_data[var] * 0.001
                        grid_data[var]["units"] = "m"
                elif grid_data[var].units != "m":
                    print(f"Unrecognised unit: {grid_data[var].units}")
        elif "wind_speed" in std_names:
            grid_data["inst_wind_speed"] = np.sqrt(
                grid_data["u"] ** 2 + grid_data["v"] ** 2
            )
        else:
            print("Invalid variable types")
            print(std_names)
            return
        # reassign attributes to make iris conversion work
        grid_data = grid_data.assign_coords(
            latitude=grid_data.latitude.astype("float32"),
            longitude=grid_data.longitude.astype("float32"),
            forecast_period=(grid_data.step / np.timedelta64(1, "s")).astype(np.int32),
        )
        grid_data["forecast_period"].attrs["units"] = "seconds"
        if "tp" in std_names or "cp" in std_names:
            # include 0 because diff will remove initial timestep
            daily_interval = [str(hours) + "h" for hours in range(0, 240 + 24, 24)]
            # aggregate to 24h periods, taking difference of cumulative values
            grid_data = grid_data.sel(step=daily_interval).diff("step")
        elif "wind_speed" in std_names:
            # skip time period 0 (as we are not interested in time difference 0)
            daily_interval = [str(hours) + "h" for hours in range(24, 240 + 24, 24)]
            # time = 4 because our data is in 6 hourly intervals, so 4 timesteps = 24 hours
            grid_data["wind_speed"] = (
                grid_data["inst_wind_speed"].rolling(step=4).mean()
            )
            # select daily data
            grid_data = grid_data.sel(step=daily_interval)

        # rename variables
        grid_data = grid_data.rename(std_names)
        # dis-aggregate to valid time and lead time
        for bt in grid_data.time.values:
            subset_bt = grid_data.sel(time=bt)
            for lt in subset_bt.step.values:
                subset_bt_lt = subset_bt.sel(step=lt)

                for var, var_name in std_names.items():
                    da = subset_bt_lt[var_name]
                    # assign attributes from original data
                    da.attrs = grid_data[var_name].attrs
                    da.attrs["standard_name"] = var_name
                    # format for IMPROVER conversion
                    da = da.drop_vars(["step"])
                    if var == "cp" or var == "tp":
                        da = da.drop_vars(["surface"])
                    elif var == "wind_speed":
                        da = da.assign_coords(
                            isobaricInhPa=da.isobaricInhPa.astype("float32")
                        )
                    da = da.rename(
                        {"time": "forecast_reference_time", "valid_time": "time"}
                    )
                    da = da.assign_coords(
                        time=da.time.astype("datetime64[s]").astype(int)
                    )
                    da["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
                    # create filenames
                    valid_time = pd.to_datetime(
                        da.time.values, unit="s", origin="unix"
                    ).strftime(
                        "%Y%m%dT%H%MZ"
                    )  # YYYYMMDDTHHMMZ format
                    lead_time = (
                        "PT" + str(lt // np.timedelta64(1, "h")).zfill(4) + "H00M"
                    )
                    if var == "cp":
                        filename = f"{valid_time}-{lead_time}-precipitation_accumulation_from_convection.nc"
                    elif var == "tp":
                        filename = (
                            f"{valid_time}-{lead_time}-precipitation_accumulation.nc"
                        )
                    elif var == "wind_speed":
                        filename = f"{valid_time}-{lead_time}-wind_speed.nc"
                    else:
                        print("Invalid variable")
                        return
                    # save files
                    cube_da = da.to_iris()
                    print(f"Saving {filename}")
                    save_netcdf(
                        cube_da,
                        f"/path/to/processed_data/AIFS/{filename}",
                    )


# wind processing
process_aifs_to_netcdf(
    "wind",
    dt.datetime(2024, 3, 1),
    dt.datetime(2025, 10, 1),
    {"wind_speed": "wind_speed"},
)

# precip processing
process_aifs_to_netcdf(
    "precip",
    dt.datetime(2024, 3, 1),
    dt.datetime(2025, 10, 1),
    {
        "tp": "lwe_thickness_of_precipitation_amount",
        "cp": "lwe_thickness_of_convective_precipitation_amount",
    },
)
