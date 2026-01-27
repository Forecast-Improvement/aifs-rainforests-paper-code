"""Aggegating and processing HRES forecast data"""

import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from improver.utilities.save import save_netcdf
import os
import multiprocessing


def process_basetime(basetime):
    input_base_dir = "/path/to/ingest"
    output_dir = "/path/to/processed_data/HRES"
    var_dict = {
        "precipitation_accumulation": "lwe_thickness_of_precipitation_amount",
        "precipitation_accumulation_from_convection": "lwe_thickness_of_convective_precipitation_amount",
        "cape": "convective_available_potential_energy",
        "wind_speed_on_pressure_levels": "wind_speed",
    }

    formatted_basetime = basetime.strftime("%Y%m%dT%H%MZ")
    for lead_time in range(24, 240 + 24, 24):
        valid_time = basetime + pd.offsets.DateOffset(hours=lead_time)
        formatted_valid_time = valid_time.strftime("%Y%m%dT%H%MZ")
        for v, cf_name in var_dict.items():
            output_var_name = v + "-PT24H" if "precip" in v else v
            output_path = f"{output_dir}/{formatted_valid_time}-PT{lead_time:04d}H00M-{output_var_name}.nc"
            if os.path.exists(output_path):
                return
            if "precip" in v:
                input_lead_times = list(range(lead_time, lead_time - 24, -3))
            else:
                input_lead_times = list(range(lead_time, lead_time - 24, -6))
            input_paths = []
            for ilt in input_lead_times:
                input_time = basetime + pd.offsets.DateOffset(hours=ilt)
                formatted_input_time = input_time.strftime("%Y%m%dT%H%MZ")
                if "precip" in v:
                    filename = f"{formatted_input_time}-PT{ilt:04d}H00M-{v}-PT03H.nc"
                else:
                    filename = f"{formatted_input_time}-PT{ilt:04d}H00M-{v}.nc"
                input_paths.append(
                    f"{input_base_dir}/{formatted_basetime}/ecmwf_hres/ecmwf_hres_t{ilt:04d}/{filename}"
                )
            if not (all([os.path.exists(x) for x in input_paths])):
                print(f"Data missing for {output_path}")
                continue
            ds_arr = [xr.open_dataset(p) for p in input_paths]
            ds = xr.concat(ds_arr, dim="time")
            if "precip" in v:
                da = ds[cf_name].sum(dim="time")
            elif v == "cape":
                da = ds[cf_name].max(dim="time")
            elif v == "wind_speed_on_pressure_levels":
                da = ds.sel(pressure=800)[cf_name].mean(dim="time")
            fp_coord = ds_arr[0]["forecast_period"].values
            if type(fp_coord) == np.ndarray:
                fp = fp_coord[0]
            else:
                fp = fp_coord
            da = da.assign_coords(
                forecast_period=(fp / np.timedelta64(1, "s")).astype(np.int32)
            )
            da["forecast_period"].attrs["units"] = "seconds"
            da.attrs = ds[cf_name].attrs
            time_coord = ds_arr[0]["time"].values
            if type(time_coord) == np.ndarray:
                tc = time_coord[0]
            else:
                tc = time_coord
            da = da.assign_coords(time=tc.astype("datetime64[s]").astype(int))
            da["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
            cube = da.to_iris()
            save_netcdf(cube, output_path)


if __name__ == "__main__":

    dates = pd.date_range(
        dt.datetime(2024, 3, 1, 12, 0), dt.datetime(2025, 11, 11, 12, 0), freq="24H"
    )

    multiprocessing.set_start_method("forkserver")
    with multiprocessing.Pool(maxtasksperchild=100) as p:
        p.map(process_basetime, dates, chunksize=1)
        p.close()
        p.join()
