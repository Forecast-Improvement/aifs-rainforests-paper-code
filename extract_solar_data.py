"""AIFS data processing script to convert gridded netCDF files to site-specific netCDF and parquet files"""

import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import calendar
import pandas as pd
import os
from extract_sites import extract_sites, process_file, format_file, format_month_file


def get_filenames(start_date, end_date, path, var_name):
    """ Generate a list of filenames with base / reference time that is within the given month """
    results = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        valid_time = day
        valid_str = valid_time.strftime("%Y%m%d") + "T1200Z"
        hours = 0
        lead_str = f"PT{hours:04d}H00M"
        fname = f"{path}{valid_str}-{lead_str}-{var_name}.nc"
        if fname not in results:
            results.append(fname)
    return results


if __name__ == "__main__":
    # determine whether to extract by site or stay gridded and aggregate
    DO_EXTRACT = False
    long_name = "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time"
    start_date = datetime(2024, 3, 1)
    end_date = datetime(2025, 10, 1)
    source_path = "/path/to/processed_data/AIFS/"
    out_path = "/path/to/extracted_sites/AIFS/solar/"
    site_path = "/path/to/site_cube.nc"
    var_name = "clearsky_solar_radiation-PT24H"
    models = ["HRES", "AIFS"]
    for model in models:
        if model == "HRES":
            source_path = "/path/to/processed_data/HRES_solar/"
            out_path = "/path/to/extracted_sites/HRES/solar/"
        if not DO_EXTRACT:
            out_path = out_path.replace("extracted_sites", "monthly_processed_data")
        darrays = []
        file_list = get_filenames(start_date, end_date, source_path, var_name)
        export_path = f"{out_path}solar_extracted"
        for file in file_list:
            if os.path.exists(file) == False:
                print(f"File does not exist: {file}")
            else:
                da = xr.open_dataset(file)
                darrays.append(process_file(file, "solar", site_path, DO_EXTRACT))
        concat_ds = xr.concat(darrays, dim="forecast_reference_time")
        # slightly different handling since extracted file has site_id index
        # whereas non-extracted has lat/long
        if DO_EXTRACT:
            format_file(concat_ds, "solar", long_name, export_path, model)
        else:
            format_month_file(concat_ds, "solar", long_name, export_path, model)
