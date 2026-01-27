"""Regrid AIFS data using earthkit to a standard grid system compatible with IMPROVER"""

import earthkit.regrid as ekr
import earthkit.data as ekd
import datetime as dt
import pandas as pd
import os

output_folder = "/path/to/aifs_data_ekr"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def regrid_aifs_data(var_name, start_date, end_date, var_list):
    """Regrid AIFS data for a given variable over a date range for use with IMPROVER."""
    for month_start in pd.date_range(start_date, end_date, freq="MS"):
        month_str = month_start.strftime("%Y_%m")
        path = f"/path/to/aifs/monthly/{var_name}_{month_str}.grib"
        output = f"{output_folder}/{var_name}_{month_str}.grib"
        if os.path.exists(output):
            print("Skipping existing file:", output)
            continue
        data = ekd.from_source("file", path)
        data = data.sel(shortName=var_list)  # choose subset of variables
        print("Regridding:", output)
        data = ekr.interpolate(data, out_grid={"grid": [0.25, 0.25]}, method="linear")
        data.save(output)


# wind
# date ranges 2024_03 to 2025_10
start_date_wind = dt.datetime(2024, 3, 1)
end_date_wind = dt.datetime(2025, 10, 1)
regrid_aifs_data("wind", start_date_wind, end_date_wind, ["u", "v"])

# precip
# date ranges 2024_03 to 2025_10
start_date_precip = dt.datetime(2024, 3, 1)
end_date_precip = dt.datetime(2025, 10, 1)
regrid_aifs_data("precip", start_date_precip, end_date_precip, ["cp", "tp"])
