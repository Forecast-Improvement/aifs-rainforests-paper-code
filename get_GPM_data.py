"""Aggregate GPM satellite observations"""

import xarray as xr
from datetime import datetime, timedelta
import calendar
import pandas as pd
import os
from train_models import test, val

base_dir = "/path/to/obs/gpm/"


def gpm_aggregations_from_m_y(year, month, var_type, path):
    """Generate a list of filenames with base / reference time that is within the given month"""
    start = datetime(year, month, 1)
    ndays = calendar.monthrange(year, month)[1] + 10
    print(ndays)
    results = []
    for day in range(ndays):
        base_time = start + timedelta(days=day) + timedelta(hours=12)
        daily_accumulations = []
        for hours in range(1, 25):
            valid_str = (base_time + timedelta(hours=hours)).strftime("%Y%m%dT%H00Z")
            lead_str = f"PT0000H00M"
            fname = f"{path}{valid_str}-{lead_str}-{var_type}.nc"

            if os.path.exists(fname):
                print("Adding", fname, hours)
                daily_accumulations.append(xr.load_dataset(fname))
            else:
                print("Not found:", fname)
        total_24h = xr.combine_by_coords(daily_accumulations).sum(dim="time")
        valid_time = base_time + pd.Timedelta(hours=24)
        total_24h = total_24h.expand_dims(time=[valid_time])
        results.append(total_24h)
    return xr.combine_by_coords(results)


if __name__ == "__main__":
    # select testing or val months
    selected_months = val
    gpm_obs = []
    for date in selected_months:
        y, m = date.split("_")
        monthly_agg = gpm_aggregations_from_m_y(
            int(y), int(m), "precipitation_accumulation-PT01H", base_dir
        )
        gpm_obs.append(monthly_agg)
    gpm_all_obs = xr.combine_by_coords(gpm_obs)
    gpm_all_obs.to_netcdf(
        f"/path/to/processed_data/GPM/{selected_months}_GPM_obs.nc"
    )
