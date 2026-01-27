"""AIFS data processing script to convert gridded netCDF files to site-specific netCDF and parquet files"""

import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import calendar
import pandas as pd
from improver.utilities.save import save_netcdf
import os
import iris


def extract_sites(grid_data, site_locations, var_name):
    """ Extract a subset of gridded data for specific site locations """
    lat_diff = grid_data["latitude"].values[1] - grid_data["latitude"].values[0]
    lon_diff = grid_data["longitude"].values[1] - grid_data["longitude"].values[0]
    tolerance = np.minimum(np.abs(lat_diff), np.abs(lon_diff))
    min_lat, max_lat = (
        np.min(grid_data["latitude"].values) - tolerance,
        np.max(grid_data["latitude"].values) + tolerance,
    )
    min_lon, max_lon = (
        np.min(grid_data["longitude"].values) - tolerance,
        np.max(grid_data["longitude"].values) + tolerance,
    )
    site_locations = site_locations.where(
        (site_locations["latitude"] >= min_lat)
        & (site_locations["latitude"] <= max_lat)
        & (site_locations["longitude"] >= min_lon)
        & (site_locations["longitude"] <= max_lon),
        drop=True,
    )
    if var_name in ["cp", "tp", "cape"]:
        # use nearest neighbour
        site_data = grid_data.sel(
            latitude=site_locations["latitude"],
            longitude=site_locations["longitude"],
            method="nearest",
            tolerance=tolerance,
        )
    elif var_name in ["wind_speed", "solar"]:
        # use linear interpolation
        site_data = grid_data.interp(
            latitude=site_locations["latitude"],
            longitude=site_locations["longitude"],
            method="linear",
        )
    else:
        print("Invalid variable for site extraction:", var_name)
        return None
    site_data = site_data.drop_vars(["latitude", "longitude"])
    return site_data


def filenames_from_m_y(year, month, var_type, path):
    """Generate a list of filenames with base / reference time that is within the given month"""
    start = datetime(year, month, 1)
    ndays = calendar.monthrange(year, month)[1]
    results = []
    for day in range(ndays):
        base_time = start + timedelta(days=day)
        start_day = 1
        end_day = 11
        if var_type == "clearsky_solar_radiation-PT24H":
            end_day += 1
        for n in range(start_day, end_day):
            hours = n * 24
            lead = timedelta(days=n)
            valid_time = base_time + lead
            valid_str = valid_time.strftime("%Y%m%d") + "T1200Z"
            if var_type == "clearsky_solar_radiation-PT24H":
                hours = 0
            lead_str = f"PT{hours:04d}H00M"
            fname = f"{path}{valid_str}-{lead_str}-{var_type}.nc"
            if fname not in results:
                results.append(fname)
    return results


def process_file(path, var_name, site_path, do_extract):
    """Open file and extract site locations"""
    print(f"Processing {path}")
    site_locations = xr.open_dataset(site_path)
    # site_locations.load()
    try:
        grid_data = xr.open_dataset(path)
        grid_data.load()
    except:
        print(f"Error opening file for {path}")
        return
    try:
        if do_extract:
            da = extract_sites(grid_data, site_locations, var_name)
        else:
            da = grid_data
        da.load()
        da = da.expand_dims(dim="forecast_reference_time")
        print("Dims before", da.dims)
        if var_name == "solar":
            # need to remove this dim
            da = da.isel(bnds=0).reset_coords(drop=True)
            da = da.assign_coords(forecast_reference_time=da.time_bnds)
        else:
            da = da.expand_dims(dim="forecast_period")
        print("Dims after", da.dims)
    except Exception as e:
        print(f"Error extracting sites for {path}")
        print(e)
        return
    return da


def format_month_file(ds, var_type, var_name, out_path, model):
    """ convert to iris cube and save as netcdf """
    ds[var_name] = ds[var_name].astype("float32")
    ds = ds.assign_coords(
        forecast_reference_time=ds.forecast_reference_time,
        latitude=ds.latitude,
        longitude=ds.longitude,
    )
    # requirements to make .to_iris work
    if var_type == "solar":
        print("solar")
    else:
        ds = ds.assign_coords(
            forecast_period=(ds.forecast_period / np.timedelta64(1, "s")).astype(
                "int32"
            )
        )
        ds["forecast_period"].attrs["units"] = "seconds"
    # select specific variable
    da = ds[var_name]
    da.attrs = ds[var_name].attrs
    # convert to iris cube and save as netcdf
    cube = da.to_iris()
    save_netcdf(cube, f"{out_path}.nc")


def format_file(ds, var_type, var_name, out_path, model):
    ### Format and save extracted site data to df->parquet and netcdf cube ###

    # convert to dataframe then save as parquet
    df = ds.to_dataframe().reset_index()
    print(df)
    print(df.columns)
    print("Varname:", var_name)
    if var_type != "solar":
        # solar has differing column structure
        df = df[
            ["time", "forecast_reference_time", "forecast_period", "site_id", var_name]
        ]
    else:
        df = df[["forecast_reference_time", "site_id", var_name]]
    df.to_parquet(f"{out_path}.parquet")
    # convert to iris cube and save as netcdf
    ds[var_name] = ds[var_name].astype("float32")
    ds = ds.assign_coords(
        forecast_reference_time=ds.forecast_reference_time, site_id=ds.site_id
    )
    # requirements to make .to_iris work
    if var_type == "solar":
        # ds["forecast_reference_time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
        print("solar")
    else:
        ds = ds.assign_coords(
            forecast_period=(ds.forecast_period / np.timedelta64(1, "s")).astype(
                "int32"
            )
        )
        ds["forecast_period"].attrs["units"] = "seconds"
    # select specific variable
    da = ds[var_name]
    da.attrs = ds[var_name].attrs
    cube = da.to_iris()
    # the to_iris conversion leaves the site_id as an auxillary variable even though it was a dimension coord with a name in the xarray dataArray,
    # since site_id is an auxillary variable in the netcdf it shows up as having a missing dimension name "dim2"
    # to fix this we need to promote the site_id to a dimension variable but this requires that the site_id variable is monotonic
    # to achieve this we sort the cube by the site_id first then use the promote function
    cube = sort_cube_by_coord(cube, "site_id")
    iris.util.promote_aux_coord_to_dim_coord(cube, "site_id")
    save_netcdf(cube, f"{out_path}.nc")


def sort_cube_by_coord(cube, coord):
    # code taken from https://gist.github.com/pelson/9763057
    # sorts a cube by a dimension
    # this is required to ensure a coord is monotonic and can then be promoted
    coord_to_sort = cube.coord(coord)
    assert coord_to_sort.ndim == 1, "One dim coords only please."
    (dim,) = cube.coord_dims(coord_to_sort)
    index = [slice(None)] * cube.ndim
    index[dim] = np.argsort(coord_to_sort.points)
    return cube[tuple(index)]


if __name__ == "__main__":
    do_extract = False
    start_date = datetime(2024, 3, 1)
    end_date = datetime(2025, 10, 1)
    # dict to map variable types to filenames and long names
    var_types = ["solar", "cp", "tp", "wind_speed"]
    name_dict = {
        "tp": {
            "filename": "precipitation_accumulation",
            "long_name": "lwe_thickness_of_precipitation_amount",
        },
        "cp": {
            "filename": "precipitation_accumulation_from_convection",
            "long_name": "lwe_thickness_of_convective_precipitation_amount",
        },
        "wind_speed": {"filename": "wind_speed", "long_name": "wind_speed"},
        "solar": {
            "filename": "clearsky_solar_radiation-PT24H",
            "long_name": "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
        },
    }
    HRES_var_types = ["cape"] + var_types.copy()
    HRES_names_dict = {
        "tp": {
            "filename": "precipitation_accumulation-PT24H",
            "long_name": "lwe_thickness_of_precipitation_amount",
        },
        "cp": {
            "filename": "precipitation_accumulation_from_convection-PT24H",
            "long_name": "lwe_thickness_of_convective_precipitation_amount",
        },
        "wind_speed": {
            "filename": "wind_speed_on_pressure_levels",
            "long_name": "wind_speed",
        },
        "solar": {
            "filename": "clearsky_solar_radiation-PT24H",
            "long_name": "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
        },
        "cape": {
            "filename": "cape",
            "long_name": "convective_available_potential_energy",
        },
    }
    BASE_OUT_PATH = "/path/to/extracted_sites/"
    BASE_SOURCE_PATH = "/path/to/processed_data/"
    site_path = "/path/to/site_cube.nc"
    models = ["AIFS", "HRES"]
    for model in models:
        out_path = f"{BASE_OUT_PATH}{model}/"
        source_path = f"{BASE_SOURCE_PATH}{model}/"
        if model == "HRES":
            var_types = HRES_var_types
            name_dict = HRES_names_dict
        # a non-extract option to instead create monthly sets of daily aggregations
        if not do_extract:
            out_path = out_path.replace("extracted_sites", "monthly_processed_data")
        for month_start in pd.date_range(start_date, end_date, freq="MS"):
            year = month_start.strftime("%Y")
            month = month_start.strftime("%m")
            for var in var_types:
                export_path = (
                    f"{out_path}{name_dict[var]["filename"]}_{year}_{month}".replace(
                        "-PT24H", ""
                    ).replace("_on_pressure_levels", "")
                )
                file_list = filenames_from_m_y(
                    int(year), int(month), name_dict[var]["filename"], source_path
                )
                darrays = []
                # skip if files already exist
                if os.path.exists(export_path + ".nc") or os.path.exists(
                    export_path + ".parquet"
                ):
                    print(f"Skipping existing files for {var} {year}-{month}")
                    continue
                # extract sites for each file
                for file in file_list:
                    if os.path.exists(file) == False:
                        print(f"File does not exist: {file}")
                    else:
                        da = xr.open_dataset(file)
                        darrays.append(process_file(file, var, site_path, do_extract))
                print(f"Extracted {len(darrays)} data arrays for {var} {year}-{month}")
                if len(darrays) == 0:
                    print(
                        f"No data arrays extracted for {var} {year}-{month}, skipping save."
                    )
                else:
                    print(len(darrays))
                    print(darrays[0])
                    if var == "solar":
                        # solar has no forecast period dim so need to concat differently
                        concat_ds = xr.concat(darrays, dim="forecast_reference_time")
                    else:
                        concat_ds = xr.combine_by_coords(darrays)
                    if do_extract:
                        format_file(
                            concat_ds,
                            var,
                            name_dict[var]["long_name"],
                            export_path,
                            model,
                        )
                    else:
                        format_month_file(
                            concat_ds,
                            var,
                            name_dict[var]["long_name"],
                            export_path,
                            model,
                        )
