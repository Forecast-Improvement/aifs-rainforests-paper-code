"""Generate solar data from ancillary data using IMPROVER function"""

from improver.cli import generate_clearsky_solar_radiation
from improver.utilities.save import save_netcdf
import pandas as pd
import numpy as np
import datetime as dt
import iris
import xarray as xr

output_base_dir = "/path/to/processed_data"
models = ["HRES", "AIFS"]
for model in models:
    output_dir = f"{output_base_dir}/HRES_solar"
    if model == "HRES":
        target_grid_file = "/path/to/ecmwf_hres_orography.nc"
        surface_altitude_file = "/path/to/ecmwf_hres_orography.nc"
    if model == "AIFS":
        target_grid_file = "/path/to/aifs_data_ekr/ancil_regridded.nc"
        surface_altitude_file = "/path/to/aifs_data_ekr/ancil_regridded.nc"
        grid_data = xr.load_dataset(
            "/path/to/aifs_data_ekr/ancil_regridded.grib",
            engine="cfgrib",
        )
        grid_data = grid_data.sel(latitude=slice(5, -60), longitude=slice(75, 180))
        grid_data["surface_altitude"] = (
            grid_data["z"] / 9.80665
        )  # convert from geopotential to altitude

        # format so we can save as netcdf and access it further below
        grid_data = grid_data.drop_vars(["surface"])
        da = grid_data["surface_altitude"]
        da.attrs["units"] = "m"
        da["standard_name"] = "surface_altitude"
        da["long_name"] = "Surface Altitude"
        da = da.assign_coords(
            latitude=da.latitude.astype("float32"),
            longitude=da.longitude.astype("float32"),
        )
        cube = da.to_iris()
        cube.remove_coord("time")
        cube.remove_coord("forecast_period")
        cube.remove_coord("forecast_reference_time")
        save_netcdf(cube, target_grid_file)

    dates = pd.date_range(
        dt.datetime(2024, 3, 1, 12, 0), dt.datetime(2025, 11, 11, 12, 0), freq="24H"
    )
    target_grid = iris.load_cube(target_grid_file)
    surface_altitude = iris.load_cube(surface_altitude_file)
    linke_turbidity = target_grid.copy(data=np.ones(target_grid.shape) * 3)
    linke_turbidity.rename("linke_turbidity")
    print("Dims of target grid:", target_grid.shape)
    print("Dims of surface altitude:", surface_altitude.shape)
    for d in dates:
        print(d)
        valid_time_str = d.strftime("%Y%m%dT%H%MZ")
        cube = generate_clearsky_solar_radiation.process(
            target_grid,
            surface_altitude,
            linke_turbidity=linke_turbidity,
            time=d,
            accumulation_period=24,
            temporal_spacing=30,
        )
        save_netcdf(
            cube,
            f"{output_dir}/{valid_time_str}-PT0000H00M-clearsky_solar_radiation-PT24H.nc",
        )
