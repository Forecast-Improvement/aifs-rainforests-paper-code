from improver.cli import apply_rainforests_calibration
from improver.utilities.cube_manipulation import compare_coords
from improver.utilities.save import save_netcdf
import iris
import json
from datetime import datetime, timedelta
from iris.coords import DimCoord
import xarray as xr
import cartopy
import pandas as pd
import os

from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
import numpy as np


def get_calibrated_day_grid(
    ref_time, lead_time, model_config_path, model, model_approach=""
):
    """
    ref_time in yyyymmdd format as a string
    lead_time in hours as an int
    path to json file for model config
    model either "AIFS" or "HRES"
    suffix


    returns filename of saved calibrated forecast
    """
    print("Model approach,", model_approach)

    # single day data
    input_dir = f"/path/to/processed_data/{model}"

    # Parse reference time
    ref_dt = datetime.strptime(ref_time, "%Y%m%d")
    # Add lead time (hours)
    val_dt = ref_dt + timedelta(hours=int(lead_time))
    # Format back to yyyymmdd
    val_time = val_dt.strftime("%Y%m%d")

    if model == "AIFS":
        raw_path = f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-precipitation_accumulation.nc"
        fc_cube = iris.load_cube(raw_path)
        fc_cube.units = "m"

        var_cubes = [
            iris.load_cube(f)
            for f in [
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-precipitation_accumulation.nc",
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-precipitation_accumulation_from_convection.nc",
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-wind_speed.nc",
                f"{input_dir}/{val_time}T1200Z-PT0000H00M-clearsky_solar_radiation-PT24H.nc",
            ]
        ]
    else:
        raw_path = f"{input_dir}/{ref_time}T1200Z-PT00{lead_time}H00M-precipitation_accumulation-PT24H.nc"
        fc_cube = iris.load_cube(raw_path)
        fc_cube.units = "m"

        var_cubes = [
            iris.load_cube(f)
            for f in [
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-precipitation_accumulation-PT24H.nc",
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-precipitation_accumulation_from_convection-PT24H.nc",
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-cape.nc",
                f"{input_dir}/{ref_time}T1200Z-PT{lead_time:04d}H00M-wind_speed_on_pressure_levels.nc",
                f"/path/to/processed_data/HRES_solar/{val_time}T1200Z-PT0000H00M-clearsky_solar_radiation-PT24H.nc",
            ]
        ]
    print("Raw data from", raw_path)
    for cube in var_cubes:
        print("name:", cube.name())
        if cube.name() in [
            "lwe_thickness_of_precipitation_amount",
            "lwe_thickness_of_convective_precipitation_amount",
        ]:
            print("unit changed")
            cube.units = "m"

    if model == "HRES":
        # these cause issues in the IMPROVER method
        var_cubes[4].coord("latitude").bounds = None
        var_cubes[4].coord("longitude").bounds = None

        # have to remake coords without the 'coord system' attribute which doesn't seem to have an easy way to remove
        # it doesn't show up as attribute so you can't do the normal pop method
        for x in ["latitude", "longitude"]:
            old = var_cubes[4].coord(x)
            dims = var_cubes[4].coord_dims(old)
            new = DimCoord(
                points=old.points,
                standard_name=old.standard_name,
                var_name=getattr(old, "var_name", None),
                units=old.units,
                attributes=old.attributes.copy(),
            )
            if old.has_bounds():
                new.bounds = old.bounds
            var_cubes[4].remove_coord(old)
            var_cubes[4].add_dim_coord(new, dims[0])

    if model_approach == "combined_lt":
        lead_time_cube = create_new_diagnostic_cube(
            name="n_lead_days",
            units="days",
            template_cube=fc_cube,
            mandatory_attributes=generate_mandatory_attributes([fc_cube]),
            optional_attributes=fc_cube.attributes,
            # broadcast current lead-time accross all coords
            data=np.broadcast_to(
                lead_time,
                fc_cube.data.shape,
            ),
        )
        var_cubes.append(lead_time_cube)
        print("Added lead time cube")

    with open(model_config_path) as f:
        model_config = json.load(f)

    output_thresholds = [
        0.0,
        0.00001,
        0.00005,
        0.0001,
        0.0002,
        0.0004,
        0.0006,
        0.001,
        0.002,
        0.005,
        0.007,
        0.01,
        0.015,
        0.025,
        0.035,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
    ]

    fc = apply_rainforests_calibration.process(
        fc_cube,
        *var_cubes,
        model_config=model_config,
        bin_data=True,
        output_thresholds=output_thresholds,
    )

    base_dir = f"/path/to/model_output/{approach}/{model}/case_study/"
    os.makedirs(base_dir, exist_ok=True)
    calib_path = f"{base_dir}fc_{ref_time}_{lead_time}.nc"
    save_netcdf(fc, calib_path)
    return calib_path, raw_path


ref_time = "20250701"
lead_time = 24
approach = "combined_lt"
suffix = "_combined_lt"
for model in ["AIFS", "HRES"]:
    if approach:
        model_config_path = (
            f"/path/to/model_output/{approach}/{model}/model_config.json"
        )
    else:
        model_config_path = f"/path/to/model_output/{model}/model_config.json"
    calib_path, raw_path = get_calibrated_day_grid(
        ref_time, lead_time, model_config_path, model, model_approach=approach
    )
    print(calib_path, raw_path)


def expected_value(fcst_da):
    width = (
        fcst_da["threshold"]
        .diff("threshold")
        .assign_coords({"threshold": fcst_da["threshold"].values[:-1]})
    )
    mid = fcst_da["threshold"].isel(threshold=slice(None, -1)) + width * 0.5
    diff = fcst_da.diff(dim="threshold")
    if diff.mean() < 0:
        diff = -diff
    height = diff.assign_coords({"threshold": fcst_da["threshold"].values[:-1]})
    exp_val = (height * mid).sum("threshold", skipna=False)
    exp_val.attrs["units"] = fcst_da.attrs["units"]
    return exp_val
