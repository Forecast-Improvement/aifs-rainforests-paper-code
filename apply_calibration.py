"""Apply RainForests calibration to extracted sites OR gridded data depending on USING_GRIDDED"""

from improver.cli import apply_rainforests_calibration
from improver.utilities.save import save_netcdf
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
import iris
import json
import pandas as pd
import datetime as dt
import calendar
from model_config import THRESHOLDS_POS
import numpy as np
import os

# import sets of months
from train_models import train, test, val

# determine whether to calibrate on extracted sites or gridded data
USING_GRIDDED = False
# determine whether to calibrate on training, testing or validation months
SELECTED_MONTHS = test
START_DATE = dt.datetime(2024, 3, 1)
END_DATE = dt.datetime(2025, 10, 1)

# input to rainforests throws errors when using numpy array
# list comprehension instead
# THRESHOLDS = [t / 1000.0 for t in THRESHOLDS_POS]
THRESHOLDS = [
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


def get_cube_subset(cube, ref_time, lead_time, var_type):
    """constrain cube to the supplied reference time and lead time, behaviour depends on variable type
    Returns the subsetted cube"""
    if var_type in ["tp", "cp"]:
        # it appears m has been dropped in the extract sites step
        cube.units = "m"
    if var_type == "solar":
        valid_time = ref_time + pd.Timedelta(days=lead_time)
        constr_val = iris.Constraint(
            forecast_reference_time=lambda cell: cell.point == valid_time
        )
        output = cube.extract(constr_val)
    else:
        lead_time_s = lead_time * 86400
        constr_ref = iris.Constraint(
            forecast_reference_time=lambda cell: cell.point == ref_time
        )
        constr_lt = iris.Constraint(
            forecast_period=lambda cell: cell.point == lead_time_s
        )
        output = cube.extract(constr_ref & constr_lt)
    # check if missing data before return
    if 0 in cube.shape:
        print(f"Missing data for {ref_time}, {lead_time}, {var_type}")
        return None
    else:
        return output


# potential approaches
# "combined_param" - trained using a parameter set optimised with CRPS
# "multi_param" - trained using seperate parameters for various rain thresholds
# "combined_lt" - trained using a single model for each lead time
# "combined_lt_new_param" - same as above but with new parameters
# "default" - uses original parameters and one model per lead time / thresholds

for approach in ["combined_lt_new_param"]:
    for model in ["AIFS", "HRES"]:
        var_types = ["wind", "cp", "tp", "solar"]
        # slightly different process if we are using GPM data for validation or the extracted site data
        if not USING_GRIDDED:
            input_dir = "/path/to/extracted_sites/"
        else:
            input_dir = "/path/to/monthly_processed_data/"
        if model == "AIFS":
            input_dir += "AIFS"
        else:
            input_dir += "HRES"
            var_types.append("cape")
        config_dir = f"/path/to/model_output/{approach}/{model}/model_config.json"
        with open(config_dir) as f:
            model_config = json.load(f)
        for month_start in pd.date_range(START_DATE, END_DATE, freq="MS"):
            # get full month
            year = month_start.year
            month = month_start.month
            print(year, month)
            num_days_of_month = calendar.monthrange(year, month)[1]
            end_date = dt.datetime(year, month, num_days_of_month)
            date_range = pd.date_range(start=month_start, end=end_date, freq="D")
            # offset to get the correct time of day used
            offset = dt.timedelta(hours=12)
            if not USING_GRIDDED:
                output_dir = f"/path/to/model_output/{approach}/{model}/calibrated_months/calibrated_precip_{year}_{month:02}.nc"
            else:
                output_dir = f"/path/to/model_output/{approach}/{model}/calibrated_gridded_months/calibrated_precip_{year}_{month:02}.nc"
            if f"{year}_{month:02}" not in SELECTED_MONTHS:
                print(f"Skipping for now, '{year}_{month:02}' not in selected set")
            elif os.path.exists(output_dir):
                print(f"'{year}_{month:02d}' already exists, skipping")
            else:
                fc_cube = iris.load_cube(
                    f"{input_dir}/precipitation_accumulation_{year}_{month:02}.nc"
                )
                var_cubes = [
                    iris.load_cube(f)
                    for f in [
                        f"{input_dir}/wind_speed_{year}_{month:02}.nc",
                        f"{input_dir}/precipitation_accumulation_from_convection_{year}_{month:02}.nc",
                        f"{input_dir}/precipitation_accumulation_{year}_{month:02}.nc",
                        f"{input_dir}/solar/solar_extracted.nc",
                    ]
                ]
                if model == "HRES":
                    cape = iris.load_cube(f"{input_dir}/cape_{year}_{month:02}.nc")
                    var_cubes.append(cape)
                calibs = iris.cube.CubeList()
                for ref_time in date_range:
                    for lead_time in range(1, 11):
                        test_cubes = []
                        for i, var in enumerate(var_types):
                            cube_subset = get_cube_subset(
                                var_cubes[i], ref_time + offset, lead_time, var
                            )
                            test_cubes.append(cube_subset)
                        test_fc_cube = get_cube_subset(
                            fc_cube, ref_time + offset, lead_time, "tp"
                        )
                        # check all cubes exist
                        all_present = True
                        for cube in test_cubes + [test_fc_cube]:
                            if not cube:
                                print("Missing data, skipping time period", year, month)
                                all_present = False
                        if all_present:
                            # if combining lead time, then we require an additional input variable
                            if approach in ["combined_lt", "combined_lt_new_param"]:
                                lead_time_cube = create_new_diagnostic_cube(
                                    name="n_lead_days",
                                    units="days",
                                    template_cube=test_fc_cube,
                                    mandatory_attributes=generate_mandatory_attributes(
                                        [test_fc_cube]
                                    ),
                                    optional_attributes=test_fc_cube.attributes,
                                    # broadcast current lead-time accross all coords
                                    data=np.broadcast_to(
                                        lead_time,
                                        test_fc_cube.data.shape,
                                    ),
                                )
                                test_cubes.append(lead_time_cube)

                            print(
                                f"Now calibrating {model} for lead time {lead_time}, ref time {ref_time} and approach {approach}"
                            )
                            fc = apply_rainforests_calibration.process(
                                test_fc_cube,
                                *test_cubes,
                                model_config=model_config,
                                bin_data=True,
                                output_thresholds=THRESHOLDS,
                            )
                            calibs.append(fc)
                concat_ds = calibs.merge_cube()
                print(concat_ds)
                print(f"Saving to {output_dir}")
                save_netcdf(
                    concat_ds,
                    output_dir,
                )
