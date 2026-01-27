from improver.cli import apply_rainforests_calibration
from improver.utilities.cube_manipulation import compare_coords
import iris
import json
import pandas as pd
import datetime as dt
from improver.utilities.save import save_netcdf

# month data site data
input_dir = "/path/to/extracted_sites/AIFS"


fc_cube = iris.load_cube(f"{input_dir}/precipitation_accumulation_2025_07.nc")

var_cubes = [
    iris.load_cube(f)
    for f in [
        f"{input_dir}/wind_speed_2025_07.nc",
        f"{input_dir}/precipitation_accumulation_from_convection_2025_07.nc",
        f"{input_dir}/precipitation_accumulation_2025_07.nc",
        f"{input_dir}/solar/solar_extracted.nc",
    ]
]


with open("/path/to/model_output/AIFS/model_config.json") as f:
    model_config = json.load(f)

output_thresholds = [0, 0.001, 0.002, 0.1]


def get_cube_subset(cube, ref_time, lead_time, var_type):
    print(ref_time)
    print(lead_time)
    print(var_type, cube.units)
    if var_type in ["tp", "cp"]:
        # it appears m has been dropped in the extract sites step
        cube.units = "m"
    if var_type == "solar":
        valid_time = ref_time + pd.Timedelta(days=lead_time)
        constr_val = iris.Constraint(
            forecast_reference_time=lambda cell: cell.point == valid_time
        )
        return cube.extract(constr_val)
    else:
        lead_time_s = lead_time * 86400
        constr_ref = iris.Constraint(
            forecast_reference_time=lambda cell: cell.point == ref_time
        )
        constr_lt = iris.Constraint(
            forecast_period=lambda cell: cell.point == lead_time_s
        )
        return cube.extract(constr_ref & constr_lt)


start_date = dt.datetime(2025, 7, 1)
end_date = dt.datetime(2025, 7, 31)

date_range = pd.date_range(start=start_date, end=end_date, freq="D")

offset = dt.timedelta(hours=12)

var_types = ["wind", "cp", "tp", "solar"]
calibs = iris.cube.CubeList()
for ref_time in date_range[0:5]:
    for lead_time in range(1, 11):
        test_cubes = []
        for i, var in enumerate(var_types):
            test_cubes.append(
                get_cube_subset(var_cubes[i], ref_time + offset, lead_time, var)
            )
        test_fc_cube = get_cube_subset(fc_cube, ref_time + offset, lead_time, "tp")

        fc = apply_rainforests_calibration.process(
            test_fc_cube,
            *test_cubes,
            model_config=model_config,
            bin_data=True,
            output_thresholds=output_thresholds,
        )

        calibs.append(fc)

concat_ds = calibs.merge_cube()
print(concat_ds)
save_netcdf(concat_ds, "/path/to/model_output/test_concat.nc")
