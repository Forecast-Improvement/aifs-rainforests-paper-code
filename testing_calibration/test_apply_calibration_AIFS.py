from improver.cli import apply_rainforests_calibration
from improver.utilities.cube_manipulation import compare_coords
import iris
import json


# single day data
input_dir = "/path/to/processed_data/AIFS/"
fc_cube = iris.load_cube(
    f"{input_dir}/20240302T1200Z-PT0024H00M-precipitation_accumulation.nc"
)


var_cubes = [
    iris.load_cube(f)
    for f in [
        f"{input_dir}/20240302T1200Z-PT0024H00M-precipitation_accumulation.nc",
        f"{input_dir}/20240302T1200Z-PT0024H00M-precipitation_accumulation_from_convection.nc",
        f"{input_dir}/20240302T1200Z-PT0024H00M-wind_speed.nc",
        f"{input_dir}/20240303T1200Z-PT0000H00M-clearsky_solar_radiation-PT24H.nc",
    ]
]

print(var_cubes[3].summary())
print(var_cubes[2].summary())

with open("/path/to/model_output/AIFS/model_config.json") as f:
    model_config = json.load(f)

output_thresholds = [0, 0.001, 0.002, 0.1]

fc = apply_rainforests_calibration.process(
    fc_cube,
    *var_cubes,
    model_config=model_config,
    bin_data=True,
    output_thresholds=output_thresholds,
)

print(fc)
