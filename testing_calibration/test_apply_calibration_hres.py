from improver.cli import apply_rainforests_calibration
from improver.utilities.cube_manipulation import compare_coords
import iris
import json
from iris.coords import DimCoord
from improver.utilities.save import save_netcdf


input_dir = "/path/to/processed_data/HRES/"
fc_cube = iris.load_cube(
    f"{input_dir}/20240302T1200Z-PT0024H00M-precipitation_accumulation-PT24H.nc"
)
var_cubes = [
    iris.load_cube(f)
    for f in [
        f"{input_dir}/20240303T1200Z-PT0024H00M-precipitation_accumulation-PT24H.nc",
        f"{input_dir}/20240303T1200Z-PT0024H00M-precipitation_accumulation_from_convection-PT24H.nc",
        f"{input_dir}/20240303T1200Z-PT0024H00M-cape.nc",
        f"{input_dir}/20240303T1200Z-PT0024H00M-wind_speed_on_pressure_levels.nc",
        f"{input_dir}/20240303T1200Z-PT0000H00M-clearsky_solar_radiation-PT24H.nc",
    ]
]

with open("/path/to/model_output/HRES/model_config.json") as f:
    model_config = json.load(f)


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


output_thresholds = [0, 0.001, 0.002, 0.1]

fc = apply_rainforests_calibration.process(
    fc_cube,
    *var_cubes,
    model_config=model_config,
    bin_data=True,
    output_thresholds=output_thresholds,
)

save_netcdf(fc, f"/path/to/model_output/test_calibration.nc")
