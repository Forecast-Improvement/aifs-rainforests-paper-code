"""Regrid ancillary data to be used in calculating solar data."""

import earthkit.regrid as ekr
import earthkit.data as ekd

path = f"/path/to/ancil.grib"
output = f"/path/to/aifs_data_ekr/ancil_regridded.grib"

# load in  AIFS ancil data
data = ekd.from_source("file", path)
# regrid to lat/lon grid
data = ekr.interpolate(data, out_grid={"grid": [0.25, 0.25]}, method="linear")
data.save(output)
