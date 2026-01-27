"""Make site cube for extracting forecasts at sites."""

import pandas as pd
import xarray as xr
import numpy as np

site_data = pd.read_parquet("/path/to/obs/precipitation_accumulation-PT24H.parquet")
metadata = pd.read_parquet("/path/to/obs/station_metadata.parquet")
metadata = metadata.loc[
    metadata["STN_NUM"].isin(np.unique(site_data.index.get_level_values("spot_index")))
]

lat = xr.DataArray(
    metadata["LAT"].values,
    name="latitude",
    coords={"site_id": metadata["STN_NUM"].values},
)
lon = xr.DataArray(
    metadata["LON"].values,
    name="longitude",
    coords={"site_id": metadata["STN_NUM"].values},
)
altitude = xr.DataArray(
    metadata["STN_HT"].values,
    name="altitude",
    coords={"site_id": metadata["STN_NUM"].values},
)
site_cube = xr.Dataset({"latitude": lat, "longitude": lon, "altitude": altitude})
site_cube.to_netcdf("/path/to/site_cube.nc")
