"""Aggregate obs to 24-hour periods."""

import pandas as pd
from pathlib import Path

obs_dir = Path("/path/to/obs")
df = pd.read_parquet(obs_dir / "precipitation_accumulation-PT01H.parquet")
df_arr = []
df.reset_index(inplace=True)
for site_id in df["spot_index"].unique():
    sub_df = df.loc[df["spot_index"] == site_id]
    sub_df.set_index("time", inplace=True)
    sub_df.sort_index(inplace=True)
    agg = (
        sub_df.rolling(pd.Timedelta(hours=24), closed="right", min_periods=24)[
            "lwe_thickness_of_precipitation_amount"
        ]
        .sum()
        .to_frame("lwe_thickness_of_precipitation_amount")
    )
    agg["spot_index"] = site_id
    agg = agg.reset_index()
    agg = agg.loc[agg["time"].dt.hour == 12]
    agg.set_index(["time", "spot_index"], drop=True, inplace=True, append=True)
    df_arr.append(agg)
df = pd.concat(df_arr, axis=0)
df = df.loc[df["lwe_thickness_of_precipitation_amount"].notnull()]
df.to_parquet(obs_dir / "precipitation_accumulation-PT24H.parquet")
