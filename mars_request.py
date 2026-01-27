"""Get AIFS data from MARS web API"""

from ecmwfapi import ECMWFService
import datetime as dt
import pandas as pd
import os
import pathlib


output_folder = "/path/to/aifs_data/"
if not (os.path.exists(output_folder)):
    pathlib.Path(output_folder).mkdir(parents=True)

start_date = dt.datetime(2025, 4, 1)
end_date = dt.datetime(2025, 11, 1)

for month_start in pd.date_range(start_date, end_date, freq="MS"):
    days_in_month = pd.date_range(month_start, month_start + pd.offsets.MonthEnd())
    start_datestr = month_start.strftime("%Y%m%d")
    end_datestr = (month_start + pd.offsets.MonthEnd()).strftime("%Y%m%d")
    month_str = month_start.strftime("%Y_%m")
    server = ECMWFService("mars")
    # wind
    output_path_month = f"{output_folder}/wind_{month_str}.grib"
    try:
        if not os.path.exists(output_path_month):
            server.execute(
                {
                    "class": "ai",
                    "date": f"{start_datestr}/to/{end_datestr}",
                    "expver": "1",
                    "levtype": "pl",
                    "levelist": "700",
                    "param": "131/132",
                    "stream": "oper",
                    "time": "12",
                    "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240",
                    "type": "fc",
                },
                output_path_month,
            )
    except Exception as e:
        print(e)
        if not os.path.exists(output_path_month):
            for d in days_in_month:
                try:
                    d_str = d.strftime("%Y%m%d")
                    output_path = f"{output_folder}/wind_{d_str}.grib"
                    if not os.path.exists(output_path):
                        print(f"Retrieving {output_path}")
                        server.execute(
                            {
                                "class": "ai",
                                "date": f"{d_str}",
                                "expver": "1",
                                "levtype": "pl",
                                "levelist": "700",
                                "param": "131/132",
                                "stream": "oper",
                                "time": "12",
                                "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240",
                                "type": "fc",
                            },
                            output_path,
                        )
                except Exception as e:
                    print(e)
    # precip (convective 143, total 228)
    output_path_month = f"{output_folder}/precip_{month_str}.grib"
    try:
        if not os.path.exists(output_path_month):
            server.execute(
                {
                    "class": "ai",
                    "date": f"{start_datestr}/to/{end_datestr}",
                    "expver": "1",
                    "levtype": "sfc",
                    "param": "143/228",
                    "stream": "oper",
                    "time": "12",
                    "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240",
                    "type": "fc",
                },
                output_path_month,
            )
    except Exception as e:
        if not os.path.exists(output_path_month):
            for d in days_in_month:
                try:
                    d_str = d.strftime("%Y%m%d")
                    output_path = f"{output_folder}/precip_{d_str}.grib"
                    if not os.path.exists(output_path):
                        server.execute(
                            {
                                "class": "ai",
                                "date": f"{d_str}",
                                "expver": "1",
                                "levtype": "sfc",
                                "param": "143/228",
                                "stream": "oper",
                                "time": "12",
                                "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240",
                                "type": "fc",
                            },
                            output_path,
                        )
                except Exception as e:
                    print(e)
