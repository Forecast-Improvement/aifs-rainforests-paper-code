import pandas as pd
import numpy as np
import lightgbm as lgb
import treelite
import tl2cgen
import os
import json

thresholds = np.array(
    [
        10,
        1,
    ]
) * 0.001

MODEL_TO_CF_VARIABLES = {
    "precipitation_accumulation_from_convection_mm": "lwe_thickness_of_convective_precipitation_amount",  # noqa: E501
    "precipitation_accumulation_mm_fcst": "lwe_thickness_of_precipitation_amount",
    "wind_speed_m_s-1": "wind_speed",
    "convective_cloud_depth_m": "convective_cloud_depth",
    "cape_J_kg-1": "cape",
    "ensemble_mean_precipitation_accumulation_mm": "ensemble_mean_lwe_thickness_of_precipitation_amount",  # noqa: E501
    "ensemble_std_precipitation_accumulation_mm": "ensemble_std_lwe_thickness_of_precipitation_amount",  # noqa: E501
}


data = pd.read_parquet("/path/to/ecmwf_HRES-training-2023-04-PT24H.parquet")
data = data.loc[data["lead_time_hours"] == 24]
data = data.rename(columns=MODEL_TO_CF_VARIABLES)

for nwp_model in ["HRES", "AIFS"]:
    base_output_dir = "/path/to/test_models/"
    for model_type in ["models", "compiled_models"]:
        curr_dir = f"{base_output_dir}/{nwp_model}/{model_type}"
        os.makedirs(curr_dir, exist_ok=True)
    y_vars = ["lwe_thickness_of_precipitation_amount"]
    if nwp_model == "AIFS":    
        x_vars = ["lwe_thickness_of_precipitation_amount", "wind_speed", "lwe_thickness_of_convective_precipitation_amount"]
    else:
        x_vars = ["lwe_thickness_of_precipitation_amount", "wind_speed", "lwe_thickness_of_convective_precipitation_amount", "cape"]

    for t in thresholds:
        train_data = lgb.Dataset(data[x_vars].values, (data[y_vars].values.flatten() >= t).astype(int) )

        params ={
            "num_tree": 10,
                "learning_rate": 0.025,
                "num_leaves": 5,
                "subsample": 0.8,
                "bagging_freq": 1,
                "objective": "regression",
                "verbose": -1,
                "num_threads": 48,
                "bin_construct_sample_cnt": int(1e9),
                "seed": 0}
        
        gbdt = lgb.train(params, train_data)
        filename = f"test_model_{t:0.06f}.txt"
        model_dir = f"{base_output_dir}/{nwp_model}/models/"
        gbdt_path = f"{model_dir}/{filename}"
        gbdt.save_model(str(gbdt_path))

    # compile
    compiled_dir = f"{base_output_dir}/{nwp_model}/compiled_models"
    for model_file in os.listdir(model_dir):
        model_path = f"{model_dir}/{model_file}"
        model = treelite.frontend.load_lightgbm_model(str(model_path))
        tl2cgen.export_lib(model,
            toolchain="gcc",
            libpath=str(model_path),
            verbose=False,
            params={"parallel_comp": 48, "quantize": 1},
        )

    # write config
    model_config = {}
    for lead_time in [24]:
        model_config[lead_time] = {}
        for t in thresholds:

            gbdt_path = f"{model_dir}/test_model_{t:0.06f}.txt"
            treelite_path = f"{compiled_dir}/test_model_{t:0.06f}.so"

            model_config[lead_time][f"{t:0.06f}"] = {
                "lightgbm_model": str(gbdt_path),
                "treelite_model": str(treelite_path),
            }

    output_filename = f"{base_output_dir}/{nwp_model}/model_config.json"

    with open(output_filename, "w") as model_config_json:
        json.dump(model_config, model_config_json, indent=4)
