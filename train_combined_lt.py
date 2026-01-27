"""Training script for RainForests calibration with combined lead times"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import treelite
import tl2cgen
import os
import json
import polars as pl
from model_config import (
    FORCED_BINS_IN_M,
    GBDT_PARAMETERS,
    THRESHOLDS_POS,
    TARGET_THRESHOLDS,
    GBDT_PARAMS_AIFS,
    GBDT_PARAMS_HRES,
    GBDT_PARAMS_ALL,
    GBDT_UPDATED_PARAMETERS,
)
from train_models import (
    create_forced_bins_file,
    get_merged_set,
    quality_control,
    remove_non_overlapping,
    train,
    val,
)
# choose training approach (e.g. gridded or at extracted sites)
APPROACH = "combined_lt_new_param"

if __name__ == "__main__":
    obs_path = "/path/to/obs/precipitation_accumulation-PT24H.parquet"
    obs = pl.read_parquet(obs_path)
    # cast so it matches other inputs
    obs = obs.with_columns(pl.col("time").cast(pl.Datetime("ns")))
    # var_names = ["wind_speed", "solar", "tp", "cp"]
    pred_path = "/path/to/extracted_sites/"
    var_names_AIFS = ["wind_speed", "solar", "tp", "cp"]
    var_names_HRES = var_names_AIFS + ["cape"]

    # get merged set and sort columns
    train_data_HRES = get_merged_set(train, obs, var_names_HRES, pred_path + "HRES")
    val_data_HRES = get_merged_set(val, obs, var_names_HRES, pred_path + "HRES")
    train_data_AIFS = get_merged_set(train, obs, var_names_AIFS, pred_path + "AIFS")
    val_data_AIFS = get_merged_set(val, obs, var_names_AIFS, pred_path + "AIFS")

    # remove non-overlapping
    train_data_HRES = remove_non_overlapping(train_data_HRES, train_data_AIFS)
    train_data_AIFS = remove_non_overlapping(train_data_AIFS, train_data_HRES)
    val_data_HRES = remove_non_overlapping(val_data_HRES, val_data_AIFS)
    val_data_AIFS = remove_non_overlapping(val_data_AIFS, val_data_HRES)

    # list for tracking model performance
    comparisons = []

    # convert lead time to int, potentially will work nicer for training
    train_data_HRES["n_lead_days"] = train_data_HRES["forecast_period"].dt.days
    train_data_AIFS["n_lead_days"] = train_data_AIFS["forecast_period"].dt.days
    val_data_HRES["n_lead_days"] = val_data_HRES["forecast_period"].dt.days
    val_data_AIFS["n_lead_days"] = val_data_AIFS["forecast_period"].dt.days

    print(train_data_AIFS["n_lead_days"])
    print(train_data_HRES["n_lead_days"])



    for nwp_model in ["HRES", "AIFS"]:
        if nwp_model == "AIFS":
            train_data = train_data_AIFS
            val_data = val_data_AIFS
        else:
            train_data = train_data_HRES
            val_data = val_data_HRES

        train_data_pd = quality_control(train_data)
        val_data_pd = quality_control(val_data)

        base_output_dir = f"/path/to/model_output/{APPROACH}"
        for model_type in ["models", "compiled_models"]:
            curr_dir = f"{base_output_dir}/{nwp_model}/{model_type}"
            os.makedirs(curr_dir, exist_ok=True)
        y_vars = ["lwe_thickness_of_precipitation_amount_observed"]
        x_vars = [
            "lwe_thickness_of_precipitation_amount",
            "lwe_thickness_of_convective_precipitation_amount",
            "wind_speed",
            "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
            "n_lead_days",
        ]
        if nwp_model == "HRES":
            x_vars += ["convective_available_potential_energy"]

        # need to sort columns because they are just referred to by index in model
        x_vars = np.sort(x_vars)
        print("sorted columns:", x_vars)
        forced_bin_file = create_forced_bins_file(
            x_vars,
            f"/path/to/model_output/{nwp_model}_forced_bins.json",
        )
        # train
        for threshold in THRESHOLDS_POS:
            t = threshold / 1000.0  # convert to m
            train_dataset = lgb.Dataset(
                train_data_pd[x_vars].values,
                (train_data_pd[y_vars].values.flatten() >= t).astype(int),
            )
            val_dataset = lgb.Dataset(
                val_data_pd[x_vars].values,
                (val_data_pd[y_vars].values.flatten() >= t).astype(int),
            )
            if APPROACH == "multi_param":
                nearest_threshold = threshold_lookup[threshold]
                if nwp_model == "AIFS":
                    params = GBDT_PARAMS_AIFS[nearest_threshold]
                else:
                    params = GBDT_PARAMS_HRES[nearest_threshold]
                print(f"Chosen params, {nearest_threshold}")
            elif APPROACH == "combined_param":
                params = GBDT_PARAMS_ALL[nwp_model]["all"]
                print(f"Chosen params, combined {nwp_model}")
            elif APPROACH == "combined_lt_new_param":
                print("using new params for combined_lt")
                params = GBDT_UPDATED_PARAMETERS[nwp_model]
            else:
                params = GBDT_PARAMETERS["default"]
                print("Chosen params, default")
            params["forcedbins_filename"] = str(forced_bin_file)
            gbdt = lgb.train(
                params,
                train_dataset,
            )
            print("Size of training dataset:", train_dataset.num_data())
            filename = f"model_{t:0.06f}.txt"
            model_dir = f"{base_output_dir}/{nwp_model}/models"
            gbdt_path = f"{model_dir}/{filename}"
            gbdt.save_model(str(gbdt_path))
            print("Training threshold", t, "at all lead times")
        # compile
        compiled_dir = f"{base_output_dir}/{nwp_model}/compiled_models"
        for model_file in os.listdir(model_dir):
            model_path = f"{model_dir}/{model_file}"
            model = treelite.frontend.load_lightgbm_model(str(model_path))
            compiled_file = model_file.replace(".txt", ".so")
            print("saving to", compiled_dir)
            tl2cgen.export_lib(
                model,
                toolchain="gcc",
                libpath=f"{compiled_dir}/{compiled_file}",
                verbose=False,
                params={"parallel_comp": 48, "quantize": 1},
            )
        # write config
        model_config = {}
        for lead_time in range(1, 11):
            model_config[lead_time] = {}
            for threshold in THRESHOLDS_POS:
                t = threshold / 1000.0
                gbdt_path = f"{model_dir}/model_{t:0.06f}.txt"
                treelite_path = f"{compiled_dir}/model_{t:0.06f}.so"

                model_config[lead_time][f"{t:0.06f}"] = {
                    "lightgbm_model": str(gbdt_path),
                    "treelite_model": str(treelite_path),
                }
        output_filename = f"{base_output_dir}/{nwp_model}/model_config.json"
        with open(output_filename, "w") as model_config_json:
            json.dump(model_config, model_config_json, indent=4)
    # print off model performance metrics
    for comparison in comparisons:
        print(comparison)
