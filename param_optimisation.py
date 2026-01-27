"""Uses Optuna to train on a subset of data. Aims to optimise parameters for a given metric. """

from train_models import (
    quality_control,
    get_merged_set,
    remove_non_overlapping,
    train,
    val,
    name_dict,
    create_forced_bins_file,
)
import polars as pl
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import optuna
import gc
from sklearn.metrics import mean_squared_error
from crps import crps_threshold
from functools import partial
from sklearn.metrics import brier_score_loss

# use a representatitive subset of thresholds
# defined in m for consistency
thresholds = [0.1, 0.2, 0.4, 1, 10, 50, 100]
lead_times = [1, 5, 10]

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

train_data = {}
train_data["AIFS"] = train_data_AIFS
train_data["HRES"] = train_data_HRES

val_data = {}
val_data["AIFS"] = val_data_AIFS
val_data["HRES"] = val_data_HRES

y_vars = ["lwe_thickness_of_precipitation_amount_observed"]
x_vars = {}
x_vars["AIFS"] = [
    "lwe_thickness_of_precipitation_amount",
    "lwe_thickness_of_convective_precipitation_amount",
    "wind_speed",
    "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
]
x_vars["HRES"] = x_vars["AIFS"] + ["convective_available_potential_energy"]


def objective_seperate(trial, xvars, train, val, threshold):
    """
    Objective function for finding optimal parameters for individual thresholds
    """
    param = {
        "num_tree": trial.suggest_int("num_tree", 100, 500),
        "learning_rate": 0.025,
        "num_leaves": trial.suggest_int("num_leaves", 2, 40),
        "subsample": 0.8,
        "bagging_freq": 1,
        "objective": "regression",
        "verbose": -1,
        "num_threads": 48,
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100),
        "bin_construct_sample_cnt": int(1e9),
        "seed": 0,
    }
    param["forcedbins_filename"] = str(forced_bin_file)
    scores = []
    for lt in lead_times:
        # subset data to the specific lead time
        subset = train_data_pd[
            train_data_pd["forecast_period"] == pd.Timedelta(days=lt)
        ]
        val_subset = val_data_pd[
            val_data_pd["forecast_period"] == pd.Timedelta(days=lt)
        ]
        y = subset[y_vars].values.flatten()
        y_val = val_subset[y_vars].values.flatten()
        t = threshold / 1000.0  # convert to m
        train_dataset = lgb.Dataset(
            subset[xvars].values,
            (y >= t).astype(int),
        )
        gbdt = lgb.train(
            param,
            train_dataset,
        )
        prob_preds = gbdt.predict(val_subset[xvars])
        # clean up memory
        del train_dataset, gbdt
        gc.collect()
        # enforce values between 0 and 1 (probabilities)
        prob_preds = np.clip(prob_preds, 0.0, 1.0)
        scores.append(brier_score_loss((y_val >= t).astype(int), prob_preds))
    # return mean brier score across all lead times
    return float(np.mean(scores))


def objective_combined(trial, xvars, train, val):
    """
    Objective function for finding one set of parameters that works best at all rainfall thresholds
    """
    param = {
        "num_tree": trial.suggest_int("num_tree", 100, 500),
        "learning_rate": 0.025,
        "num_leaves": trial.suggest_int("num_leaves", 2, 40),
        "subsample": 0.8,
        "bagging_freq": 1,
        "objective": "regression",
        "verbose": -1,
        "num_threads": 48,
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100),
        "bin_construct_sample_cnt": int(1e9),
        "seed": 0,
    }
    param["forcedbins_filename"] = str(forced_bin_file)
    scores = []
    for lt in lead_times:
        subset = train_data_pd[
            train_data_pd["forecast_period"] == pd.Timedelta(days=lt)
        ]
        val_subset = val_data_pd[
            val_data_pd["forecast_period"] == pd.Timedelta(days=lt)
        ]
        y = subset[y_vars].values.flatten()
        y_val = val_subset[y_vars].values.flatten()
        prob_preds = []
        thresholds_m = []
        for threshold in thresholds:
            t = threshold / 1000.0  # convert to m
            thresholds_m.append(t)
            train_dataset = lgb.Dataset(
                subset[xvars].values,
                (y >= t).astype(int),
            )
            gbdt = lgb.train(
                param,
                train_dataset,
            )
            preds = gbdt.predict(val_subset[xvars])
            prob_preds.append(preds)
            # clear from memory
            del train_dataset, gbdt
            gc.collect()
        # organise for crps input
        prob_preds = np.stack(prob_preds, axis=-1)
        thresholds_m = np.array(thresholds_m)
        cdf_preds = 1.0 - prob_preds
        # if getting negative values, enforce monotone increasing
        cdf_preds = np.maximum.accumulate(cdf_preds, axis=-1)
        crps = crps_threshold(
            observations=y_val,
            forecasts=cdf_preds,
            thresholds=thresholds_m,
        )
        scores.append(crps.mean())
    # return mean CRPS across all thresholds and lead times
    return float(np.mean(scores))


# choose number of trials to run
# 1 trial takes about 10 minutes to get through the whole loop
n_trials_setting = 100
storage_loc = "sqlite:///db.sqlite3" # Specify the storage URL here.

for model in ["AIFS", "HRES"]:
    train_data_pd = quality_control(train_data[model])
    val_data_pd = quality_control(val_data[model])
    # need to sort columns because they are just referred to by index in model
    selected_x_vars = np.sort(x_vars[model])
    forced_bin_file = create_forced_bins_file(
        selected_x_vars,
        f"/path/to/model_output/{model}_forced_bins.json",
    )
    # create seperate threshold parameters
    for t in thresholds:
        wrapped_objective_seperate = partial(
            objective_seperate,
            xvars=selected_x_vars,
            train=train_data_pd,
            val=val_data_pd,
            threshold=t,
        )
        study = optuna.create_study(
            direction="minimize",
            storage=storage_loc, 
            study_name=f"seperate_{model}_{t:.4f}",
            load_if_exists=True,
        )
        study.optimize(wrapped_objective_seperate, n_trials=n_trials_setting)
        print(f"Best value: {study.best_value} (params: {study.best_params})")
    # find best overall parameters for thresholds
    study = optuna.create_study(
        direction="minimize",
        storage=storage_loc,  # Specify the storage URL here.
        study_name=f"combined_{model}",
        load_if_exists=True,
    )
    # wrapped allows for additional arguments
    wrapped_objective_combined = partial(
        objective_combined, xvars=selected_x_vars, train=train_data_pd, val=val_data_pd
    )
    study.optimize(wrapped_objective_combined, n_trials=n_trials_setting)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
