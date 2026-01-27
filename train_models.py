"""Training script for RainForests calibration"""

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
)

# dividing mm values by 1000 to convert to metres
RADAR_PRECISION = 0.05 / 1000
RECORD_RAINFALL = 0.96
# Parameters for adjusting observations for dew
# target proportions of observations <= 0.2 mm
TARGET_SMALL_OBS_PROPORTION_RADAR = 0.12
TARGET_SMALL_OBS_PROPORTION_GAUGE = 0.05
# solar irradiance threshold below which observations <= 0.2 mm are adjusted to account for dew
DEW_IRRADIANCE_THRESHOLD = 1.8e7


# make training, validation and testing split
train = ["2024_03", "2024_04", "2024_05", "2024_08", "2024_11", "2025_02", "2025_05"]
val = ["2024_06", "2024_09", "2024_12", "2025_03", "2025_06"]
test = ["2024_07", "2024_10", "2025_01", "2025_04", "2025_07"]

name_dict = {
    "tp": {
        "filename": "precipitation_accumulation",
        "long_name": "lwe_thickness_of_precipitation_amount",
    },
    "cp": {
        "filename": "precipitation_accumulation_from_convection",
        "long_name": "lwe_thickness_of_convective_precipitation_amount",
    },
    "wind_speed": {"filename": "wind_speed", "long_name": "wind_speed"},
    "solar": {
        "filename": "clearsky_solar_radiation",
        "long_name": "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
    },
    "cape": {"filename": "cape", "long_name": "convective_available_potential_energy"},
}


def remove_non_overlapping(data1, data2):
    """Remove data from data1 with forecast reference times that do not appear in data2"""
    print(len(data1))
    mask = data1["forecast_reference_time"].isin(data2["forecast_reference_time"])
    filtered = data1[mask]
    print(len(filtered))
    print(filtered.head())
    return filtered.reset_index()


def compare_props(model, train_data, x_vars, y_vars, t, lt, comparisons):
    """Helper function to monitor model performance with a very basic metric of comparing proportion of exceedance"""
    pred = model.predict(train_data[x_vars].values).mean()
    actual = (train_data[y_vars].values.flatten() >= t).mean()
    # print(f"Predicted prop: {pred} vs actual: {actual}")
    comparisons.append(
        f"For t >{t}m with lead time {lt} days, the predicted proportion is {pred:.5f} vs {actual:.5f} actual"
    )


def get_merged_set(month_list, obs, vars, pred_path):
    """Read and join dataframes on 'site_id' and 'time' to create dataset with all variables
    including observation values for precipitation

    Returns pandas dataframe
    """
    var_dfs = {}
    month_dfs = []
    for y_m in month_list:
        for var in vars:
            if var == "solar":
                path = f"{pred_path}/solar/solar_extracted.parquet"
            else:
                path = f"{pred_path}/{name_dict[var]["filename"]}_{y_m}.parquet"
            var_dfs[var] = pl.read_parquet(path)

        # get first set and iteratively join
        output_df = var_dfs[vars[0]]

        # iteratively join
        for var in vars[1:]:
            # print(var_dfs[var].dtypes)
            if var == "solar":
                output_df = output_df.join(
                    var_dfs[var],
                    left_on=["time", "site_id"],
                    right_on=["forecast_reference_time", "site_id"],
                    how="inner",
                )
            else:
                output_df = output_df.join(
                    var_dfs[var],
                    on=[
                        "time",
                        "forecast_reference_time",
                        "site_id",
                        "forecast_period",
                    ],
                    how="inner",
                )
        output_df = output_df.join(
            obs,
            left_on=["time", "site_id"],
            right_on=["time", "spot_index"],
            how="inner",
            suffix="_observed",
        )
        month_dfs.append(output_df)
    dataset = pl.concat(month_dfs, how="vertical")
    return dataset.to_pandas()


def identify_large_increment_gauges(data):
    """Remove observations from gauges where the minimum non-zero observation is
    greater than 0.2 mm.

    Args:
        data: training data
    Returns:
        Original dataframe with additional boolean column exclude_gauge
    """

    unique_gauge_obs = data[
        ["site_id", "lwe_thickness_of_precipitation_amount_observed"]
    ].drop_duplicates()
    eps = 1e-6  # compare equality within tolerance to account for floating-point inaccuracies
    min_nonzero = (
        unique_gauge_obs.loc[
            unique_gauge_obs["lwe_thickness_of_precipitation_amount_observed"] > eps
        ]
        .groupby("site_id")["lwe_thickness_of_precipitation_amount_observed"]
        .min()
    )
    exclude_gauges = min_nonzero.loc[
        min_nonzero > (0.2 + eps) / 1000.0
    ].index.values  # divide by 1000 to account for m unit
    data["exclude_gauge"] = False
    data.loc[data["site_id"].isin(exclude_gauges), "exclude_gauge"] = True

    return data


def exclude_dew(data_pd):
    """Modify nonzero observations <= 0.2 mm, setting some to zero, to counteract the effect of
    dew observations on model training. Modifies data in-place.

    Args:
        data:
            The training dataset.
        gauge_sites:
            Array listing the gauge observation sites.
    """

    # dataframe to hold columns created as intermediate steps of calculation
    calc_data = pd.DataFrame(index=data_pd.index)
    calc_data["less_than_0.2"] = (
        data_pd["lwe_thickness_of_precipitation_amount_observed"] > 0
    ) & (
        data_pd["lwe_thickness_of_precipitation_amount_observed"]
        <= (0.2) / 1000.0  # + RADAR_PRECISION maybe need to not remove this
    )
    bins = data_pd[
        "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time"
    ].quantile(np.arange(0, 1.02, 0.02))
    calc_data["irradiance_bin"] = pd.cut(
        data_pd[
            "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time"
        ],
        bins=bins,
        retbins=False,
        labels=0.5 * (bins.values[1:] + bins.values[:-1]),
    )
    calc_data["is_gauge"] = True
    calc_data["binned_prob"] = calc_data.groupby(["is_gauge", "irradiance_bin"])[
        "less_than_0.2"
    ].transform("mean")
    calc_data["target_prob"] = np.where(
        calc_data["is_gauge"].values,
        TARGET_SMALL_OBS_PROPORTION_GAUGE,
        TARGET_SMALL_OBS_PROPORTION_RADAR,
    )
    calc_data["set_to_zero_prob"] = (
        1 - calc_data["target_prob"] / calc_data["binned_prob"]
    )
    np.random.seed(0)
    calc_data["set_to_zero"] = (
        np.random.random(len(data_pd)) < calc_data["set_to_zero_prob"]
    )
    adjust_bool = calc_data["less_than_0.2"] & (
        data_pd[
            "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time"
        ]
        <= DEW_IRRADIANCE_THRESHOLD
    )
    data_pd.loc[
        adjust_bool & calc_data["set_to_zero"],
        "lwe_thickness_of_precipitation_amount_observed",
    ] = 0
    return data_pd


def create_forced_bins_file(sorted_train_columns, filepath):
    """Create a json file specifying the forced bins using the
    numeric feature indices of the precipitation features."""
    forced_bins = FORCED_BINS_IN_M
    precip_column_indices = []
    for i, column in enumerate(sorted_train_columns):
        if "precipitation" in column:
            precip_column_indices.append(i)
    forced_bins_config = [
        {"feature": i, "bin_upper_bound": forced_bins} for i in precip_column_indices
    ]
    with open(filepath, "w") as f:
        json.dump(forced_bins_config, f)
    return filepath


def quality_control(data):
    """Apply identify_large_increment_gauages and exclude_dew functions"""

    print(f"Data len before exclude gauge {len(data)}")
    data = identify_large_increment_gauges(data)
    # remove gauges with large intervals
    data = data[data["exclude_gauge"] == 0]
    print(f"Data len after exclude gauge {len(data)}")

    # exclude observations that are likely to be dew rather than precipitation
    print(
        f"Proportion 0s before dew func: {(data["lwe_thickness_of_precipitation_amount_observed"] == 0).mean()}"
    )
    data = exclude_dew(data)
    print(
        f"Proportion 0s after dew func:  {(data["lwe_thickness_of_precipitation_amount_observed"] == 0).mean()}"
    )

    # exclude observations that are greater than the record recorded
    print(f"Data len before exclude record amount {len(data)}")
    data = data[
        data["lwe_thickness_of_precipitation_amount_observed"] < RECORD_RAINFALL
    ]
    print(f"Data len after exclude record amount {len(data)}")
    return data


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

    # get nearest thresholds for params if using multi_param
    threshold_lookup = {
        float(x): float(TARGET_THRESHOLDS[np.argmin(np.abs(TARGET_THRESHOLDS - x))])
        for x in THRESHOLDS_POS
    }

    # training approach options "multi_param", "combined_param", "default"
    for approach in ["default"]:
        for nwp_model in ["HRES", "AIFS"]:
            if nwp_model == "AIFS":
                train_data = train_data_AIFS
                val_data = val_data_AIFS
            else:
                train_data = train_data_HRES
                val_data = val_data_HRES
            train_data_pd = quality_control(train_data)
            val_data_pd = quality_control(val_data)
            base_output_dir = f"/path/to/model_output/{approach}"
            for model_type in ["models", "compiled_models"]:
                curr_dir = f"{base_output_dir}/{nwp_model}/{model_type}"
                os.makedirs(curr_dir, exist_ok=True)
            y_vars = ["lwe_thickness_of_precipitation_amount_observed"]
            x_vars = [
                "lwe_thickness_of_precipitation_amount",
                "lwe_thickness_of_convective_precipitation_amount",
                "wind_speed",
                "integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
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
                # lead times of 1-10 days
                for lt in range(1, 11):
                    train_data_subset = train_data_pd[
                        train_data_pd["forecast_period"] == pd.Timedelta(days=lt)
                    ]
                    val_data_subset = val_data_pd[
                        val_data_pd["forecast_period"] == pd.Timedelta(days=lt)
                    ]
                    train_dataset = lgb.Dataset(
                        train_data_subset[x_vars].values,
                        (train_data_subset[y_vars].values.flatten() >= t).astype(int),
                    )
                    val_dataset = lgb.Dataset(
                        val_data_subset[x_vars].values,
                        (val_data_subset[y_vars].values.flatten() >= t).astype(int),
                    )
                    if approach == "multi_param":
                        nearest_threshold = threshold_lookup[threshold]
                        if nwp_model == "AIFS":
                            params = GBDT_PARAMS_AIFS[nearest_threshold]
                        else:
                            params = GBDT_PARAMS_HRES[nearest_threshold]
                        print(f"Chosen params, {nearest_threshold}")
                    elif test == "combined_param":
                        params = GBDT_PARAMS_ALL[nwp_model]["all"]
                        print(f"Chosen params, combined {nwp_model}")
                    else:
                        params = GBDT_PARAMETERS["default"]
                        print(f"Chosen params, default")
                    params["forcedbins_filename"] = str(forced_bin_file)
                    gbdt = lgb.train(
                        params,
                        train_dataset,
                    )
                    print("training_set_size:", train_dataset.num_data())
                    # generate simple performance metric
                    compare_props(
                        gbdt, train_data_subset, x_vars, y_vars, t, lt, comparisons
                    )
                    filename = f"model_{t:0.06f}_{lt:02d}.txt"
                    model_dir = f"{base_output_dir}/{nwp_model}/models"
                    gbdt_path = f"{model_dir}/{filename}"
                    gbdt.save_model(str(gbdt_path))
                    print("Training threshold", t, "at lead time", lt, "days")
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
                    gbdt_path = f"{model_dir}/model_{t:0.06f}_{lead_time:02d}.txt"
                    treelite_path = f"{compiled_dir}/model_{t:0.06f}_{lead_time:02d}.so"

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
