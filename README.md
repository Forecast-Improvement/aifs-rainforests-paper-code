# aifs-rainforests

## Introduction

Integrating deterministic AIFS with RainForests calibration for precipitation forecasting. Utilising forecasts in the period between 2024-03 and 2025-07. 

## Methods

### regrid AIFS data using earthkit
`ekr_regridding.py` and `ancil_regrid.py`: change grid system from AIFS format to latitude/longitude coordinates for AIFS environmental data and ancillary data (used downstream to calculate solar irradiance). Use environment `ekr.yml`.

### aggregate HRES and AIFS data
Aggregate output of IMPROVER-ingest suite to produce daily aggregations of input variables for Rainforests ECMWF-HRES model. For HRES use `aggregate_hres_data.py` with environment `processing.yml`. For AIFS use `aggregate_AIFS_data.py` with `processing.yml`.

### generate solar irradiance data
Generate clearky solar irradiance data since this is a variable in the Rainforests model. Use `generate_solar_data.py` with environment `processing.yml`.

### extract data at site locations
To create training data, extract all weather variables at site locations and aggregate by month (with relation to forecast reference time). For all variables aggregated monthly use `extract_sites.py` and for solar data across the study time-period use `extract_solar.py`, for both scripts use the environment `processing.yml`. 

### optimise parameters
To find the ideal parameters for training run `param_optimisation.py`, this makes use of optuna package to systematically identify the best parameters for training. Use environment `processing.yml`, optionally run `pip install optuna-dashboard` to visualise process.

### train models 
Split the dataset by month into training, validation and test sets. For each month join all weather variables and observation data into a single dataframe which is then used to train a gradient-boosted decision-tree model at each forecast period and rainfall threshold. Use `train_models.py` with `processing.yml`. Alternatively, use `train_combined_lt.py` to train using a single model for all lead times.

### apply calibration
To apply the RainForests calibration on month-aggregated data use `apply_calibration.py` with `processing.yml`.