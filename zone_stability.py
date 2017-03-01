#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import (
    split_pollutant_dataset, preprocess_dataset, split_train_dev,
    get_Y, zone_station_train, zone_station_dev
)
from features import (
    make_features, make_seqential_features, get_seq_Y
)
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
from utils import evaluate_mse, my_split_fold, random_split_fold
import multiprocessing as mp
import xgboost as xgb


# Load data
X_train_path = "/Users/thomasopsomer/data/plume-data/X_train.csv"
Y_train_path = "/Users/thomasopsomer/data/plume-data/Y_train.csv"
df = pd.read_csv(X_train_path, index_col="ID")
df = preprocess_dataset(df)
Y = pd.read_csv(Y_train_path, index_col="ID")
# split for each pollutant
NO2_df, PM10_df, PM25_df = split_pollutant_dataset(df)


#
roll_mean_conf = {
    2: ["windspeed", "cloudcover"],
    4: ["windspeed", "cloudcover"],
    5: ["precipintensity", "precipprobability"],
    6: ["temperature"],
    10: ["precipintensity"],
    12: ["pressure", "cloudcover"],
    18: ["windbearingcos", "windbearingsin", "temperature"],
    24: ["pressure", "precipprobability", "windbearingcos"],
    32: ["windbearingsin"],
    48: ["pressure", "windbearingcos", "windbearingsin"],
    15: ["windspeed"],
    96: ["windbearingcos", "temperature", "windspeed"],
    144: ["temperature", "pressure"],
    288: ["temperature", "cloudcover"],
}

shift_config = {
    "temperature": [2],
    "cloudcover": [2],
    "pressure": [2],
    "windbearingsin": [6],
    "windbearingcos": [6],
    "windspeed": [2]
}

features_config = {
    "rolling_mean": False,
    #"deltas_mean": [6, 48, 96, 120, 192],
    #"roll_mean_conf": roll_mean_conf,
    "shift_config": shift_config,
    "log": True,
    "pollutant": False,
    # "temp_dec_freq": 2,
    #"scale_dec": True,
    "scale_temp": True,
    "scale_time": True,
    "remove_temporal": False,
    "rolling_std": False,
    # "deltas_std": [24, 48, 96, 120]
}


def train_eval(args):
    """ """  
    x_train, x_dev, y_train, y_dev, info = args
    res = {}
    res["info"] = info
    # fit ridge reg
    lr = linear_model.Ridge(alpha=1, normalize=True)
    lr.fit(x_train, y_train)
    mse_tr, mse_dev = evaluate_mse(lr, x_train, x_dev, y_train, y_dev)
    res["lr"] = (mse_tr, mse_dev)
    # # fit xgb
    xgb_model = xgb.XGBRegressor(max_depth=7, n_estimators=100, reg_lambda=5)
    xgb_model.fit(x_train, y_train)
    mse_tr, mse_dev = evaluate_mse(xgb_model, x_train, x_dev, y_train, y_dev)
    res["xgb"] = (mse_tr, mse_dev)
    # # fit rfr regressor
    rfr = ensemble.RandomForestRegressor(n_estimators=10, max_depth=15, n_jobs=1)
    rfr.fit(x_train, y_train)
    mse_tr, mse_dev = evaluate_mse(rfr, x_train, x_dev, y_train, y_dev)
    res["rfr"] = (mse_tr, mse_dev)
    # fit sgd regressor
    sgd = linear_model.SGDRegressor(loss="huber", penalty="l2", alpha=0.001, n_iter=30)
    sgd.fit(x_train, y_train)
    mse_tr, mse_dev = evaluate_mse(sgd, x_train, x_dev, y_train, y_dev)
    res["sgd"] = (mse_tr, mse_dev)
    return res


shift_config = {
    "temperature": [2],
    "cloudcover": [2],
    "pressure": [2],
    "windbearingsin": [6],
    "windbearingcos": [6],
    "windspeed": [2]
}

features_config = [
    {
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": False,
        "rolling_mean": False,
        "rolling_std": True
    },
    {
        "temp_dec_freq": 2,
        "scale_dec": True,
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": True,
        "rolling_mean": False,
        "rolling_std": True,
    },
    {
        "temp_dec_freq": 2,
        "scale_dec": True,
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": True,
        "rolling_mean": True, "deltas_mean": [6, 48, 96, 120, 192],
        "rolling_std": True
    },
    {
        "temp_dec_freq": 2,
        "scale_dec": True,
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": True,
        "rolling_mean": False, "deltas_mean": [6, 48, 96, 120, 192],
        "rolling_std": True, "deltas_std": [24, 48, 96, 120]
    },
    {
        "temp_dec_freq": 2,
        "scale_dec": True,
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": True,
        "rolling_mean": False, "deltas_mean": [6, 48, 96, 120, 192],
        "rolling_std": True, "deltas_std": [24, 48, 96, 120], "std_pca": True, "std_pca_n": 0.9,
    },
    {
        "temp_dec_freq": 2,
        "scale_dec": True,
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": True,
        "rolling_mean": False, "deltas_mean": [6, 48, 96, 120, 192],
        "rolling_std": False, "deltas_std": [24, 48, 96, 120],
        "cwt": True, "cwt_range": np.arange(1, 11, 2), "filter": "morl",
        "cwt_pca": True, "cwt_pca_n": 0.90
    },
    {
        "temp_dec_freq": 2,
        "scale_dec": True,
        "scale_temp": True,
        "scale_time": True,
        "remove_temporal": True,
        "rolling_mean": True, "deltas_mean": [6, 48, 96, 120, 192],
        "rolling_std": True, "deltas_std": [24, 48, 96, 120], "std_pca": True, "std_pca_n": 0.9,
        "cwt": True, "cwt_range": np.arange(1, 11, 2), "filter": "morl",
        "cwt_pca": True, "cwt_pca_n": 0.90,
        "shift_config": shift_config
    },
]


def gen():
    for train, dev, train_info, dev_info in random_split_fold(NO2_df, 10):
        for i, config in enumerate(features_config):
            print i
            x_train, x_dev = make_features(train, dev, **config)
            y_train = get_Y(Y, f_train)
            y_dev = get_Y(Y, f_dev)
            yield x_train, x_dev, y_train, y_dev, (train_info, dev_info, i)



# "penalty": ["l2", "l1"],
#     "alpha": [0.0001, 0.001, 1, 10],

np.mean([x["lr"][1] for x in results])
np.mean([x["xgb"][1] for x in results])
np.mean([x["sgd"][1] for x in results])


# g = my_split_fold(NO2_df, Y, 10, features_config, info=True)
g = gen()
pool = mp.Pool(4)
results = pool.map(train_eval, g)

np.mean([x["lr"][1] for x in results2])
np.mean([x["xgb"][1] for x in results2])
np.mean([x["rfr"][1] for x in results2])

np.std([x["lr"][1] for x in results2])
np.std([x["xgb"][1] for x in results2])
np.std([x["rfr"][1] for x in results2])

np.mean([x["lr"][1] for x in results5])
np.mean([x["sgd"][1] for x in results5])
np.mean([x["rfr"][1] for x in results5])
np.mean([x["xgb"][1] for x in results5])

np.std([x["lr"][1] for x in results5])
np.std([x["sgd"][1] for x in results5])
np.std([x["rfr"][1] for x in results5])
np.std([x["xgb"][1] for x in results5])

np.mean([x["lr"][1] for x in results6])
np.mean([x["sgd"][1] for x in results6])
np.mean([x["rfr"][1] for x in results6])
np.mean([x["xgb"][1] for x in results6])

