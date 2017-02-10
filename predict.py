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
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error


def train(dataset, labels):
    """ """
    pollutants = ["NO2", "PM10", "PM25"]
    # split dataset
    NO2_df, PM10_df, PM25_df = split_pollutant_dataset(dataset)
    # build data dict
    ds = dict(((poll, df) for poll, df in zip(pollutants, split_pollutant_dataset(dataset))))
    # build features dict
    f = {}
    for poll in pollutants:
        f[poll] = {}
        f[poll]["X"] = make_features(ds[poll], rolling_mean=True,
                                     deltas=[24, 36, 48, 96])
        f[poll]["Y"] = get_Y(labels, ds[poll])
    # train model for each pollutant
    model_dict = {}
    for poll in pollutants:
        xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=200)
        # train model
        xgb_model.fit(f[poll]["X"], f[poll]["Y"])
        # mse on training set
        y_pred = xgb_model.predict(f[poll]["X"])
        mse = mean_squared_error(f[poll]["Y"], y_pred)
        print("%s: MSE on training set: %.3f" % (poll, mse))
        # store model
        model_dict[poll] = xgb_model
    # return model dict
    return model_dict


def predict(model_dict, dataset):
    """ """
    # split dataset
    NO2_df, PM10_df, PM25_df = split_pollutant_dataset(dataset)
    # build features
    NO2_f = make_features(NO2_df, rolling_mean=True,
                          deltas=[24, 36, 48, 96])
    PM10_f = make_features(PM10_df, rolling_mean=True,
                           deltas=[24, 36, 48, 96])
    PM25_f = make_features(PM25_df, rolling_mean=True,
                           deltas=[24, 36, 48, 96])
    # apply each model
    Y_pred_NO2 = pd.DataFrame(model_dict["NO2"].predict(NO2_f),
                              columns=["TARGET"], index=NO2_f.index)
    Y_pred_PM10 = pd.DataFrame(model_dict["PM10"].predict(PM10_f),
                               columns=["TARGET"], index=PM10_f.index)
    Y_pred_PM25 = pd.DataFrame(model_dict["PM25"].predict(PM25_f),
                               columns=["TARGET"], index=PM25_f.index)
    # concatenate result
    Y_pred = pd.concatenate([Y_pred_NO2, Y_pred_PM10, Y_pred_PM25], axis=0)
    # return
    return Y_pred



