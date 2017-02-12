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
from sklearn.metrics import mean_squared_error
import cPickle as pickle


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
                                     deltas_mean=[24, 48, 96, 144, 288])
        f[poll]["Y"] = get_Y(labels, ds[poll])
    # train model for each pollutant
    model_dict = {}
    for poll in pollutants:
        xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=200,
                                     reg_lambda=10)
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
                          deltas_mean=[24, 48, 96, 144, 288])
    PM10_f = make_features(PM10_df, rolling_mean=True,
                           deltas_mean=[24, 48, 96, 144, 288])
    PM25_f = make_features(PM25_df, rolling_mean=True,
                           deltas_mean=[24, 48, 96, 144, 288])
    # apply each model
    Y_pred_NO2 = pd.DataFrame(model_dict["NO2"].predict(NO2_f),
                              columns=["TARGET"], index=NO2_f.index)
    Y_pred_PM10 = pd.DataFrame(model_dict["PM10"].predict(PM10_f),
                               columns=["TARGET"], index=PM10_f.index)
    Y_pred_PM25 = pd.DataFrame(model_dict["PM25"].predict(PM25_f),
                               columns=["TARGET"], index=PM25_f.index)
    # concatenate result
    Y_pred = pd.concat([Y_pred_NO2, Y_pred_PM10, Y_pred_PM25], axis=0)
    # return
    return Y_pred


def train_predict(train, test, Y_train, model_dict=None):
    """ """
    pollutants = ["NO2", "PM10", "PM25"]
    # split dataset, build data dict
    train_ds = dict(((poll, df) for poll, df in zip(pollutants, split_pollutant_dataset(train))))
    test_ds = dict(((poll, df) for poll, df in zip(pollutants, split_pollutant_dataset(test))))
    # build features dict
    f = {}
    for poll in pollutants:
        f[poll] = {}
        f[poll]["X_train"], f[poll]["X_test"] = make_features(
            train_ds[poll], dev=test_ds[poll],
            rolling_mean=True, deltas_mean=[24, 48, 96, 144, 288])
        if Y_train is not None:
            f[poll]["Y"] = get_Y(Y_train, train_ds[poll])
    # train model for each pollutant
    if model_dict is None:
        model_dict = {}
        for poll in pollutants:
            xgb_model = xgb.XGBRegressor(max_depth=6, n_estimators=200,
                                         reg_lambda=10)
            # train model
            xgb_model.fit(f[poll]["X_train"], f[poll]["Y"])
            # store model
            model_dict[poll] = xgb_model
    # predict on train set
    preds = []
    for poll in pollutants:
        # mse on training set
        Y_pred_poll = pd.DataFrame(
            model_dict[poll].predict(f[poll]["X_train"]),
            columns=["TARGET"],
            index=f[poll]["X_train"].index)
        preds.append(Y_pred_poll)
        mse = mean_squared_error(f[poll]["Y"], Y_pred_poll)
        print("%s: MSE on training set: %.3f" % (poll, mse))
    # concat and compute global MSE
    Y_pred = pd.concat(preds, axis=0).sort_index()
    mse = mean_squared_error(Y_train, Y_pred)
    print("GLOBAL MSE on training set: %.3f" % mse)
    # predict on test set
    preds = []
    for poll in pollutants:
        Y_pred_poll = pd.DataFrame(
            model_dict[poll].predict(f[poll]["X_test"]),
            columns=["TARGET"],
            index=f[poll]["X_test"].index)
        preds.append(Y_pred_poll)
    # concatenate pred for each pollutant and sort index
    Y_pred = pd.concat(preds, axis=0).sort_index()
    #
    return Y_pred


if __name__ == '__main__':
    # Paths
    X_train_path = "/Users/thomasopsomer/data/plume-data/X_train.csv"
    X_test_path = "/Users/thomasopsomer/data/plume-data/X_test.csv"
    Y_train_path = "/Users/thomasopsomer/data/plume-data/Y_train.csv"
    # prepare data / features
    X_train = pd.read_csv(X_train_path, index_col="ID")
    X_train = preprocess_dataset(X_train)
    Y_train = pd.read_csv(Y_train_path, index_col="ID")
    X_test = pd.read_csv(X_test_path, index_col="ID")
    X_test = preprocess_dataset(X_test)
    # train
    model_dict = train(X_train, Y_train)
    # save models
    pickle.dump(model_dict, open("model/model_dict_2.pkl", 'wb'))
    # predict
    Y_pred = predict(model_dict, X_test)



