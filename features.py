#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing and features extraction

# features ideas

for time:
    - keep daytime
    - add hour of day = (daytime - daytime(0)) % 24
    - add day of year = floor(k * (daytime - daytime(0)) / 24)

for each temporal information:
    - avg of temporal information for each zone
    - difference of temporal information to average

- station_id [1,2,3,4,5]
- integer (0/1) for calm_day or not

for each static features
preprocessing:


    - fill na with 0
    - scale data with MaxAbsScaler to handle sparse data
features:
    - a features stat indicate if the |feature| > 0 - is not NaN actually
    - the preprocessed value
"""
import pandas as pd
from sklearn import preprocessing
from dataset import cols


def fillna_static(df):
    """ """
    for col in cols["static"]:
        df[col] = df[col].fillna(0)


def scale_temporal(train, features_train, dev=None, features_dev=None):
    """ """
    robust_scaler = preprocessing.RobustScaler()
    tmp = pd.DataFrame(
        robust_scaler.fit_transform(train[cols["temporal"]]),
        columns=["%s_sc" % col for col in cols["temporal"]],
        index=train.ID)
    features_train = pd.concat([features_train, tmp], axis=1)
    if dev is not None:
        tmp = pd.DataFrame(
            robust_scaler.transform(dev[cols["temporal"]]),
            columns=["%s_sc" % col for col in cols["temporal"]],
            index=dev.ID)
        features_dev = pd.concat([features_dev, tmp], axis=1)
    return features_train, features_dev


def scale_static(train, features_train, dev=None, features_dev=None):
    """ """
    # fill static NaN with 0
    fillna_static(train)
    if dev is not None:
        fillna_static(dev)
    # scale data with MaxAbsScaler to handle sparse static data
    max_abs_scaler = preprocessing.MaxAbsScaler()
    tmp = pd.DataFrame(
        max_abs_scaler.fit_transform(train[cols["static"]]),
        columns=["%s_sc" % col for col in cols["static"]],
        index=train.ID)
    features_train = pd.concat([features_train, tmp], axis=1)
    if dev is not None:
        tmp = pd.DataFrame(
            max_abs_scaler.transform(dev[cols["static"]]),
            columns=["%s_sc" % col for col in cols["static"]],
            index=dev.ID)
        features_dev = pd.concat([features_dev, tmp], axis=1)
    return features_train, features_dev


def binarize_static(train, features_train, dev=None, features_dev=None):
    """ """
    # binary static features
    binarizer = preprocessing.Binarizer()
    tmp = pd.DataFrame(
        binarizer.fit_transform(train[cols["static"]]),
        columns=["%s_i" % col for col in cols["static"]],
        index=train.ID)
    features_train = pd.concat([features_train, tmp], axis=1)
    if dev is not None:
        tmp = pd.DataFrame(
            binarizer.fit_transform(dev[cols["static"]]),
            columns=["%s_i" % col for col in cols["static"]],
            index=dev.ID)
        features_dev = pd.concat([features_dev, tmp], axis=1)
    return features_train, features_dev


def delta_temporal_station_zone(train, features_train, dev=None,
                                features_dev=None):
    """ """
    # add difference of average in zone and value in station
    for col in cols["temporal"]:
        features_train["delta_%s" % col] = train["%s_avg" % col] - train[col]
        if dev is not None:
            features_dev["delta_%s" % col] = dev["%s_avg" % col] - dev[col]
    # scaled it
    scaler = preprocessing.StandardScaler()
    features_train[["delta_%s" % col for col in cols["temporal"]]] = \
        scaler.fit_transform(
            features_train[["delta_%s" % col for col in cols["temporal"]]])
    if dev is not None:
        features_dev[["delta_%s" % col for col in cols["temporal"]]] = \
            scaler.transform(
                features_dev[["delta_%s" % col for col in cols["temporal"]]])
    return features_train, features_dev


def add_temporal_rolling_mean(delta, train, features_train, dev, features_dev):
    """ """
    # compute rolling mean of step delta
    rol_df = train.groupby("block")[cols["temporal"]] \
        .rolling(delta, min_periods=0).mean() \
        .reset_index(0, drop=True)
    rol_df.rename(
        columns=dict((col, "%s_mean_%i" % (col, delta))
                     for col in cols["temporal"]),
        inplace=True)
    features_train = features_train.merge(
        rol_df, left_index=True, right_index=True, copy=False)
    if dev is not None:
        rol_df = dev.groupby("block")[cols["temporal"]] \
            .rolling(delta, min_periods=0).mean() \
            .reset_index(0, drop=True)
        rol_df.rename(
            columns=dict((col, "%s_mean_%i" % (col, delta))
                         for col in cols["temporal"]),
            inplace=True)
        features_dev = features_dev.merge(
            rol_df, left_index=True, right_index=True,
            suffixes=("", "_mean_%i" % delta))
    # scale it
    scaler = preprocessing.RobustScaler()
    features_train[
        ["%s_mean_%i" % (col, delta) for col in cols["temporal"]]
    ] = scaler.fit_transform(
        features_train[["%s_mean_%i" % (col, delta)
                        for col in cols["temporal"]]])
    if dev is not None:
        features_dev[
            ["%s_mean_%i" % (col, delta) for col in cols["temporal"]]
        ] = scaler.transform(
            features_dev[["%s_mean_%i" % (col, delta)
                          for col in cols["temporal"]]])
    return features_train, features_dev


def make_features(train, dev=None, normalize=False,
                  rolling_mean=True, deltas=[]):
    """ """
    f_train = train[["ID", "zone_id", "daytime", "hour_of_day", "is_calmday"]]
    f_train.set_index("ID", inplace=True)
    # f_train.drop("ID", axis=1, inplace=True)
    if dev is not None:
        f_dev = dev[["ID", "zone_id", "daytime", "hour_of_day", "is_calmday"]]
        f_dev.set_index("ID", inplace=True)
        # f_dev.drop("ID", axis=1, inplace=True)
    else:
        f_dev = None
    # scale temporal features with robust scaling
    f_train, f_dev = scale_temporal(train, f_train, dev, f_dev)
    # scale data with MaxAbsScaler to handle sparse static data
    f_train, f_dev = scale_static(train, f_train, dev, f_dev)
    # binary static features
    f_train, f_dev = binarize_static(train, f_train, dev, f_dev)
    # add diff for temporal data between station value and zone avg
    f_train, f_dev = delta_temporal_station_zone(
        train, f_train, dev, f_dev)
    # Rolling mean of step delta
    if rolling_mean:
        for delta in deltas:
            f_train, f_dev = add_temporal_rolling_mean(
                delta, train, f_train, dev, f_dev)

    # l2 normalize
    if normalize:
        f_train = preprocessing.normalize(f_train)
        if dev is not None:
            f_dev = preprocessing.normalize(f_dev)
    #
    if dev is not None:
        return f_train, f_dev
    else:
        return f_train


if __name__ == '__main__':
    """ """
    # data = df[["ID"]]
    pass



