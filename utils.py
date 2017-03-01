#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from features import make_features
from dataset import split_train_dev, get_Y


def evaluate_mse(model, x_train, x_dev, y_train, y_dev):
    """ """
    # mse on training set
    y_pred_train = model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    #print("MSE on training set: %.3f" % mse_train)
    # mse on test set
    y_pred_val = model.predict(x_dev)
    mse_val = mean_squared_error(y_dev, y_pred_val)
    #print("MSE on test set: %.3f" % mse_val)
    #
    return mse_train, mse_val


def build_test_fold(train, dev):
    """ """
    a = pd.DataFrame(-1, index=train.index, columns=["fold"])
    b = pd.DataFrame(0, index=dev.index, columns=["fold"])
    test_fold = pd.concat([a, b], axis=0)
    return test_fold


def get_feature_importances(df, model):
    """Plot feature importance like xgb.plot_importances()"""
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    features = df.columns
    indices = np.argsort(importances)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(np.arange(0, len(indices) * 2, 2), importances[indices],
             height=1, color='b', align='center')
    plt.yticks(np.arange(0, len(indices) * 2, 2), features[indices])
    plt.margins(y=0)
    plt.show()


def shuffle_XY(X, Y):
    """ """
    tmp = pd.concat([X, Y], axis=1).sample(frac=1)
    return tmp[X.columns], tmp[Y.columns]


def my_split_fold(X, Y, k=3, features_config={}, info=False):
    """
    Split dataset on train dev, choising random station
    to put in the dev set
    Then make the features and yield tuples of (train, dev)
    features as many times as k.
    """
    for i in range(k):
        # split data set in train dev
        train, dev, train_info, dev_info = split_train_dev(X, info=True)
        if info:
            print("train: %i samples in zone - station %s: "
                  % (train.shape[0], train_info))
            print("dev: %i samples in zone - station %s: "
                  % (dev.shape[0], dev_info))
        # build features for train and dev set
        f_train, f_dev = make_features(train, dev, **features_config)
        # get corresponding labels
        y_train = get_Y(Y, f_train)
        y_dev = get_Y(Y, f_dev)
        yield f_train, f_dev, y_train, y_dev, (train_info, dev_info)


def random_split_fold(X, k=3):
    """
    Split dataset on train dev, choising random station
    to put in the dev set
    Then make the features and yield tuples of (train, dev)
    features as many times as k.
    """
    for i in range(k):
        # split data set in train dev
        train, dev, train_info, dev_info = split_train_dev(X, info=True)
        yield train, dev, train_info, dev_info


