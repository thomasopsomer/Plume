#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluate_mse(model, x_train, x_dev, y_train, y_dev):
    """ """
    # mse on training set
    y_pred_train = model.predict(x_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    print("MSE on training set: %.3f" % mse_train)
    # mse on test set
    y_pred_val = model.predict(x_dev)
    mse_val = mean_squared_error(y_dev, y_pred_val)
    print("MSE on test set: %.3f" % mse_val)
    #
    return mse_train, mse_val


def build_test_fold(train, dev):
    """ """
    a = pd.DataFrame(-1, index=train.index, columns=["fold"])
    b = pd.DataFrame(0, index=dev.index, columns=["fold"])
    test_fold = pd.concat([a, b], axis=0)
    return test_fold
