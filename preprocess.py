import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def standard_scale(xTrain, xTest):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(xTrain)
    X_test_scaled = scaler.transform(xTest)
    return X_train_scaled, X_test_scaled


def minmax_range(xTrain, xTest):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(xTrain)  
    X_test_scaled = scaler.transform(xTest)        
    return X_train_scaled, X_test_scaled


def add_irr_feature(xTrain, xTest):
    train_irr = np.random.normal(0, 50, size=(xTrain.shape[0], 2))
    X_train_modified = np.hstack((xTrain, train_irr))
    test_irr = np.random.normal(0, 50, size=(xTest.shape[0], 2))
    X_test_modified = np.hstack((xTest, test_irr))
    return X_train_modified, X_test_modified
