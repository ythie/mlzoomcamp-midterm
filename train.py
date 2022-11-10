#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import math

# parameters

output_file = f'model.bin'


# data preparation

df = pd.read_csv('SolarPrediction.csv', sep=',')
print('Sample of original dataframe:\n', df.head(), '\n')

dfnum = df.drop(['UNIXTime', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
dfnum['Temperature'] = (5 / 9 * (dfnum['Temperature'] - 32)).round(2) # Convert to Celsius
print('Sample of numerical dataframe:\n', dfnum.head(), '\n')

# training 

X_train, X_test, y_train, y_test = train_test_split(dfnum, dfnum.pop('Radiation'), test_size=0.2, random_state=42, shuffle=True)
print('X_train:\n', X_train.head(), '\n')
print('X_test:\n', X_test.head(), '\n')
print('y_train:\n', y_train.head(), '\n')
print('y_test:\n', y_test.head(), '\n')

# training the final model

print('Training the final model\n')

dtrain = xgb.DMatrix(X_train, label=y_train)
xgb_params = {
    'eta': 0.1, 
    'max_depth': 10,
    'min_child_weight': 5,

    'objective': 'reg:squarederror',
    'subsample': 1.0,

    'colsample_bytree': 1.0,
    'seed': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200)
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)
print('Sample of predicted values:\n', y_pred[:5], '\n')

score = math.sqrt(mean_squared_error(y_test.values, y_pred))
print('RMSE: ', score, '\n')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump(model, f_out)

print(f'The model is saved to {output_file}')