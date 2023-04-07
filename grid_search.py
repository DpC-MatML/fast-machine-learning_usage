#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import json

from fml.parameter_opt import GridSearch
from fml.feature_selection import MRMR
from fml.validates import Validate
from fml.data import read_data, DataObject
from fml.sampling import random_split

# 导入算法
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
# from xgboost import XGBRegressor




data= DataObject().from_df(
    pd.read_csv('BMG_all.csv', index_col=0).iloc[:, 1:]
)

print(data.X)
print(data.Y)
# data = DataObject(
#     X=data.iloc[:, 3:].values,
#     Y=data.iloc[:, 2].values,
#     Xnames=data.columns[3:],
#     Yname=data.columns[2],
#     indexes=data.iloc[:, 0],
# )

# v = Validate(SVR, data, **{"kernel": "linear", "C": 1, "epsilon": 0.01, "gamma": 0.5})
# v.validate_train()

# train_obj, test_obj = random_split(data, percent=0.2)

"""
    #  SVR和对应的参数设置
# algo = SVR
# params_dict = dict(
#     C=np.linspace(1, 100, 100),
#     epsilon=np.linspace(0.001, 0.1, 100),
#     gamma=0.006
# )

# algo = SVR
# params_dict = dict(
#     C=17,
#     epsilon=0.001,
#     gamma=0.006
# )

#  XGBoost和对应的参数设置
# algo = XGBRegressor
# params_dict = dict(
#     learning_rate=np.linspace(0.01, 1, 50),
#     max_depth=np.linspace(1, 6, 6),
#     min_child_weight=np.linspace(1, 6, 6)
# )

# GBR和对应的参数设置
algo = GradientBoostingRegressor
params_dict = dict(
    # n_estimators=np.linspace(1, 101, 100),
    n_estimators=range(20, 201, 20),
    learning_rate=np.linspace(0.001, 0.1, 100)
    # max_depth=range(3, 10, 1)

)

algo = GradientBoostingRegressor
params_dict = dict(
    n_estimators=120,
    learning_rate=0.06
)
"""

normalize = Normalize.FeatureScaler('normalize')
scaler = Normalize.TargetMapping()
data.X = normalize(X)
data.Y = scaler(Y)

scaler_file = open('scaler.pkl', 'wb')
pickle.dump(scaler, scaler_file)
scaler_file.close()



gbr = SVR
params_dict_svr = dict(
    # n_estimators=np.linspace(1, 101, 100),
    n_estimators=range(20, 201, 20),
    learning_rate=np.linspace(0.001, 0.1, 100)
    # max_depth=range(3, 10, 1)

)

# # 导入网格搜索，留一法
gridsearch = GridSearch(n_jobs=6)
gridsearch.fit(gbr, data, cv=True, **params_dict_svr)
result = gridsearch.results
print(type(gridsearch.best_result))
# print('resultt:::')
print(gridsearch.results)