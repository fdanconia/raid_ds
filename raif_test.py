import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("./input"))


#git clone https://gitlab.com/YannBerthelot/kaggle_pystacknet.git
print(os.listdir("kaggle_pystacknet/pystacknet"))
#pip install "kaggle_pystacknet/pystacknet"
import pystacknet

train=pd.read_csv("./input/train.csv")
test=pd.read_csv("./input/test.csv")


def feature_engineering(df):
    df.fillna(0, inplace=True)
    return df.select_dtypes(include=['float64', 'int'])


train = feature_engineering(train)
test = feature_engineering(test)



X=train.drop(["per_square_meter_price",],axis=1)
Y=train["per_square_meter_price"]

X_test=test
#Y_test=test


# X.info()
# x_train, x_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [200, 500, 700, 1000],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],

}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=True)
CV_rfc.fit(X, Y)

print(CV_rfc.best_params_)

