# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:59:54 2020

@author: 61426
"""

import os 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

DATA_PATH = 'datasets//housing.csv'

housing = pd.read_csv(DATA_PATH)

housing.hist(bins=50,figsize=(20,15))
plt.show()

# 分为训练集和测试集
trian_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)

housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5,5.0,inplace=True)

# 分层抽样
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print(housing['income_cat'].value_counts()/len(housing))

for set in (strat_train_set,strat_test_set):
    set.drop(['income_cat'],axis=1,inplace=True)
    
housing = strat_train_set.copy()

housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=housing['population']/100,label='population',\
             c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True)    
plt.show()

# 相关性
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
scatter_matrix(housing[attributes],figsize=(12,8))

housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['polulation_per_household'] = housing['population']/housing['households']

coor_martix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))


housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_test_set["median_house_value"].copy()

# 处理缺值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)

# 处理文字
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)

# 独热向量处理类别
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

# 自定义转换器
from sklearn.base import BaseEstimator,TransformerMixin
rooms_ix,bedrooms_ix,polulation_ix,household_ix = 3,4,5,6

