# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os

irisdf = pd.read_csv('datasets/IRIS.csv')
#irisdf.tail()
#irisdf.info()

from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()
irisdf['species_enc'] = labelEncoder.fit_transform(irisdf['species'])

# oddelime predikujici featury od target variable
# y - target variable
ydf = irisdf['species_enc'].values
# x - hodnoty z petal_lengh a petal_width
xdf = irisdf[['petal_length', 'petal_width']].values

from sklearn.model_selection import train_test_split
Xtrain, Xtest = train_test_split(xdf,test_size=0.2)
Ytrain, Ytest = train_test_split(ydf,test_size=0.2)

from sklearn.preprocessing import StandardScaler
# udelame standardizaci numerickych features pomoci StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.fit_transform(Xtest)