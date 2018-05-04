# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:47:08 2018

@author: squirke
"""

##Importing libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##Importing libariesImporting the datat
##np.set_printoptions(threshold = np.nan)
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
##dfx = pd.DataFrame(X)
Y = dataset.iloc[:, 3].values


## splitting data into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

