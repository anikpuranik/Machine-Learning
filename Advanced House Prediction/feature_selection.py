'''3. Feature Selection'''
# importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# loading dataset
DIRECTORY = '/Users/aayushpuranik/.spyder-py3/Advance House Prediction Regression'
dataset = pd.read_csv(os.path.join(DIRECTORY, 'training_dataset.csv'))
dataset.head()

# Capture dependent variable
y_train = dataset.SalePrice

# drop dependent and Id column
x_train = dataset.drop(columns=['SalePrice','Id'])

'''First, specify alpha foe Lasso Regression model.
The bigger the alpha the lesser the number of features will be selected.
Then use selectFromModel to select feature which coefficients are non-zero.'''

# very essential to say random_state=0 i.e. seed_value
feature_sel_model = SelectFromModel(Lasso(alpha=0.001, random_state=0))
feature_sel_model.fit(x_train, y_train)
feature_sel_model.get_support()

# important features in the dataset for given alpha
selected_features = x_train.columns[(feature_sel_model.get_support())]

# required dataset
x_train = x_train[selected_features]

# converting to np array for model training
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train)
