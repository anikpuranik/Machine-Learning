'''2. Feature Engineering'''
# importing libraries
import os
import numpy as np
import pandas as pd

# loading dataset
DIRECTORY = '/Users/aayushpuranik/.spyder-py3/dataset/house-prices-advanced-regression-techniques'
train_data = pd.read_csv(os.path.join(DIRECTORY, 'train.csv'))
test_data = pd.read_csv(os.path.join(DIRECTORY, 'test.csv'))

'''In data analysis we will analyze to find out:
    1. Missing Values.
    2. Temporal Variables.
    3. Categorical variables:To Labels.
    4. Normalze the values of the variable.'''
    
#1. Missing values
# Categorical features
cat_nan = [feature for feature in train_data 
                            if train_data[feature].isnull().sum() > 1 
                            and train_data[feature].dtypes == 'O'
                    ]

# Checking missing value proportion
for feature in cat_nan:
    print(feature, train_data[feature].isnull().mean())
    
# Replacing missing values with new label
def replace_cat_features(dataset, features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data

train_data = replace_cat_features(train_data, cat_nan)
train_data[cat_nan].isnull().sum()

# Numerical features
num_nan = [feature for feature in train_data
                   if train_data[feature].isnull().sum() > 1 
                   and train_data[feature].dtypes != 'O'
           ]

# Checking missing value proportion
for feature in num_nan:
    print(feature, train_data[feature].isnull().mean())
    
# Replacing the numerical Missing values
for feature in num_nan:
    #replace with median
    median_value = train_data[feature].median()
    
    # create new feature to capture nan values
    train_data[feature+'nan'] = np.where(train_data[feature].isnull(), 1, 0)
    train_data[feature].fillna(median_value, inplace = True)
    
train_data[num_nan].isnull().sum()

# Temporal variables
for feature in year_feature:
    train_data[feature] = train_data['YrSold'] - train_data[feature]

# Converting skew distribution into logrithmic
log_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in log_features:
    train_data[feature] = np.log(train_data[feature])

#3. Categorical
# Rare Categorical Feature
'''Remove the categorical variables that are present less than 1% of the observation.'''
for feature in categorical_data:
    temp = train_data.groupby(feature)['SalePrice'].count()/len(train_data)
    temp_df = temp[temp>0.01].index
    train_data[feature] = np.where(train_data[feature].isin(temp_df), 
                                   train_data[feature], 
                                   'Rare_var'
                                   )
for feature in categorical_data:
    labels_orderd = train_data.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_orderd = {k:i for i,k in enumerate(labels_orderd, 0)}
    train_data[feature] = train_data[feature].map(labels_orderd)

#4. Normalization
feature_scale = [feature for feature in train_data.columns 
                         if feature not in ['Id', 'SalePrice']
                ]

# We are using MinMaxScaler but try out with StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data[feature_scale])

scaler.transform(train_data[feature_scale])

data = pd.concat([train_data[['Id', 'SalePrice']].reset_index(drop=True),
                     pd.DataFrame(scaler.transform(train_data[feature_scale]), 
                     columns=feature_scale)], sort=False, axis=1)
    
data.head()
data.to_csv('training_dataset.csv', index=False)
