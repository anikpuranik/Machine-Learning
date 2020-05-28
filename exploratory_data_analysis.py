'''1. Exploratory Data Analysis'''
# importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading dataset
DIRECTORY = '/Users/aayushpuranik/.spyder-py3/dataset/house-prices-advanced-regression-techniques'
train_data = pd.read_csv(os.path.join(DIRECTORY, 'train.csv'))
test_data = pd.read_csv(os.path.join(DIRECTORY, 'test.csv'))

print(train_data.shape)
train_data.head()
# Now ID is not the importannt column in dataset. So we can remove it.


'''In data analysis we will analyze to find out:
    1. Missing Values
    2. Numerical Values
    3. Categorical Variables
    4. Cardinality of Categorical Variables
    5. Outliers
    6. Relationship between independent and dependent variables'''


# 1. Missing values
null_data = [features 
             for features in train_data.columns
             if train_data[features].isnull().sum()>1
             ]

for feature in null_data:
    print(feature, np.round(train_data[feature].isnull().mean(), 4), '%missing values')
    
# Impact of missing values on dependent variable
for feature in null_data:
    data = train_data.copy()
    
    # conveting all null values to 1 and rest to 0
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # Calculate the mean of SalePrice where data is missing
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    '''For all the missing values the Dependent Variable is very high. 
    Thus this comes as an important feature.'''
        
    
# 2. Numerical Variables
num_data = train_data.select_dtypes(include=['int64','float64'])

# Temporal Variable (Eg: Datetime Variables)
year_feature = [feature
                for feature in num_data 
                if ('Yr' in feature 
                    or 'Year' in feature)
                ]
print(feature, num_data[year_feature].head())
train_data.groupby('YrSold')['SalePrice'].median().plot()

#Since price is dropping with the time which is usually not the behaviour in real life.
#We will investigate furthur. We are checking SalePrice with repect no of years.
for feature in year_feature:
    if feature!='YrSold':
        data = num_data.copy()
        data[feature] = data['YrSold']-data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(feature)
        plt.show()
        
# Checking discreate feature
discrete_feature = [feature 
                    for feature in num_data 
                    if (len(num_data[feature].unique())<25 
                        and feature not in year_feature+['Id'])
                    ]

# Dependency of Discrete variable to SalePrice dependent variables
for feature in discrete_feature:
    data = train_data.copy()
    data.groupby(feature)['SalePrice'].median().plot()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

# Checking continous feature
continous_feature = [feature 
                     for feature in num_data 
                     if feature not in discrete_feature+year_feature+['Id']
                     ]

# Dependency of Continous variable to SalePrice dependent variables
for feature in continous_feature:
    data = train_data.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()
    
'''Now this data is not a gaussian distribution.
For solving regression problem we need to work on gaussian distribution
So we need to convert all the non-gaussian continous variables into 
gaussian variables.'''
# Converting into logirithmic scale.
for feature in continous_feature:
    data = train_data.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.title(feature + ' vs SalePrice')
        plt.show()   
        
# Outliars
for feature in continous_feature:
    data = train_data.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.title(feature + ' vs SalePrice')
        plt.show()
    # There are multiple outliars
# This only work for continous variable
        

# 3.Categorical Feature
cat_data = train_data.select_dtypes(include='O')
print(cat_data.head())
'''cardinality means no. of unique category in categorical feature.'''

# Cardinality of Categorical Features
for feature in cat_data:
    print(feature, len(cat_data[feature].unique()))
'''We have three class with greater than 10 unique categories.
So, we avoid those variables to directly categorize to unique numbers.
Other than than we can go ahead with one-hot encoding or pd_get_dummies.'''

# Dependency of categorical variable to SalePrice
for feature in cat_data:
    data = train_data.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    
