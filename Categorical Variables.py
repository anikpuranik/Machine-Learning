import pandas as pd
data = ['Hello','Hey','Hi']
data2 = data.copy()

''' We transform the data into numbers within a single column.'''
from sklearn.preprocessing import LabelEncoder
data = LabelEncoder().fit_transform(data)
>>> output: 
  array([0, 1, 2])

''' We transform the data into numbers into the three different columns.'''
transform = pd.get_dummies(data2)
'''
>>> output : 
   Hello  Hey  Hi
0      1    0   0
1      0    1   0
2      0    0   1
'''
