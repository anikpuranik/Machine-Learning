# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# loading dataset
dataset = pd.read_csv('BankNote_Authentication.csv')
print(dataset)

# Independent and Dependent Features
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=0)

### Implement Random Forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

## Prediction
y_pred = classifier.predict(X_test)

### Check Accuracy
score = accuracy_score(y_test, y_pred)
print(score)

### Create a Pickle file using serialization 
pickle_out = open("docker.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
