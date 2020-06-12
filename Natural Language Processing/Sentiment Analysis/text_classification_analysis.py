# importing libraries
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#-------------------------------loading dataset--------------------------------
dataset = pd.read_csv('/Users/aayushpuranik/.spyder-py3/dataset/Sentiments.csv', delimiter='\t')
classes = list(set(dataset.Labels))
#-------------------------------preprocessing----------------------------------
def preprocessing(dataset):
    reviews = []
    for text in dataset.Reviews:
        text = text.lower()
        text = re.sub('''[!.*%:,$()&-/+"[0-9]]*''', '', text)
        reviews.append(text)
    dataset.Reviews = reviews
    return dataset
dataset = preprocessing(dataset)
#---------------------------------tf_idf---------------------------------------
def Tf_idf(dataset):
    vect = TfidfVectorizer()
    review = vect.fit_transform(dataset.Reviews).toarray()
    return pd.DataFrame(review), vect
reviews,vectorizer = Tf_idf(dataset)
reviews['labels'] = dataset.Labels
dataset = reviews
#------------------------------feature selection-------------------------------
# 1.Odds Ratio
def odd_ratio(dataset, clas, dependent_variable='labels'):
    feature_values = dict()
    dependent_variable = 'labels'
    for column in dataset.columns:
        if column != dependent_variable:
            n_f_ck = sum(dataset[dependent_variable][dataset[column]!=0]==clas)
            n_f_ck_not = sum(dataset[dependent_variable][dataset[column]!=0]!=clas)           
            n_ck = sum(dataset[dependent_variable]==clas)
            n_ck_not = sum(dataset[dependent_variable]!=clas)           
            num = (n_f_ck/n_ck) * (1 - n_f_ck_not/n_ck_not)
            deno = (n_f_ck_not/n_ck_not)*(1 - n_f_ck_not/n_ck_not)           
            if deno == 0 or num == 0:
                feature_values[column] = 0
            else:
                value = np.log(num/deno)
                if value!= 0:
                    feature_values[column] = value
                else:
                    feature_values[column] = 0           
    return feature_values  

def odd_ratio_feature_generator(dataset, classes, dependent_variable='labels'):
    odd_feature = []
    for clas in classes:
        odd_feature.append(odd_ratio(dataset.copy(), clas))
    odd_feature = pd.DataFrame(odd_feature)
    return odd_feature


# 2.chi-square
def chi_square(dataset, clas, dependent_variable='labels'):
    feature_values = dict()
    for column in dataset.columns:
        if column != dependent_variable:
            n_f_ck = sum(dataset[dependent_variable][dataset[column]!=0]==clas)
            n_f_ck_not = sum(dataset[dependent_variable][dataset[column]!=0]!=clas)
            n_f_not_ck = sum(dataset[dependent_variable][dataset[column]==0]==clas)
            n_f_not_ck_not = sum(dataset[dependent_variable][dataset[column]==0]!=clas)           
            n_f = sum(dataset[column]!=0)
            n_f_not = sum(dataset[column]==0)
            n_ck = sum(dataset[dependent_variable]==clas)
            n_ck_not = sum(dataset[dependent_variable]!=clas)           
            num = len(dataset) * pow((n_f_ck*n_f_not_ck_not) - (n_f_ck_not*n_f_not_ck), 2)
            deno = n_f*n_f_not*n_ck*n_ck_not           
            if num == 0 or deno == 0:
                feature_values[column] = 0
            else:
                feature_values[column] = num/deno   
    return feature_values

def chi_square_feature_generator(dataset, classes, dependent_variable='labels'):
    chi_feature = []
    for clas in classes:
        chi_feature.append(chi_square(dataset.copy(), clas))
    chi_feature = pd.DataFrame(chi_feature)
    return chi_feature


# 3. gss coefficient
def gss_coefficient(dataset, clas, dependent_variable = 'labels'):
    feature_values = dict()
    for column in dataset.columns:
        if column != dependent_variable:
            n_f_ck = sum(dataset[dependent_variable][dataset[column]!=0]==clas)
            n_f_ck_not = sum(dataset[dependent_variable][dataset[column]!=0]!=clas)
            n_f_not_ck = sum(dataset[dependent_variable][dataset[column]==0]==clas)
            n_f_not_ck_not = sum(dataset[dependent_variable][dataset[column]==0]!=clas)           
        feature_values[column] = (n_f_ck*n_f_not_ck_not) - (n_f_ck_not*n_f_not_ck)
    return feature_values

# gss
def gss_feature_generator(dataset, classes, dependent_variable = 'labels'):
    gss_feature = []
    for clas in classes:
        gss_feature.append(gss_coefficient(dataset.copy(), clas))
    gss_feature = pd.DataFrame(gss_feature)
    return gss_feature

#--------------------splitting training and testing data-----------------------
def splitting_train_test(reviews, labels):
    x_train, x_test, y_train, y_test = tts(reviews, labels, train_size=0.8, random_state=0)
    return x_train, x_test, y_train, y_test
#-----------------training model and validating model--------------------------
def training_models(x_train, x_test, y_train, y_test, model, model_name, dataset_name):
    model.fit(x_train, y_train)
    pre = model.predict(x_test)
    cm = confusion_matrix(y_test, pre)
    print(cm, model_name, dataset_name, sum(np.diagonal(cm)))
    return model



odd_feature = odd_ratio_feature_generator(dataset, classes[:-1])
#chi_values=chi_feature.max()
odd_columns = list(odd_feature.loc[:, (odd_feature==0).any(axis=0)].columns)
odd_data = dataset.drop(columns=odd_columns)

chi_feature = chi_square_feature_generator(dataset, classes[:-1])
#chi_values=chi_feature.max()
chi_columns = list(chi_feature.loc[:, (chi_feature==0).any(axis=0)].columns)
chi_data = dataset.drop(columns=chi_columns)

gss_feature = gss_feature_generator(dataset, classes[:-1])
#gss_values=gss_feature.max()
gss_columns = list(gss_feature.loc[:, (gss_feature==0).any(axis=0)].columns)
gss_data = dataset.drop(columns=gss_columns)

models = []

labels = odd_data['labels']
odd_data.drop(columns=['labels'], inplace=True)
x_train, x_test, y_train, y_test = splitting_train_test(odd_data, labels)
odd_ratio_log_model = training_models(x_train, x_test, y_train, y_test, LogisticRegression(), 'Logistic', 'odd_data')
odd_ratio_naive_model = training_models(x_train, x_test, y_train, y_test, GaussianNB(), 'NaiveBayes', 'odd_data')
odd_ratio_svm_model = training_models(x_train, x_test, y_train, y_test, SVC(gamma='auto'), 'SVM', 'odd_data')
odd_ratio_decision_tree_model = training_models(x_train, x_test, y_train, y_test, DecisionTreeClassifier(), 'DecisionTree', 'odd_data')
models.append([odd_ratio_log_model, 'odd_ratio_log_model'])
models.append([odd_ratio_naive_model, 'odd_ratio_naive_model'])
models.append([odd_ratio_svm_model, 'odd_ratio_svm_model'])
models.append([odd_ratio_decision_tree_model, 'odd_ratio_decision_tree_model'])

labels = chi_data['labels']
chi_data.drop(columns=['labels'], inplace=True)
x_train, x_test, y_train, y_test = splitting_train_test(chi_data, labels)
chi_log_model = training_models(x_train, x_test, y_train, y_test, LogisticRegression(), 'Logistic', 'chi_data')
chi_naive_model = training_models(x_train, x_test, y_train, y_test, GaussianNB(), 'NaiveBayes', 'chi_data')
chi_svm_model = training_models(x_train, x_test, y_train, y_test, SVC(gamma='auto'), 'SVM', 'chi_data')
chi_decision_tree_model = training_models(x_train, x_test, y_train, y_test, DecisionTreeClassifier(), 'DecisionTree', 'chi_data')
models.append([chi_log_model, 'chi_log_model'])
models.append([chi_naive_model, 'chi_naive_model'])
models.append([chi_svm_model, 'chi_svm_model'])
models.append([chi_decision_tree_model, 'chi_decision_tree_model'])

labels = gss_data['labels']
gss_data.drop(columns=['labels'], inplace=True)
x_train, x_test, y_train, y_test = splitting_train_test(gss_data, labels)
gss_log_model = training_models(x_train, x_test, y_train, y_test, LogisticRegression(), 'Logistic', 'gss_data')
gss_naive_model = training_models(x_train, x_test, y_train, y_test, GaussianNB(), 'NaiveBayes', 'gss_data')
gss_svm_model = training_models(x_train, x_test, y_train, y_test, SVC(gamma='auto'), 'SVM', 'gss_data')
gss_decision_tree_model = training_models(x_train, x_test, y_train, y_test, DecisionTreeClassifier(), 'DecisionTree', 'gss_data')
models.append([gss_log_model, 'gss_log_model'])
models.append([gss_naive_model, 'gss_naive_model'])
models.append([gss_svm_model, 'gss_svm_model'])
models.append([gss_decision_tree_model, 'gss_decision_tree_model'])


#----------------------------testing model-------------------------------------
text = 'The food was good.'
text = text.lower()
text = re.sub('''[!.*%:,$()&-/+"[0-9]]*''', '', text)
vector = vectorizer.transform([text]).toarray()

odd_vector = pd.DataFrame(vector.copy())
odd_vector.drop(columns=odd_columns, inplace=True)

chi_vector = pd.DataFrame(vector.copy())
chi_vector.drop(columns=chi_columns, inplace=True)

gss_vector = pd.DataFrame(vector.copy())
gss_vector.drop(columns=gss_columns, inplace=True)

for ind,model in enumerate(models):
    if ind<4:
        print(model[0].predict(odd_vector), end=' ')
    elif 3<ind<9:
        print(model[0].predict(chi_vector), end=' ')
    else:
        print(model[0].predict(gss_vector), end=' ')
    print(model[1])

