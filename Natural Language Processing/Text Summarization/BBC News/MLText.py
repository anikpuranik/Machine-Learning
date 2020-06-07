# importing inbuilt libraries
import os
# importing third party libraires
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.naive_bayes import GaussianNB
# importing inbuilt modules
import ML_approach as k


path = '/Users/aayushpuranik/.spyder-py3/dataset/BBC News Summary/BBC News Summary/News Articles/'
categories = os.listdir(path)
all_files = os.listdir('/Users/aayushpuranik/.spyder-py3/dataset/BBC News Summary/'
                       + 'BBC News Summary/News Articles/'+categories[0])

# loading text and summaries
def get_data(all_files):
    all_sentences = []                                                         # Storing all sentences
    all_summary = []                                                           # Storing complete summary
    all_words = []                                                             # Storing all words
    all_text = []                                                              # Storing all texts
    for file in all_files:
        text, summary = k.loading_file('business', file)
        text, title, paragraphs, words, sentences = k.data_extraction(text)
        summary_sentences = k.summary_extraction(summary)
        all_sentences.extend(sentences)
        all_summary.extend(summary_sentences)
        all_words.extend(words)
        all_text.extend(text)
    return all_summary, all_text
all_summary,all_text = get_data(all_files[:100])

# gather files
x = [k.fuzzy_dataset_generator('business', file) for file in all_files[:100]]
y = [k.fuzzy_dataset_generator('business', file) for file in all_files[350:351]]

# generating dataset
x = pd.concat(x)
y = y[0]
t = ''.join(all_text)
print(len(sent_tokenize(t)))

x.dropna(inplace=True)
x_train = x.iloc[:, :6].to_numpy()
x_labels = x.iloc[:, -1].to_numpy()
y_train = y.iloc[:, :6].to_numpy()
y_labels = y.iloc[:, -1].to_numpy()


# spliiting into training and test set
#x_train, x_test, y_train, y_test = train_test_splx, y, test_size=0.2, random_state=0)

# preparing the model
model = GaussianNB()
model = model.fit(x_train, x_labels)
prediction = model.predict(y_train)

# evaluating the model
result = (prediction == y_labels) 
print(result.sum())
