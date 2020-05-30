# importing libraies
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# loading text data
def file_to_be_summarised(path='/Users/aayushpuranik/.spyder-py3/Text Summarization/text2.txt'):
    with open(path, 'r') as f:
        text = f.read().lower()
    return text

# getting the text
text = file_to_be_summarised()

# sentence segmentation
sentences = sent_tokenize(text)

# generating tf-idf matrix
def tf_idf_matrix_generator(sentences):
    vectorizer = TfidfVectorizer()
    to_values = vectorizer.fit_transform(sentences)
    tf_idf_matrix = to_values.toarray()
    return tf_idf_matrix
tf_idf_matrix = tf_idf_matrix_generator(sentences)

# generating evaluating matrix
def required_matrix(tf_idf_matrix, sentences):
    sentence_score_matrix = sum(tf_idf_matrix.T)
    for ind, sentence in enumerate(sentences):
        no_of_words = len(word_tokenize(sentence))
        sentence_score_matrix[ind] /= no_of_words
    return sentence_score_matrix
sentence_score_matrix = required_matrix(tf_idf_matrix, sentences)

# scoring rank
def generating_summary(sentence_score_matrix, sentences):
    threshold = sorted(sentence_score_matrix, reverse=True)[10]
    summary = ''
    for numbers, strings in zip(sentence_score_matrix, sentences):
        if numbers > threshold:
            summary += strings
    return summary
summary = generating_summary(sentence_score_matrix, sentences)

# display summary
print(summary)

