# importing libraries
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

# loading dataset
def file_to_be_summarised(path='/Users/aayushpuranik/.spyder-py3/Text Summarization/text2.txt'):
    with open(path, 'r') as f:
        text = f.read().lower()
    return text

# getting the text
text = file_to_be_summarised()

# converting into text
paragraphs = text.split('\n')

# generating word_frequency matrix
def word_freq_matrix(text):
    # stopwords matrix
    stop_words = stopwords.words('english')             
    # word_matrix
    word_matrix = (word 
                   for word in word_tokenize(text) 
                   if (word not in stop_words
                       and word.isalpha()
                       and len(word)>1)
                   )
    return FreqDist(word_matrix)    
word_frequency = word_freq_matrix(text)

# sentence matrix
def sentence_generation(text, all_words):
    sent_matrix = []
    for sent in sent_tokenize(text):
        words = [word for word in word_tokenize(sent) if word.isalpha()]
        if len(words) > 0:
            sent_matrix.append(words)
    return sent_matrix

sentences = sentence_generation(text, word_frequency.keys())