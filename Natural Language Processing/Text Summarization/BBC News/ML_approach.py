#------------------------importing libraries-----------------------------------
# inbuilt libraires
import os
import re
# third party libraries
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#-------------------------loading data-----------------------------------------
def loading_file(file):

    Directory = '/Users/aayushpuranik/.spyder-py3/dataset/BBC News Summary/BBC News Summary'
    #1. text data
    with open(os.path.join(Directory, 'News Articles/business/'+file), 'r') as f:
        text_data = f.read()
    
    #2. summary data
    with open(os.path.join(Directory, 'Summaries/business/'+file), 'r') as f:
        summary_data = f.read()    

    return text_data, summary_data

#actual_text, actual_summary = loading_file('001.txt')
#------------------------text preprocessing------------------------------------                                        # Stemmer Initialization using Snowball Stemmer
#1. Extracting Title, Text, Sentences, Words, Paragraphs
def data_extraction(text):
    paragraphs = text.split('\n')
    title = paragraphs[0]
    paragraphs = paragraphs[1:]
    text = ' '.join(paragraphs[1:])
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    return text, title, paragraphs, words, sentences    
#text, title, paragraphs, words, sentences = data_extraction(actual_text)
    
#2. Extracting Summary into sentences
def summary_extraction(text):
    '''Sentences are not in proper format. So sentence segmentation wont work.
    Thus, we are splitting sentences and combining again in required format.'''
    first_letters = re.findall('[.]["]*[A-Z]', text)                                
    summary_sentences = re.split('[.]["]*[A-Z]', text)
    for ind, sent in enumerate(summary_sentences):
        if ind!=0:
            sent = first_letters[ind-1][1:] + sent
        if ind!=len(summary_sentences)-1:
            sent += '.'
        summary_sentences[ind] = sent    
    return summary_sentences
#summary_sentences = summary_extraction(actual_summary)
        
#3. # Removing stopwords, numbers and special characters
def removing_unneccessay_words(words, text) -> list:
    stop_words = stopwords.words('english')   
    stemmer = SnowballStemmer('english')               
    words = [stemmer.stem(word)
             for word in words
             if (word not in stop_words
                 and word.isalpha()                                            
                 )
             ]
    word_frequency = FreqDist(word_tokenize(text))
    return words, word_frequency
#words, word_frequency = removing_unneccessay_words(words, text)

#4. Forming tokenized sentence with relevant wordset
def sentence_word_filter(words, sentences) -> list:
    tokenized_sentences = []                                                   # space separated words in a sentence
    stemmer = SnowballStemmer('english')
    for sentence in sentences:
        word_set = [stemmer.stem(word)                                         # word set of each sentences
                    for word in word_tokenize(sentence.lower())
                    ]
        sent = [stemmer.stem(word)                                             # sentence with relevant words             
                for word in word_set 
                if word in words
                ]
        tokenized_sentences.append(sent)
    return tokenized_sentences
#tokenized_sentences = sentence_word_filter(words, sentences)

#---------------------------sentence features----------------------------------
#feature_set = []                                                               # containing all features
# 1. Title Feature
def title_feature_extraction(title, sentences, tokenized_sentences) -> list:
    stemmer = SnowballStemmer('english')   
    title_words = [stemmer.stem(word) 
                   for word in word_tokenize(title.lower())
                   ]
    title_feature = dict()
    for token_sentence,sentence in zip(tokenized_sentences, sentences):
        for word in title_words:
            if word in token_sentence:
                title_feature[sentence] = title_feature.get(sentence, 0) + 1
        title_feature[sentence] = title_feature.get(sentence, 0)/len(title_words)
    return title_feature
#title_feature = title_feature_extraction(title.lower(), sentences, tokenized_sentences)
#feature_set.append(title_feature)

# 2. Sentence Length
def sentence_length_extraction(sentences, tokenized_sentences) -> dict:
    sentence_length = {sentence:len(token_sentence) 
                       for sentence, token_sentence in zip(sentences, tokenized_sentences)
                       }
    max_length_sentence = max(sentence_length.values())
    sentence_length = {strings:number/max_length_sentence 
                       for strings,number in sentence_length.items()
                       }
    return sentence_length
#sentence_length = sentence_length_extraction(sentences, tokenized_sentences)
#feature_set.append(sentence_length)

# 3. Term weight (TF or TF-IDF)
def term_weight_generation(text, words, word_frequency, 
                           sentences, tokenized_sentences) -> dict:
    term_weight = dict()        
    for sentence, token_sentence in zip(sentences, tokenized_sentences):
        for word in token_sentence:
            term_weight[sentence] = (term_weight.get(sentence, 0)
                                     + word_frequency[word]
                                     )
    max_term_weight = max(term_weight.values())
    term_weight = {sentence:value/max_term_weight 
                   for sentence, value in term_weight.items()
                   }
    return term_weight
#term_weight = term_weight_generation(text, words, word_frequency, sentences, tokenized_sentences)
#feature_set.append(term_weight)

# 4. Sentence Position
def sentence_position_extraction(paragraphs):
    sentence_position = dict()
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        for position,sentence in enumerate(sentences):
            sentence_position[sentence] = (len(sentences)-position)/len(sentences)
    return sentence_position

#sentence_position = sentence_position_extraction(paragraphs)
#feature_set.append(sentence_position)

# 5. Semntence to sentence similarity
def sentence_similarity_extraction(sentences, tokenized_sentences):
    sentence_similarity = {}
    vectorizer = CountVectorizer()
    sent = [' '.join(token_sentence) for token_sentence in tokenized_sentences]
    
    bag_of_words_matrix = vectorizer.fit_transform(sent)
    bag_of_words_matrix = bag_of_words_matrix.toarray()
    
    sent_similarity = cosine_similarity(bag_of_words_matrix)
    np.fill_diagonal(sent_similarity, 0)
    sent_similarity_final = sum(sent_similarity)
    sent_similarity_final = sent_similarity_final/sum(sent_similarity_final)
    
    sentence_similarity = {sentence:sent_similarity_final[ind]
                           for ind,sentence in enumerate(sentences)}

    return sentence_similarity
#sentence_similarity = sentence_similarity_extraction(sentences, tokenized_sentences)
#feature_set.append(sentence_similarity)

# 6. Numerical Data
def numerical_data_extraction(sentences):
    numerical_data = {}
    for sentence in sentences:
        word_list = word_tokenize(sentence)
        for word in word_list:
            if re.search('\d', word) is not None:
                numerical_data[sentence] = numerical_data.get(sentence, 0) + 1
        numerical_data[sentence] = numerical_data.get(sentence, 0)/len(word_list)
    return numerical_data

#numerical_data = numerical_data_extraction(sentences)
#feature_set.append(numerical_data)
#-----------------------fuzzy logic in featurs---------------------------------
'''For applying logic all the values have been checked for each feature and 
based on the values lower_threshold and higher threshold is chosed.'''

def fuzzy_logic(dictionary, sentences):
    max_value = max(dictionary.values())
    for key in dictionary:
        if dictionary[key] >= 4*max_value/5:
            dictionary[key] = 2
        elif dictionary[key] >= 3*max_value/5:
            dictionary[key] = 1
        elif dictionary[key] <= max_value/5:
            dictionary[key] = -2
        elif dictionary[key] < 2*max_value/5:
            dictionary[key] = -1
        else:
            dictionary[key] = 0
    return dictionary

def generate_fuzzy_features(sentences, features):
    feature_set = []
    for feature in features:
        feature_set.append(fuzzy_logic(feature, sentences))
    return feature_set
#feature_set = generate_fuzzy_features(sentences, feature_set)






#---------------------generating fuzzy dataset---------------------------------
def features_generator(file='001.txt'):
    #1. loading file
    actual_text, actual_summary = loading_file(file)
    
    #2. data extraction
    text, title, paragraphs, words, sentences = data_extraction(actual_text)
    summary_sentences = summary_extraction(actual_summary)
    words, word_frequency = removing_unneccessay_words(words, text)
    tokenized_sentences = sentence_word_filter(words, sentences)
    
    #3. creating feature set for all the features
    feature_set = []
    feature_set.append(title_feature_extraction(title.lower(), 
                                                sentences, tokenized_sentences))
    feature_set.append(sentence_length_extraction(sentences, tokenized_sentences))
    feature_set.append(term_weight_generation(text, words, word_frequency, 
                                              sentences, tokenized_sentences))
    feature_set.append(sentence_position_extraction(paragraphs))
    feature_set.append(sentence_similarity_extraction(sentences, tokenized_sentences))
    feature_set.append(numerical_data_extraction(sentences))
    
    #4. passing through fuzzy generator
    feature_set = generate_fuzzy_features(sentences, feature_set)  
    
    return feature_set, summary_sentences

# fuzzy_dataset, summary_sentences = fuzzy_dataset_generator()

    
    
'''After obtaining fuzzy dataset we are left with two options.
1. We can generate summary by scoring the sentence based on all the values.
   Summary generated by fuzzy logic set holds good for over 80% similarity with
   the summary performed by humans and can be confirmed.
2. We can use summary generated by humans and then create a summarization ML model
   by generating classification problem. For this we need large dataset and it 
   is observed have to much better performance than any method obtained so far.
   '''
   
#-----------------1. Summary generated by Fuzzy logic--------------------------
def fuzzy_summary():
    #1. generating fuzzy dataset and summary 
    fuzzy_dataset, summary_sentences = features_generator()
    
    #2. scoring sentences
    score_matrix = pd.DataFrame(fuzzy_dataset).T
    score_matrix = score_matrix.sum(axis=1)

    #3. summarizing text
    summary = ''
    threshold = 0
    for ind, sentence in enumerate(score_matrix.index):
        if score_matrix.iloc[ind] > threshold:
            summary+= sentence + ' '

    #4. displaying summary
    print(summary)


#-------------2. This is the creation of machine learning model----------------
def fuzzy_dataset_generator(file):
    fuzzy_dataset, summary_sentences = features_generator()
    fuzzy_dataset = pd.DataFrame(fuzzy_dataset).T
    sentence_present = list()
    for sentence in fuzzy_dataset.index:
        sentence_present.append(int(sentence in summary_sentences))
    fuzzy_dataset[fuzzy_dataset.shape[1]] = sentence_present
    return fuzzy_dataset

#fuzzy_dataset.append(fuzzy_dataset_generator())
#final_dataset = pd.DataFrame(fuzzy_dataset).T