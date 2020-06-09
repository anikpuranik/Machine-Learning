from nltk.tokenize import word_tokenize
from nltk.collections import defaultdict
from nltk import bigrams, trigrams
import pickle

train_text = '''I lived my life happily and energetically. I am Aniket, I am from Ujjain. My schools name is CJCS. I studied in SGSITS, Indore.
I was an employee of Infosys. Currently, I am an alumini of Infosys and employee of Saggezza.'''

tokens = [word 
          for word in word_tokenize(train_text.lower()) 
          if word.isalpha()
          ]

def bigram_filler(text):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    
    for token in bigrams(tokens):
        model[token[0]][token[1]] += 1
    
    sent = word_tokenize(text.lower())[-1]
    suggestions = model[sent]
    
    return suggestions

    
def trigram_filler(text):
    model = defaultdict(lambda: defaultdict(lambda: 0))

    for token in trigrams(tokens):
        model[(token[0],token[1])][token[2]] += 1
    
    sent = tuple(word_tokenize(text.lower())[-2:])
    suggestions = model[sent]
    
    return suggestions


def suggest(text):
    if len(word_tokenize(text)) > 1:
        suggestion = trigram_filler(text)
    else:
        suggestion = bigram_filler(text)

    suggestions = dict(sorted(suggestion.items(), key=lambda kv: (kv[0], kv[1]))).keys()        
    return suggestions

test_text = 'I'
suggestions = suggest(test_text)
print(suggestions)

pickle_out = open('autofiller suggestion.pkl', 'wb')
pickle.dump(suggest, pickle_out)
pickle_out.close()
