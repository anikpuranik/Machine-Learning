#------------------------importing libraries-----------------------------------
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#----------------------------text data-----------------------------------------
text = '''Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.
I’ve coached mummy and mom clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”
I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.
“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don’t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you’re willing to put yourself on the line or settle for less. And that’s fine if you’re content to receive less, as long as you’re not regretful later.
If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.
Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I’m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you’re an overnight sensation, to sustain it for long, particularly if you don’t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success — simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.
So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what’s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don’t leave your dreams to chance.'''

#---------------------------data preprocessing---------------------------------
# 1. Sentence Segmentation
def sentence_segmentation(text):
    title = 'Those Who Are Resilient Stay In The Game Longer'
    text = text[len(title):]
    sentences = sent_tokenize(text.lower())
    return title.lower(), text.lower(), sentences

title, text, sentences = sentence_segmentation(text)

# 2. Tokenization
# 3. Removing Stop Word
# 4. Word Stemming
def data_preprocess(text):
    words = set(word_tokenize(text))
    stop_words = stopwords.words('english')
    #stemmer = SnowballStemmer('english')
    words = [word
             for word in words 
             if (word not in stop_words and
                 word.isalpha() and
                 len(word) > 1)
             ]
    tokenized_sentences = []
    for sentence in sentences:
        sent_words = [word 
                     for word in word_tokenize(sentence)
                     if word in words
                     ]
        tokenized_sentences.append(sent_words)
    return words, tokenized_sentences

words, tokenized_sentences = data_preprocess(text)

#---------------------------sentence features----------------------------------
# 1. Title Feature
def title_feature_extraction(title, sentences, tokenized_sentences) -> dict:
    title_words = word_tokenize(title)
    title_feature = dict()
    for token_sentence,sentence in zip(tokenized_sentences, sentences):
        for word in token_sentence:
            if word in title_words:
                title_feature[sentence] = title_feature.get(sentence, 0) + 1
        title_feature[sentence] = title_feature.get(sentence, 0)/len(token_sentence)
    return title_feature

title_feature = title_feature_extraction(title, sentences, tokenized_sentences)

# 2. Sentence Length
def sentence_length_extraction(sentences, tokenized_sentences):
    sentence_length = {sentence:len(token_sentence) 
                       for sentence, token_sentence in zip(sentences, tokenized_sentences)
                       }
    max_length_sentence = max(sentence_length.values())
    sentence_length = {strings:number/max_length_sentence 
                       for strings,number in sentence_length.items()}
    return sentence_length

sentence_length = sentence_length_extraction(sentences, tokenized_sentences)

# 3. Term weight (TF or TF-IDF)
def term_weight_generation(text, words, sentences, tokenized_sentences):
    term_weight = dict()
    word_frequency = FreqDist(word_tokenize(text))
    for word in word_frequency.copy():
        if word not in words:
            del word_frequency[word]
            
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

term_weight = term_weight_generation(text, words, sentences, tokenized_sentences)

# 4. Sentence Position
def sentence_position_extraction(sentences):
    sentence_position = dict()
    for position,sentence in enumerate(sentences):
        sentence_position[sentence] = (len(sentences)-position)/len(sentences)
    return sentence_position

sentence_position = sentence_position_extraction(sentences)


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

sentence_similarity = sentence_similarity_extraction(sentences, tokenized_sentences)

# 6. Numerical Data
def numerical_data_extraction(sentences):
    numerical_data = {}
    for sentence in sentences:
        word_list = word_tokenize(sentence)
        for word in word_list:
            if word.isdigit():
                numerical_data[sentence] = numerical_data.get(sentence, 0) + 1
        numerical_data[sentence] = numerical_data.get(sentence, 0)/len(word_list)
    return numerical_data

numerical_data = numerical_data_extraction(sentences)

#-----------------------fuzzy logic in featurs---------------------------------
'''For applying logic all the values have been checked for each feature and 
based on the values lower_threshold and higher threshold is chosed.'''
def fuzzy_logic(dictionary, Vh_thresh, H_thresh, L_thresh, VL_thresh):
    for key in dictionary:
        if dictionary[key] >= Vh_thresh:
            dictionary[key] = 2
        elif dictionary[key] > H_thresh:
            dictionary[key] = 1
        elif dictionary[key] < VL_thresh:
            dictionary[key] = -2
        elif dictionary[key] < L_thresh:
            dictionary[key] = -1
        else:
            dictionary[key] = 0
    return dictionary

title_feature = fuzzy_logic(title_feature, 0.4, 0.2, 0.1, 0.05)
sentence_length = fuzzy_logic(sentence_length, 0.5, 0.3, 0.15,0.1)
term_weight = fuzzy_logic(term_weight, 0.5, 0.25, 0.18, 0.1)
sentence_position = fuzzy_logic(sentence_position, 0.9, 0.75, 0.4, 0.25)
sentence_similarity = fuzzy_logic(sentence_position, 0.32, 0.24, 0.2, 0.1)
numerical_data = fuzzy_logic(numerical_data, 0.1, 0.05, 0.005, 0.001)
    
#-----------------------scoring sentences--------------------------------------
def sentence_scoring(sentences, title_feature, 
                     sentence_length, sentence_position):
                        score_matrix = dict()
                        for sentence in sentences:
                            score_matrix[sentence] = (title_feature[sentence]
                                                      + sentence_length[sentence]
                                                      + sentence_position[sentence]
                                                      + term_weight[sentence]
                                                      + sentence_similarity[sentence]
                                                      + numerical_data[sentence]
                                                      )
                        return score_matrix
                    
score_matrix = sentence_scoring(sentences, title_feature, 
                                sentence_length, sentence_position)

#-----------------------summarizing text---------------------------------------
def summary_generation(sentences, score_matrix, threshold=0):
    summary = ''
    for sentence in sentences:
        if score_matrix[sentence] > threshold:
            summary+= sentence
    return summary

summary = summary_generation(sentences, score_matrix, threshold=2)
#------------------------final ouput-------------------------------------------
# displaying summary
print(summary)
