# Tokenizer allows to vectorize sentences (each word converted to integer) based on binary, word_freq, tf_idf.
# pad_sequences helps in sentences uniform by providing features like maximum length, truncation, type of padding. 
# importing libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love my dog!",
    "I love my cat",
    "you love my dog and cat"
    "Amazing this is a do"
    ]

# intializing parameters of tokenizer
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# dictionary of word with index's
word_index = tokenizer.word_index
print("Word indexs:\n",word_index)

# generating sequences based in word index
sequences = tokenizer.texts_to_sequences(sentences)
print("Sequence:\n",sequences)

# Here, we are trying to encode the word we have trained earlier on
checks = [
    "I love Pizza",
    "I am dog fan"
    ]
check_sequences = tokenizer.texts_to_sequences(checks)
print(check_sequences)

# padding the sequence
padded_sequence = pad_sequences(sequences, maxlen=10, value=-1,
                                padding='post', truncating='pre',
                                )
print("Paading sequence:\n",padded_sequence)

padded_check_sequence = pad_sequences(check_sequences, maxlen=5, 
                                padding='post', truncating='pre',
                                )
print("Paading check sequence:\n",padded_check_sequence)
