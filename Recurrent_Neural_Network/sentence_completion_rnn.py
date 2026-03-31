import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense, SimpleRNN

df = pd.read_csv('data/qoute_dataset.csv')
# print(df.head())
#                                                quote           Author
# 0  “The world as we have created it is a process ...  Albert Einstein
# 1  “It is our choices, Harry, that show what we t...     J.K. Rowling
# 2  “There are only two ways to live your life. On...  Albert Einstein
# 3  “The person, be it gentleman or lady, who has ...      Jane Austen
# 4  “Imperfection is beauty, madness is genius and...   Marilyn Monroe

# print(df.shape)
# (3038, 2) there are 3038 sentences/quotes

quotes = df['quote']
# print(quotes.head())
# 0    “The world as we have created it is a process ...
# 1    “It is our choices, Harry, that show what we t...
# 2    “There are only two ways to live your life. On...
# 3    “The person, be it gentleman or lady, who has ...
# 4    “Imperfection is beauty, madness is genius and...
# Name: quote, dtype: object

#  preprocessing data, removing capital letters,punctuations,etc
quotes = quotes.str.lower()
translator =str.maketrans("","",string.punctuation)
quotes = quotes.apply(lambda x:x.translate(translator))
# print(quotes.head())
# 0    “the world as we have created it is a process ...
# 1    “it is our choices harry that show what we tru...
# 2    “there are only two ways to live your life one...
# 3    “the person be it gentleman or lady who has no...
# 4    “imperfection is beauty madness is genius and ...
# Name: quote, dtype: object

# tokenizing the data to get vocabulary

vocabulary_size = 10000
tokenizer = Tokenizer(num_words = vocabulary_size)
tokenizer.fit_on_texts(quotes)
word_index = tokenizer.word_index
# print(len(word_index))
# print(list(word_index.items())[:10])
# 8978
# [('the', 1), ('you', 2), ('to', 3), ('and', 4), ('a', 5), ('i', 6), ('is', 7), ('of', 8), ('that', 9), ('it', 10)]

sequence = tokenizer.texts_to_sequences(quotes)
# print(quotes[0])
# print(sequence[0])
# “the world as we have created it is a process of our thinking it cannot be changed without changing our thinking”
# [713, 62, 29, 19, 16, 946, 10, 7, 5, 1156, 8, 70, 293, 10, 145, 12, 809, 104, 752, 70, 2461]

X = []
y = []

for seq in sequence:
    for i in range (1,len(seq)):
        input_seq = seq[:i]
        output_seq = seq[i]
        X.append(input_seq)
        y.append(output_seq)

# print(len(X))
# 85271
# print(len(y))
# 85271

#  as the length is very big lets use padding to reduce/standardise length length
# [128] -> [0,0,128]
# [136,128] - > [0,136,128]

max_length = max(len(x) for x in X)
# print(max_length)
# 745

X_padded = pad_sequences(X,maxlen = max_length,padding = 'pre')
# print(X_padded)
# [[   0    0    0 ...    0    0  713]
#  [   0    0    0 ...    0  713   62]
#  [   0    0    0 ...  713   62   29]
#  ...
#  [   0    0    0 ...    9   19 1125]
#  [   0    0    0 ...   19 1125    3]
#  [   0    0    0 ... 1125    3  169]]

# print(X_padded[0])
# [  0   0   0  ..... 713]

y = np.array(y)
# print(X_padded.shape)
# (85271, 745)
# print(y.shape)
# (85271,)

#  one hot encoding

y_one_hot = to_categorical(y,num_classes =vocabulary_size)
# print(y_one_hot.shape)
# (85271, 10000)

# tokenization->vectorization->embeddings
embedding_dim = 50
rnn_units = 128
rnn_model = Sequential()
# inside each vector each value will be converted into embeddings, a 3d vecotr will be created called tensor
rnn_model.add(Embedding(input_dim=vocabulary_size,output_dim=embedding_dim,input_length=max_length))
# rnn_model.add(SimpleRNN(units=rnn_units,activation='tanh'))
rnn_model.add(SimpleRNN(units=rnn_units))
rnn_model.add(Dense(units=vocabulary_size,activation='softmax'))
rnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# print(rnn_model.summary())
# ------creating lstm model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length))
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocabulary_size, activation='softmax'))
lstm_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(lstm_model.summary())
