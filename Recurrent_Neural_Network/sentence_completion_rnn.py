# =============================================================================
#  Sentence Completion with RNN & LSTM
# =============================================================================
#
#  What this script does:
#  ──────────────────────
#  Trains two neural network models on 3,038 famous quotes to predict the
#  next word in any sentence — like autocomplete on your phone.
#
#  Example:
#    Input  →  "the world is"
#    Output →  "beautiful"   (whatever word the model learned comes next)
#
#  How it works — step by step:
#  ────────────────────────────
#  Step 1  →  Load 3,038 famous quotes from a CSV file
#  Step 2  →  Clean text — lowercase everything, strip all punctuation
#  Step 3  →  Tokenise — give each unique word a number (vocab = 10,000 words)
#  Step 4  →  Build training pairs — every prefix of every quote → next word
#             (3,038 quotes expand into 85,271 training samples)
#  Step 5  →  Pad sequences — all inputs made the same length (745)
#  Step 6  →  One-hot encode — each target word becomes a 10,000-long vector
#  Step 7  →  Build models — a SimpleRNN and an LSTM side by side
#  Step 8  →  Train the LSTM — 10 epochs, batch size 128, 10% validation
#  Step 9  →  Save the trained LSTM to models/lstm_model.keras
#  Step 10 →  (Optional) Use the predictor function to generate next words
#
# =============================================================================

# ──────────────────────────────────────────────
# Imports — standard libraries
# ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# GPU setup — use GPU if available, else fall back to CPU
# ──────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Training on GPU: {gpus[0].name}")
else:
    print("No GPU found — training on CPU")

# ──────────────────────────────────────────────
# Imports — Keras layers and utilities
# (using tf_keras for Keras 2 compatibility with TF 2.20+)
# ──────────────────────────────────────────────
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import Embedding, LSTM, Dense, SimpleRNN

# ──────────────────────────────────────────────
# Load dataset — 3,038 famous quotes
# ──────────────────────────────────────────────
df = pd.read_csv('data/qoute_dataset.csv')
# print(df.head())
# Output:
#                                                quote           Author
# 0  “The world as we have created it is a process ...  Albert Einstein
# 1  “It is our choices, Harry, that show what we t...     J.K. Rowling
# 2  “There are only two ways to live your life. On...  Albert Einstein
# 3  “The person, be it gentleman or lady, who has ...      Jane Austen
# 4  “Imperfection is beauty, madness is genius and...   Marilyn Monroe

# print(df.shape)
# Output: (3038, 2)  ← 3,038 rows, 2 columns (quote + Author)

quotes = df['quote']
# print(quotes.head())
# Output:
# 0    "The world as we have created it is a process ...
# 1    "It is our choices, Harry, that show what we t...
# 2    "There are only two ways to live your life. On...
# 3    "The person, be it gentleman or lady, who has ...
# 4    "Imperfection is beauty, madness is genius and ...
# Name: quote, dtype: object

# ──────────────────────────────────────────────
# Preprocessing — lowercase everything and remove punctuation
# so the model only sees clean, simple words
# ──────────────────────────────────────────────
quotes = quotes.str.lower()
translator =str.maketrans("","",string.punctuation)
quotes = quotes.apply(lambda x:x.translate(translator))
# print(quotes.head())
# Output (after cleaning):
# 0    the world as we have created it is a process ...
# 1    it is our choices harry that show what we tru...
# 2    there are only two ways to live your life one...
# 3    the person be it gentleman or lady who has no...
# 4    imperfection is beauty madness is genius and ...
# Name: quote, dtype: object

# ──────────────────────────────────────────────
# Tokenisation — assign a unique number to each word
# We keep only the top 10,000 most common words
# ──────────────────────────────────────────────
vocabulary_size = 10000
tokenizer = Tokenizer(num_words = vocabulary_size)
tokenizer.fit_on_texts(quotes)
word_index = tokenizer.word_index
# print(len(word_index))
# print(list(word_index.items())[:10])
# Output:
# 8978   ← total unique words found in the dataset
# [('the', 1), ('you', 2), ('to', 3), ('and', 4), ('a', 5), ('i', 6), ('is', 7), ('of', 8), ('that', 9), ('it', 10)]

sequence = tokenizer.texts_to_sequences(quotes)
# print(quotes[0])
# print(sequence[0])# Output:# “the world as we have created it is a process of our thinking it cannot be changed without changing our thinking”
# [713, 62, 29, 19, 16, 946, 10, 7, 5, 1156, 8, 70, 293, 10, 145, 12, 809, 104, 752, 70, 2461]

# ──────────────────────────────────────────────
# Build training pairs — for each quote, generate every possible
# prefix → next word combination
# e.g. "the world as" → next word is "we"
#      "the world"    → next word is "as"
# This expands 3,038 quotes into 85,271 training samples
# ──────────────────────────────────────────────
X = []
y = []

for seq in sequence:
    for i in range (1,len(seq)):
        input_seq = seq[:i]
        output_seq = seq[i]
        X.append(input_seq)
        y.append(output_seq)

# print(len(X))
# Output: 85271  ← 85,271 training samples
# print(len(y))
# Output: 85271

# ──────────────────────────────────────────────
# Padding — make all input sequences the same length
# by adding zeros at the front of shorter sequences
# e.g. [128]        → [0, 0, ..., 0, 128]
#      [136, 128]   → [0, 0, ..., 136, 128]
# ──────────────────────────────────────────────
max_length = max(len(x) for x in X)
# print(max_length)
# Output: 745  ← the longest quote has 745 words, so all inputs are padded to 745

X_padded = pad_sequences(X, maxlen=max_length, padding='pre')
# print(X_padded)
# [[   0    0    0 ...    0    0  713]
#  [   0    0    0 ...    0  713   62]
#  [   0    0    0 ...  713   62   29]
#  ...
#  [   0    0    0 ...    9   19 1125]
#  [   0    0    0 ...   19 1125    3]
#  [   0    0    0 ... 1125    3  169]]

# print(X_padded[0])
# Output: [  0   0   0  ..... 713]  ← 744 zeros followed by the first word number

y = np.array(y)
# print(X_padded.shape)
# Output: (85271, 745)  ← 85,271 samples, each 745 timesteps long
# print(y.shape)
# Output: (85271,)

# ──────────────────────────────────────────────
# One-hot encoding — convert each target word index into a
# vector of length 10,000 (all zeros except a 1 at the correct word's position)
# This tells the model exactly which word it should have predicted
# ──────────────────────────────────────────────
y_one_hot = to_categorical(y, num_classes=vocabulary_size)
# print(y_one_hot.shape)
# Output: (85271, 10000)  ← 85,271 samples, each is a 10,000-length vector

# ──────────────────────────────────────────────
# Build the RNN model
# Embedding: converts each word number into a 50-dimensional meaning vector
#            (words with similar meanings end up close together in that space)
#            This turns the 2D padded input into a 3D tensor: (samples, timesteps, embedding_dim)
# SimpleRNN: reads the sequence step by step, keeps one hidden state (128 units)
# Dense:     outputs a probability score for each of the 10,000 words
# ──────────────────────────────────────────────
embedding_dim = 50
rnn_units = 128
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=vocabulary_size,output_dim=embedding_dim,input_length=max_length))
# rnn_model.add(SimpleRNN(units=rnn_units,activation='tanh'))
rnn_model.add(SimpleRNN(units=rnn_units))
rnn_model.add(Dense(units=vocabulary_size,activation='softmax'))
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(rnn_model.summary())
#
# Model architecture:
#   Layer (type)       Output Shape          Param #
#   ────────────────────────────────────────────────
#   Embedding          (None, 745, 50)       500,000
#   SimpleRNN          (None, 128)            22,912
#   Dense              (None, 10000)       1,290,000
#   ────────────────────────────────────────────────
#   Total params: ~1.8M

# ──────────────────────────────────────────────
# Build the LSTM model
# Same structure as the RNN model above, but uses LSTM instead of SimpleRNN
# LSTM has two memory streams (hidden state + cell state) so it remembers
# context from much earlier in the quote — better for long sentences
# ──────────────────────────────────────────────
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length))
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocabulary_size, activation='softmax'))
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(lstm_model.summary())
#
# Model architecture:
#   Layer (type)       Output Shape          Param #
#   ────────────────────────────────────────────────
#   Embedding          (None, 745, 50)       500,000
#   LSTM               (None, 128)            91,648
#   Dense              (None, 10000)       1,290,000
#   ────────────────────────────────────────────────
#   Total params: ~1.9M  (more than RNN — the 4 gates cost extra params)

# ──────────────────────────────────────────────
# Train the RNN model (commented out — LSTM gives better results)
# ──────────────────────────────────────────────
# epochs = 10
# batch_size = 128
# history_rnn = rnn_model.fit(
#     X_padded, y_one_hot,
#     epochs=epochs,
#     batch_size=batch_size,
#     validation_split=0.2
# )
# rnn_model.save('models/rnn_model.keras')
# rnn_model.save('models/rnn_model.h5')

# ──────────────────────────────────────────────
# Train the LSTM model
# epochs=10     → the model sees all 85,271 samples 10 times
# batch_size=128 → processes 128 samples at a time before updating weights
# validation_split=0.1 → holds back 10% of data to check generalisation
# ──────────────────────────────────────────────
epochs = 10
batch_size = 128
history_rnn = lstm_model.fit(
    X_padded, y_one_hot,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1)

# Save the trained model (.keras is the recommended modern format)
lstm_model.save('models/lstm_model.keras')
lstm_model.save('models/lstm_model.h5')

# ──────────────────────────────────────────────
# Predictor function (uncomment to use after training)
# How it works:
#   1. Take a text prompt (e.g. "the world is")
#   2. Tokenise and pad it to max_length
#   3. Run it through the trained LSTM
#   4. Pick the word with the highest probability (argmax)
#   5. Convert the index back to a word and return it
# ──────────────────────────────────────────────
# index_to_word = {}
# for word,index in word_index.items():
#     index_to_word[index] = word
#
# def predictor(model,tokenizer,text,max_length):
#     text = text.lower()
#     seq = tokenizer.text_to_sequences([text])[0]
#     seq = pad_sequences([seq],maxlen =max_length,padding = 'pre')
#     pred = lstm_model.predict(seq,verbose = 0)
#     pred_index = np.argmax(pred)
#     return index_to_word[pred_index]
#
# seed_text = 'life is'
# next_word = predictor(lstm_model,tokenizer, seed_text,max_length)

# def genenrate_text(model,tokeniser,seed_text,max_length,n_words):
#     for _ in range(n_words):
#         next_word = predictor(model,tokenizer,seed_text,max_length)
#         if next_word == "":
#             break
#         seed_text +=" "+next_word
#     return seed_text
#
# seed = "the menaing of life"
# lstm_model = load_model("lstm_model)
# generate_text = genenrate_text(lstm_model,tokenizer,seed,max_length=10)
# print(generate_text)

import pickle
with open ("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

with open ("max_length.pkl","wb") as f:
    pickle.dump(max_length,f)
