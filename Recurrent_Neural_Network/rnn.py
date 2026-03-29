# RNN Sentiment Classifier
# Classifies short sentences as positive (1) or negative (0) using a SimpleRNN.
# Architecture: Tokenizer → Padding → Embedding → SimpleRNN → Dense (sigmoid output)

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense


# ── 1. Dataset ─────────────────────────────────────────────────────────────────
# 30 short sentences, first 15 are positive (label=1), last 15 are negative (label=0)

sentences = [
    "I love this product",
    "This movie made me smile",
    "Service was friendly and quick",
    "Today felt bright and happy",
    "This is the best day",
    "Absolutely fantastic experience",
    "I enjoyed every single moment",
    "Great job, well done",
    "The food tasted delicious",
    "Totally recommend to everyone",
    "Very satisfied with results",
    "This worked better than expected",
    "Amazing quality and value",
    "Such a pleasant surprise",
    "I feel positive about this",
    "I hate this product",
    "This movie bored me",
    "Service was rude and slow",
    "Today was cold and lonely",
    "This is the worst day",
    "Terrible experience overall",
    "I regret buying this",
    "Very disappointed with results",
    "The food tasted awful",
    "Do not recommend this",
    "It broke after one use",
    "Not worth the money",
    "Utterly frustrating and annoying",
    "I feel negative about this",
    "Such a waste of time",
]

labels = np.array([1] * 15 + [0] * 15)


# ── 2. Tokenisation and Padding ────────────────────────────────────────────────
# Convert each sentence into a sequence of integers (one integer per word).
# Pad all sequences to the same length so the RNN gets a fixed-size input.
#
# Example:
#   "I love this product"  →  [3, 26, 2, 7]  →  padded to  [3, 26, 2, 7, 0]
#                                                             ← extra 0 if shorter
#
# vocab_size=2000 means the tokenizer tracks up to 2000 unique words.
# oov_token="<OOV>" handles any word at test time that wasn't seen during training.

vocab_size = 2000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(s) for s in sequences)

X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = labels


# ── 3. Model Architecture ──────────────────────────────────────────────────────
# Embedding layer:  turns each word integer into a dense vector of size 16
#                   (learns a richer representation than one-hot — words with
#                    similar meaning end up with similar vectors)
#
# SimpleRNN layer:  8 hidden neurons, processes one word at a time
#                   return_sequences=False  → only give me the final hidden state
#                   return_state=False      → don't expose the state tensor separately
#
# Dense layer:      1 neuron + sigmoid → outputs a probability between 0 and 1
#                   ≥ 0.5  →  positive (1)
#                   < 0.5  →  negative (0)
#
# Full input shape to RNN:  [ batch_size  ×  max_length  ×  embedding_size ]
#                           [     8       ×      5       ×       16        ]

embedding_size = 16
rnn_units = 8

input_layer = Input(shape=(max_length,), dtype='int32', name="input")
x = Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True, name='embed')(input_layer)
x = SimpleRNN(units=rnn_units, return_sequences=False, return_state=False, name="simple_rnn")(x)
output_layer = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model: "functional"
# ┌─────────────────────┬───────────────────┬────────────┐
# │ Layer (type)        │ Output Shape      │    Param # │
# ├─────────────────────┼───────────────────┼────────────┤
# │ input (InputLayer)  │ (None, 5)         │          0 │
# │ embed (Embedding)   │ (None, 5, 16)     │     32,000 │
# │ simple_rnn          │ (None, 8)         │        200 │
# │ output (Dense)      │ (None, 1)         │          9 │
# └─────────────────────┴───────────────────┴────────────┘
#  Total params: 32,209


# ── 4. Training ────────────────────────────────────────────────────────────────
# batch_size=8  →  8 sentences processed together per weight update
# epochs=25     →  25 full passes through all 30 sentences

model.fit(X, y, epochs=25, batch_size=8, verbose=1)

