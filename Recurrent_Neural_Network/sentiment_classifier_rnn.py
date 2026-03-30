# RNN Sentiment Classifier
# Classifies short sentences as positive (1) or negative (0) using a SimpleRNN.
# Architecture: Tokenizer → Padding → Embedding → SimpleRNN → Dense (sigmoid output)
#
# The second half of this file (Section 5) peeks inside the trained RNN and prints
# the hidden state it produced at every timestep — so you can see the memory
# building up word by word.

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense


# ── 1. Dataset ─────────────────────────────────────────────────────────────────
# 30 short sentences — first 15 are positive (label = 1), last 15 are negative (label = 0).
# This is small on purpose — just enough to show the model working end to end.

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
# A neural network only understands numbers — not words.
# So we first assign every unique word in our dataset a number (its "token id").
# Then we convert each sentence from words → list of those numbers.
#
# Example:
#   "I love this product"  →  [3, 26, 2, 7]
#
# Problem: sentences can have different lengths — but the RNN needs every input
# to be the same size. So we pad shorter sentences with 0s at the end:
#
#   [3, 26, 2, 7]  →  padded to  [3, 26, 2, 7, 0]   (0 = empty slot)
#
# vocab_size = 2000  → track up to 2000 unique words
# oov_token  = "<OOV>"  → if the model sees a word it never saw during training,
#                          use this placeholder instead of crashing

vocab_size = 2000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(s) for s in sequences)

X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = labels


# ── 3. Model Architecture ──────────────────────────────────────────────────────
# Three layers stacked together:
#
# Embedding layer
#   Turns each word's token id into a small dense vector (16 numbers per word).
#   This is smarter than one-hot encoding — words with similar meaning end up
#   with similar vectors. The embedding is learned during training.
#   Output shape:  (batch_size, max_length, 16)
#
# SimpleRNN layer
#   The heart of the model. Reads the sentence one word at a time, left to right.
#   At each word it updates a hidden state — its running memory of the sentence.
#   return_sequences=False  → only return the hidden state after the LAST word
#                             (we don't need the state at every step here)
#   return_state=False      → don't expose the state tensor as a separate output
#   Output shape:  (batch_size, 8)  — 8 numbers summarising the whole sentence
#
# Dense layer
#   Takes the 8-number summary and makes the final yes/no call.
#   1 neuron + sigmoid  → a single probability between 0 and 1
#     ≥ 0.5  →  positive sentiment  (1)
#     < 0.5  →  negative sentiment  (0)
#
# Full data shape flowing into the RNN:
#   [ batch_size  ×  max_length  ×  embedding_size ]
#   [     8       ×      5       ×       16        ]

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
# batch_size = 8  →  process 8 sentences at once, then update the weights
# epochs    = 25  →  go through all 30 sentences 25 times from start to finish

model.fit(X, y, epochs=25, batch_size=8, verbose=1)


# ── 5. Inspecting the Hidden States (Peek Inside the RNN) ──────────────────────
# After training, we want to see what the RNN was "thinking" at each word.
# The trained model only returns the final hidden state (after the last word).
# To see the hidden state at EVERY timestep we need to build a second model
# that uses the same trained weights but with return_sequences=True.
#
# Steps:
#   1. Build a new RNN layer with return_sequences=True
#   2. Copy the trained weights from the original RNN into it
#   3. Build a small "inspection model" using the same trained embedding
#   4. Run one sentence through it and print the hidden state per word
#
# This lets us see the memory building up step by step:
#   word 1 → hidden state (memory of word 1)
#   word 2 → hidden state (memory of words 1+2)
#   ...
#   word N → hidden state (memory of the full sentence)

from tensorflow.keras.layers import SimpleRNN as SRNN

seq_inp = Input(shape=(max_length,), dtype='int32')
seq_emb = model.get_layer('embed')(seq_inp)  # reuse the trained embedding — same weights

# Build a new RNN that outputs a hidden state at every timestep, not just the last
rnn_seq = SRNN(units=rnn_units, return_sequences=True, name='rnn_seq')

# DO NOT CALL build() manually — Keras builds the layer automatically on first call
seq_hidden = rnn_seq(seq_emb)

# Copy the weights learned by the trained RNN into this new one
# so we're inspecting the actual trained memory, not random fresh weights
try:
    trained_weights = model.get_layer('simple_rnn').get_weights()
    rnn_seq.set_weights(trained_weights)
    print("Copied RNN weights into sequence-inspection RNN.")
except Exception as e:
    print("Could not copy weights automatically:", e)

inspect_model = Model(inputs=seq_inp, outputs=seq_hidden)

# Pick the first sentence and run it through the inspection model
idx = 0
example_seq = X[idx:idx+1]          # shape (1, max_length) — one sentence
hidden_seq = inspect_model.predict(example_seq)

# Print what the RNN remembered at each word
print("Sentence:", sentences[idx])
print("Token ids:", example_seq)
print("Hidden states per timestep shape:", hidden_seq.shape)
print("Hidden states (timesteps x units):")
print(np.round(hidden_seq[0], 3))

# Expected output (values depend on training randomness):
#
# Copied RNN weights into sequence-inspection RNN.
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 163ms/step
# Sentence: I love this product
# Token ids: [[ 3 26  2  7  0]]
# Hidden states per timestep shape: (1, 5, 8)
# Hidden states (timesteps x units):
# [[-0.027  0.05  -0.004 -0.043  0.112 -0.021  0.057  0.1  ]   ← after "I"
#  [-0.051  0.052  0.264 -0.174  0.109  0.118 -0.048 -0.002]   ← after "love"
#  [-0.211 -0.136  0.022 -0.102 -0.202  0.159  0.095  0.092]   ← after "this"
#  [-0.162  0.26   0.174 -0.081  0.159  0.296  0.193  0.022]   ← after "product"
#  [-0.162  0.26   0.174 -0.081  0.159  0.296  0.193  0.022]]  ← after padding 0
