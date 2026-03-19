import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# DATASET
# This is a small made-up dataset to keep things simple for learning.
# In real life you'd use thousands of rows, but for understanding
# how an ANN works, 16 rows is perfectly fine.
#
# We're trying to predict: does this plant need water? (yes=1, no=0)
# Based on three things we can measure:
#   - Soil moisture  (how wet is the soil?)
#   - Temperature    (how hot is it outside?)
#   - Sunlight hours (how many hours of sun did it get today?)
# -------------------------------------------------------------------
df = pd.DataFrame({
    "soil_moisture": [0.10, 0.15, 0.20, 0.25, 0.40, 0.60, 0.35, 0.18,
                      0.45, 0.05, 0.80, 0.27, 0.55, 0.70, 0.12, 0.30],
    "temperature_c": [34, 30, 26, 22, 28, 30, 19, 22,
                      35, 24, 33, 33, 21, 25, 20, 29],
    "sunlight_hours": [9, 8, 7, 4, 8, 10, 3, 10,
                       12, 5, 9, 11, 2, 6, 1, 9],
    "needs_water": [1, 1, 1, 0, 0, 0, 0, 1,
                    0, 1, 0, 1, 0, 0, 1, 1]
})

# print(df)
#     soil_moisture  temperature_c  sunlight_hours  needs_water
# 0            0.10             34               9            1
# 1            0.15             30               8            1
# 2            0.20             26               7            1
# 3            0.25             22               4            0
# 4            0.40             28               8            0
# 5            0.60             30              10            0
# 6            0.35             19               3            0
# 7            0.18             22              10            1
# 8            0.45             35              12            0
# 9            0.05             24               5            1
# 10           0.80             33               9            0
# 11           0.27             33              11            1
# 12           0.55             21               2            0
# 13           0.70             25               6            0
# 14           0.12             20               1            1
# 15           0.30             29               9            1

# Separate inputs (X) from the answer we want to predict (y)
X = df[['soil_moisture', 'temperature_c', 'sunlight_hours']]
y = df['needs_water']

# -------------------------------------------------------------------
# NORMALIZATION
# The three input columns are in completely different units —
# soil moisture is 0 to 1, temperature is 20 to 35, sunlight is 1 to 12.
# That mismatch confuses the model (it thinks temperature matters
# way more just because the numbers are bigger).
#
# Normalization fixes this by rescaling every column to 0-1 range.
# Formula: (value - min) / (max - min)
#
# The tiny +1e-8 at the end just protects against dividing by zero
# in case a column somehow has the same min and max.
# -------------------------------------------------------------------
X_min = X.min()
X_max = X.max()
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)

# print(X_scaled)
#     soil_moisture  temperature_c  sunlight_hours
# 0        0.066667         0.9375        0.727273
# 1        0.133333         0.6875        0.636364
# 2        0.200000         0.4375        0.545455
# 3        0.266667         0.1875        0.272727
# 4        0.466667         0.5625        0.636364
# 5        0.733333         0.6875        0.818182
# 6        0.400000         0.0000        0.181818
# 7        0.173333         0.1875        0.818182
# 8        0.533333         1.0000        1.000000
# 9        0.000000         0.3125        0.363636
# 10       1.000000         0.8750        0.727273
# 11       0.293333         0.8750        0.909091
# 12       0.666667         0.1250        0.090909
# 13       0.866667         0.3750        0.454545
# 14       0.093333         0.0625        0.000000
# 15       0.333333         0.6250        0.727273

# -------------------------------------------------------------------
# TRAIN / TEST SPLIT
# We keep 75% of the data for training (so the model can learn from it)
# and hold back 25% for testing (to check if it learned properly).
#
# stratify=y makes sure both splits have a fair mix of 0s and 1s —
# without this, by bad luck all the 1s could end up in one split.
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# -------------------------------------------------------------------
# MODEL ARCHITECTURE
# We're building a simple 3-layer neural network:
#
#   Input Layer  — 3 neurons (one for each feature)
#   Hidden Layer — 8 neurons, ReLU activation
#   Output Layer — 1 neuron, Sigmoid activation (gives 0 to 1 probability)
#
# Diagram:
#
#   soil_moisture  ─┐
#   temperature_c  ─┼──►  [ 8 neurons, ReLU ]  ──►  [ 1 neuron, Sigmoid ]  ──►  needs_water (0 or 1)
#   sunlight_hours ─┘
#
# Why ReLU in the hidden layer?
#   Fast, avoids vanishing gradient, works great for most problems.
#
# Why Sigmoid at the output?
#   We need a probability between 0 and 1 for a yes/no prediction.
# -------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# -------------------------------------------------------------------
# COMPILE
# Tells the model HOW to learn:
#   optimizer='sgd'                  — Stochastic Gradient Descent,
#                                      the method used to update weights
#   loss='binary_crossentropy'       — loss function for yes/no problems,
#                                      measures how wrong each prediction is
#   metrics=['accuracy']             — also track what % we're getting right
# -------------------------------------------------------------------
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------------------------------------------
# TRAINING
# epochs=100 means the model sees the full training data 100 times.
# Each time it looks at the data, it gets a little better.
# batch_size=4 means it updates weights every 4 rows instead of
# waiting to see all rows — this makes training faster and less noisy.
# -------------------------------------------------------------------
history = model.fit(
    X_train.values, y_train.values, validation_data=(X_test.values, y_test.values), epochs=100, batch_size=4)

# -------------------------------------------------------------------
# UNDERSTANDING THE TRAINING OUTPUT
#
# loss         — how wrong the model is on the training data.
#                Lower is better. The model actively tries to reduce this
#                by updating weights using gradient descent.
#
# accuracy     — what percentage of training examples the model got right.
#
# val_loss     — same as loss, but measured on the test data (data the
#                model has NEVER seen during training). This tells you
#                whether the model has actually learned to generalise,
#                or if it has just memorised the training examples.
#
# val_accuracy — percentage correct on test data.
#
# What to look for:
#   If accuracy is high but val_accuracy is much lower → overfitting.
#   The model memorised the training data but can't handle new data.
#
#   If both go up together → the model is genuinely learning.
# -------------------------------------------------------------------