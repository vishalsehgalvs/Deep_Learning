# =============================================================================
# IRIS SPECIES CLASSIFIER — ANN Project
# =============================================================================
# Goal: Predict which species of Iris flower a sample belongs to.
#
# The Iris dataset has 150 flower samples, each with 4 measurements:
#   - Sepal Length
#   - Sepal Width
#   - Petal Length
#   - Petal Width
#
# There are 3 species to classify (50 samples each):
#   - Iris-setosa     (label 0)
#   - Iris-versicolor (label 1)
#   - Iris-virginica  (label 2)
#
# We build two models and compare them:
#   1. Perceptron  — a single-layer model, the simplest classifier
#   2. ANN         — multi-layer neural network with Softmax output
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical


# =============================================================================
# LOAD DATA
# =============================================================================
# Read the Iris CSV file into a DataFrame
# Each row is one flower sample with 4 measurements + the species label
df = pd.read_csv('Iris.csv')

# First 5 rows look like this:
#    Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
# 0   1            5.1           3.5            1.4           0.2  Iris-setosa
# 1   2            4.9           3.0            1.4           0.2  Iris-setosa
# 2   3            4.7           3.2            1.3           0.2  Iris-setosa
# 3   4            4.6           3.1            1.5           0.2  Iris-setosa
# 4   5            5.0           3.6            1.4           0.2  Iris-setosa


# =============================================================================
# EXPLORE THE DATA
# =============================================================================
# Check how many samples we have per species — should be 50 each (balanced)
# Species
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50

# Pair plot — shows how each pair of features separates the three species
# Saved to: images/iris_species_pairplot.png
# sns.pairplot(df, hue='Species')
# plt.show()


# =============================================================================
# PREPARE THE DATA
# =============================================================================
# Drop the 'Species' column (that is what we want to predict)
# and the 'Id' column (just a row number, no useful info)
X = df.drop(columns=['Species', 'Id'], axis=True)

# The target (y) is the Species column — text labels like 'Iris-setosa'
y = df['Species']

# LabelEncoder turns text labels into numbers so the model can process them:
#   Iris-setosa     → 0
#   Iris-versicolor → 1
#   Iris-virginica  → 2
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Encoded output looks like this:
# [0 0 0 ... 1 1 1 ... 2 2 2]

# Split into training (80%) and test (20%) sets
# stratify=y_encoded makes sure each split has a balanced mix of all 3 species
# random_state=42 makes the split the same every time you run it
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# StandardScaler normalises the features — shifts each column so it has
# mean=0 and std=1. This stops one large-valued feature from dominating.
# fit_transform on train: learns the scale from training data
# fit_transform on test:  applies the same scaling to test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# =============================================================================
# MODEL 1 — PERCEPTRON (the simplest possible classifier)
# =============================================================================
# A Perceptron is a single-layer model — one neuron per class basically.
# It can only learn straight-line decision boundaries.
# Good baseline to compare against the ANN.
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train_scaled, y_train)

y_prediction_perceptron = perceptron.predict(X_test_scaled)
accuracy_perceptron = accuracy_score(y_test, y_prediction_perceptron)

# Typical result: ~87% accuracy
# The Perceptron struggles slightly with Iris-versicolor (the middle species)
# because it overlaps with Iris-virginica in feature space

# Classification report breakdown:
#               precision    recall  f1-score   support
#            0       0.83      1.00      0.91        10   ← Setosa (easy)
#            1       0.88      0.70      0.78        10   ← Versicolor (harder)
#            2       0.90      0.90      0.90        10   ← Virginica


# =============================================================================
# PREPARE LABELS FOR ANN — ONE-HOT ENCODING
# =============================================================================
# ANNs with Softmax output need labels in a different format called one-hot.
# Instead of a single number (0, 1, or 2), each label becomes a 3-value list:
#   0  →  [1, 0, 0]   (Setosa)
#   1  →  [0, 1, 0]   (Versicolor)
#   2  →  [0, 0, 1]   (Virginica)
#
# This matches the 3 output neurons — the neuron whose value is 1 wins.
y_train_categorical = to_categorical(y_train, num_classes=3)
y_test_categorical = to_categorical(y_test, num_classes=3)


# =============================================================================
# MODEL 2 — ANN (multi-layer neural network)
# =============================================================================
# Architecture:
#
#   Input (4 features)
#     ↓
#   Dense(16, ReLU)   — 16 neurons, learns combinations of the 4 measurements
#     ↓
#   Dense(8, ReLU)    — 8 neurons, refines the patterns found above
#     ↓
#   Dense(3, Softmax) — 3 output neurons, one per species
#                       Softmax turns raw scores into probabilities (sum = 1)
#                       The species with the highest probability wins
#
# Why Softmax at the output?
#   We have 3 classes — Softmax gives a probability for each class.
#   e.g. [0.02, 0.95, 0.03] → model is 95% sure it's Versicolor
#
# Loss: categorical_crossentropy — the right loss to use with Softmax + one-hot labels
# Optimizer: adam — fast and works well out of the box, no tuning needed
model_ann = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model_ann.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# =============================================================================
# TRAINING
# =============================================================================
# epochs=100    — go through the entire training set 100 times
# batch_size=8  — update weights after every 8 samples (mini-batch)
# validation_split=0.2 — hold back 20% of training data to monitor overfitting
#
# Watch during training:
#   loss going down     → model is learning
#   val_loss also down  → model is generalising (not just memorising)
#   val_loss going UP   → model is starting to overfit (memorise training data)
history = model_ann.fit(
    X_train_scaled, y_train_categorical,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)


# =============================================================================
# EVALUATION
# =============================================================================
# Evaluate on the test set (data the model has never seen during training)
loss, accuracy = model_ann.evaluate(X_test_scaled, y_test_categorical, verbose=0)
print(f"ANN Test Accuracy: {accuracy:.4f}")


# =============================================================================
# VISUALISE TRAINING — ACCURACY OVER EPOCHS
# =============================================================================
# Plots training accuracy vs validation accuracy across all 100 epochs.
# If both lines go up together  → model is genuinely learning
# If train keeps going up but val flatlines or drops → overfitting
#
# Saved graph: images/ann_training_validation_accuracy.png
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ANN Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()