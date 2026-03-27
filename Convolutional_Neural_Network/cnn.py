# ── What this project does ───────────────────────────────────────────────────
# We train two models to recognise handwritten digits (0–9) from 28×28 images.
# Model 1: Perceptron — the simplest possible neural network (our "dumb baseline")
# Model 2: CNN        — a smarter network that actually "looks" at the image shape
# At the end we compare how accurate each model is.
# ─────────────────────────────────────────────────────────────────────────────

import os                        # create folders and build file paths
import numpy as np              # fast math on large arrays of numbers
import pandas as pd             # read CSV files and work with table data
import seaborn as sns           # makes pretty statistical charts (heatmaps etc.)
import matplotlib.pyplot as plt  # general purpose plotting library
import warnings

warnings.filterwarnings('ignore')  # suppress non-critical TensorFlow/numpy messages
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split  # splits data into train and validation portions

from sklearn.linear_model import Perceptron  # Used for simple linear classification tasks.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# accuracy_score      → overall % of correct guesses
# classification_report → per-digit precision, recall and F1
# confusion_matrix    → table showing which digits get mixed up with which

from tensorflow.keras.models import Sequential  # Sequential lets you build a neural network layer-by-layer in Keras.

from tensorflow.keras.layers import Dense       # Dense makes the final predictions
from tensorflow.keras.layers import Conv2D      # Conv2D extracts features — scans small patches of the image
from tensorflow.keras.layers import Flatten    # Flatten reshapes them — turns a 2D grid into a flat list

from tensorflow.keras.layers import MaxPooling2D  # MaxPooling2D reduces size — keeps only the strongest signal per area
from tensorflow.keras.layers import Dropout       # Dropout prevents overfitting — randomly switches off neurons during training

from tensorflow.keras.utils import \
    to_categorical  # converts numeric class labels into one-hot encoded format: e.g. 3 → [0,0,0,1,0,0,0,0,0,0]
from keras.datasets import mnist  # big data set

# ── Output folder ────────────────────────────────────────────────────────────
# All generated plots are saved here so they can be referenced from notes.md.
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)  # create the folder if it doesn't already exist

# ── Load data ────────────────────────────────────────────────────────────────
# data/train.csv → 42,000 labelled images (label + 784 pixel columns)
# data/test.csv  → 28,000 images with NO labels (Kaggle submission format)
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
# print(df_train.head())
#    label  pixel0  pixel1  pixel2  ...  pixel780  pixel781  pixel782  pixel783
# 0      1       0       0       0  ...         0         0         0         0
# 1      0       0       0       0  ...         0         0         0         0
# 2      1       0       0       0  ...         0         0         0         0
# 3      4       0       0       0  ...         0         0         0         0
# 4      0       0       0       0  ...         0         0         0         0
#
# [5 rows x 785 columns]

# print(df_train.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 42000 entries, 0 to 41999
# Columns: 785 entries, label to pixel783
# dtypes: int64(785)
# memory usage: 251.5 MB

# print(df_test.isnull())
#        pixel0  pixel1  pixel2  pixel3  ...  pixel780  pixel781  pixel782  pixel783
# 0       False   False   False   False  ...     False     False     False     False
# 1       False   False   False   False  ...     False     False     False     False
# 2       False   False   False   False  ...     False     False     False     False
# 3       False   False   False   False  ...     False     False     False     False
# 4       False   False   False   False  ...     False     False     False     False
# ...       ...     ...     ...     ...  ...       ...       ...       ...       ...
# 27995   False   False   False   False  ...     False     False     False     False
# 27996   False   False   False   False  ...     False     False     False     False
# 27997   False   False   False   False  ...     False     False     False     False
# 27998   False   False   False   False  ...     False     False     False     False
# 27999   False   False   False   False  ...     False     False     False     False
#
# [28000 rows x 784 columns]

# print(df_test.isnull().sum())
# pixel0      0
# pixel1      0
# pixel2      0
# pixel3      0
# pixel4      0
#            ..
# pixel779    0
# pixel780    0
# pixel781    0
# pixel782    0
# pixel783    0
# Length: 784, dtype: int64

# ─── Preprocessing ───────────────────────────────────────────────────────────
# Think of preprocessing as "getting the data ready before handing it to the model".
# Raw CSV numbers need to be cleaned, scaled, and shaped correctly first.

# Step 1 — Separate the answer column from the pixel columns.
# Like separating an answer key from the question sheet.
X = df_train.drop("label", axis=1).values  # pixel data  → (42000, 784)
y = df_train["label"].values               # digit labels → (42000,)

# Step 2 — Shrink pixel values from 0–255 down to 0.0–1.0.
# Neural networks learn much faster when all numbers are in a small, consistent range.
# Dividing by 255 (the maximum possible pixel brightness) does exactly that.
X = X.astype("float32") / 255.0

# Step 3 — Since test.csv has no labels, we carve 20% out of train.csv for validation.
# random_state=42 makes the split reproducible — same split every time you run the code.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 — Reshape the flat pixel list back into a proper 28×28 image grid.
# The Perceptron only needs (28, 28); -1 tells NumPy to figure out the row count itself.
X_train_img = X_train.reshape(-1, 28, 28)
X_val_img   = X_val.reshape(-1, 28, 28)

# The CNN also needs a "colour channel" dimension at the end.
# 1 = grayscale (one channel); a colour photo would be 3 (R, G, B).
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_val_cnn   = X_val.reshape(-1, 28, 28, 1)

# Step 5 — One-hot encode labels: digit 3 → [0,0,0,1,0,0,0,0,0,0]
# Keras needs this format for multi-class classification (10 classes = 10 digits).
y_train_cat = to_categorical(y_train, 10)
y_val_cat   = to_categorical(y_val,   10)


# ─── Helper: plot training history ───────────────────────────────────────────
# After training, Keras records accuracy and loss after every epoch (one full
# pass through all training images). This function draws those numbers as lines
# so you can visually check whether the model is improving or starting to memorise.
# If the train line keeps going up but the val line flattens → the model is overfitting.

def plot_history(history, model_name):
    """Draw side-by-side accuracy and loss curves for train vs. validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} – Training History", fontsize=14)

    # Left chart — Accuracy (higher = better)
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Right chart — Loss (lower = better; measures how wrong the guesses are)
    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    # Save the figure to images/ with a name based on the model
    slug = model_name.lower().replace(' ', '_')
    path = os.path.join(IMG_DIR, f"{slug}_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()  # close so the script doesn't block waiting for a window
    print(f"Saved: {path}")


# ─── Helper: evaluate & visualize ────────────────────────────────────────────
# Runs the trained model on images it has NEVER seen (validation set), then
# reports how well it did using three different lenses:
#   1. Overall accuracy   — what % of all guesses were correct
#   2. Classification report — precision/recall/F1 broken down per digit
#      - Precision: of all times it said "this is a 3", how often was it right?
#      - Recall:    of all real 3s in the data, how many did it catch?
#      - F1-score:  a single score that balances precision and recall
#   3. Confusion matrix — a 10×10 heatmap; diagonal = correct, off-diagonal = mistakes

def evaluate_model(model, X_img, y_true, model_name):
    """
    Print accuracy + classification report and display a confusion matrix.
    Works for both (28,28) and (28,28,1) shaped inputs.
    """
    # model.predict() returns a probability for each digit class.
    # argmax picks whichever digit has the highest probability — that's the model's guess.
    y_pred = np.argmax(model.predict(X_img), axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  {model_name}  –  Validation Accuracy: {acc:.4f}")
    print('='*55)

    # Per-class precision, recall, F1-score
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=[str(i) for i in range(10)]))

    # Confusion matrix — rows are the true digit, columns are what the model predicted.
    # Numbers on the main diagonal are correct guesses.
    # A dark cell OFF the diagonal reveals which digits get confused with each other (e.g. 4 vs 9).
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'{model_name} – Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    slug = model_name.lower().replace(' ', '_')
    path = os.path.join(IMG_DIR, f"{slug}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Helper: show sample predictions ────────────────────────────────────────
# A quick sanity check — show 10 real digit images side by side with the model's guess.
# Green title = the model got it right   (T and P match)
# Red title   = the model got it wrong   (T and P differ)
# T = True label (the real answer), P = Predicted label (what the model guessed)

def show_sample_predictions(model, X_img, y_true, model_name, n=10):
    """Display n sample images with their true and predicted labels."""
    y_pred = np.argmax(model.predict(X_img[:n]), axis=1)

    # CNN images have an extra channel dimension (28,28,1); drop it just for display
    img_data = X_img[:n, :, :, 0] if X_img.ndim == 4 else X_img[:n]

    fig, axes = plt.subplots(1, n, figsize=(20, 3))
    fig.suptitle(f'{model_name} – Sample Predictions', fontsize=13)
    for i, ax in enumerate(axes):
        ax.imshow(img_data[i], cmap='gray')
        color = 'green' if y_pred[i] == y_true[i] else 'red'
        ax.set_title(f'T:{y_true[i]}\nP:{y_pred[i]}', color=color, fontsize=9)
        ax.axis('off')  # hide the axis ticks — we only want to see the image
    plt.tight_layout()
    slug = model_name.lower().replace(' ', '_')
    path = os.path.join(IMG_DIR, f"{slug}_sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Model 1 : Perceptron (simple baseline) ──────────────────────────────────
# The Perceptron is the simplest possible neural network.
# It looks at ALL 784 pixel values independently and makes a direct guess.
# It has no idea which pixels are neighbours — it treats the image as a flat list.
# Think of it like guessing a digit purely by counting total brightness, ignoring shape.
# We use it as a "dumb baseline" — the CNN should easily beat it.

# A single Dense layer with softmax — no hidden layers, purely linear
perceptron = Sequential([
    Flatten(input_shape=(28, 28)),       # unroll the 28×28 grid into 784 numbers in a row
    Dense(10, activation='softmax')      # 10 output neurons (one per digit); softmax converts scores to probabilities
], name="Perceptron")

# compile = "configure how training works"
# optimizer='sgd'  → Stochastic Gradient Descent — the simplest way to adjust weights each step
# loss='categorical_crossentropy' → standard penalty function for multi-class problems
# metrics=['accuracy'] → log % correct after every epoch so we can track progress
perceptron.compile(optimizer='sgd',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print("\nTraining Perceptron...")
# fit() = actually run the training loop
# epochs=10     → go through all training images 10 times
# batch_size=32 → update weights after every 32 images, not after every single one
# validation_data → at the end of each epoch, test on data the model has NEVER trained on
history_perceptron = perceptron.fit(
    X_train_img, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_img, y_val_cat),
    verbose=1
)

# Visualise training curves
plot_history(history_perceptron, "Perceptron")

# Evaluate on validation set
evaluate_model(perceptron, X_val_img, y_val, "Perceptron")

# Show sample predictions
show_sample_predictions(perceptron, X_val_img, y_val, "Perceptron")


# ─── Model 2 : Convolutional Neural Network ──────────────────────────────────
# A CNN is built specifically for images. Instead of looking at every pixel alone,
# it slides a small 3×3 "magnifying glass" (filter) across the image to find patterns.
# Early filters pick up simple things like edges and corners.
# Later filters combine those into curves, loops, and digit-shaped parts.
# Think of it like a detective: scan for clues → zoom out → make a final decision.

# Conv layers learn spatial patterns (edges, curves); pooling reduces dimensions;
# Dropout randomly zeros neurons during training to prevent overfitting
cnn = Sequential([
    # Block 1 – learn low-level features (edges, corners)
    # Conv2D(32, ...) → apply 32 different 3×3 filters; relu ignores negative responses
    # padding='same'  → keeps the output image the same width/height as the input
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),               # shrink image: 28×28 → 14×14 by keeping the max in each 2×2 patch
    Dropout(0.25),                      # randomly turn off 25% of neurons — stops the model memorising

    # Block 2 – learn higher-level patterns (curves, digit parts)
    # 64 filters this time — more filters = can detect more complex/specific patterns
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),               # shrink again: 14×14 → 7×7
    Dropout(0.25),

    # Classifier head — take all the features found above and make a final guess
    Flatten(),                          # squash 7×7×64 feature maps into a flat list of 3136 numbers
    Dense(128, activation='relu'),      # hidden layer: combines features into digit-level decisions
    Dropout(0.5),                       # stronger dropout (50%) right before the output — reduces overfitting
    Dense(10, activation='softmax')     # 10 outputs, one per digit; softmax gives probabilities summing to 1
], name="CNN")

# Adam is smarter than SGD — it adjusts the learning rate for each weight individually,
# making training faster and more stable without manual tuning.
cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Print a table of every layer, its output shape, and how many learnable parameters it has
cnn.summary()

print("\nTraining CNN...")
# batch_size=64 → slightly larger batches than the Perceptron, speeds up GPU/CPU usage
history_cnn = cnn.fit(
    X_train_cnn, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_cnn, y_val_cat),
    verbose=1
)

# Visualise training curves
plot_history(history_cnn, "CNN")

# Evaluate on validation set
evaluate_model(cnn, X_val_cnn, y_val, "CNN")

# Show sample predictions
show_sample_predictions(cnn, X_val_cnn, y_val, "CNN")


# ─── Model comparison ────────────────────────────────────────────────────────
# Grab the validation accuracy recorded at the very last epoch of each training run
# and print them side by side so it's easy to see the improvement.

# [-1] grabs the last value in the list (accuracy after the final epoch)
perc_acc = history_perceptron.history['val_accuracy'][-1]
cnn_acc  = history_cnn.history['val_accuracy'][-1]

print(f"\n{'='*40}")
print(f"  Perceptron  Val Accuracy : {perc_acc:.4f}")
print(f"  CNN         Val Accuracy : {cnn_acc:.4f}")
print(f"  CNN improvement          : +{(cnn_acc - perc_acc):.4f}")
print('='*40)

# Bar chart — a visual at-a-glance comparison of final accuracy for both models
plt.figure(figsize=(6, 4))
plt.bar(['Perceptron', 'CNN'], [perc_acc, cnn_acc], color=['steelblue', 'darkorange'])
plt.ylim(0, 1)
plt.ylabel('Validation Accuracy')
plt.title('Model Comparison – Validation Accuracy')
for i, v in enumerate([perc_acc, cnn_acc]):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')  # label each bar with its value
plt.tight_layout()
path = os.path.join(IMG_DIR, "model_comparison_accuracy.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {path}")

print("\nAll plots saved to the images/ folder.")


