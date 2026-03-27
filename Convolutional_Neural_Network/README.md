# Convolutional Neural Network — Handwritten Digit Classifier

A from-scratch comparison of two models — a Perceptron baseline and a CNN — on the classic handwritten digit recognition task using the Kaggle MNIST dataset.

---

## What This Project Does

Two neural networks are trained to classify handwritten digits (0–9) from 28×28 grayscale images:

- **Model 1 — Perceptron**: the simplest possible neural network. Flattens the image and makes a linear guess. Used as a "dumb baseline."
- **Model 2 — CNN**: a convolutional neural network that scans the image with filters to detect edges, shapes, and digit-level patterns. Should significantly outperform the Perceptron.

The final output is a side-by-side accuracy comparison of the two models.

---

## Dataset

Source: [Kaggle Digit Recogniser Competition](https://www.kaggle.com/competitions/digit-recognizer)

| File             | Rows   | Description                                             |
| ---------------- | ------ | ------------------------------------------------------- |
| `data/train.csv` | 42,000 | Labelled images. Columns: `label` + `pixel0`–`pixel783` |
| `data/test.csv`  | 28,000 | Unlabelled images. Columns: `pixel0`–`pixel783` only    |

Each row is one 28×28 grayscale image stored flat as 784 pixel values (0–255). The `test.csv` file has no labels — it is the Kaggle submission format.

---

## Project Structure

```
Convolutional_Neural_Network/
├── cnn.py          ← all code: preprocessing, models, training, evaluation
├── notes.md        ← full theory notes (CNN concepts, edge detection, pooling, etc.)
├── data/
│   ├── train.csv   ← 42,000 labelled training images
│   └── test.csv    ← 28,000 unlabelled test images
└── images/         ← all generated plots are saved here automatically
```

---

## Models

### Perceptron (Baseline)

```
Input (28×28)
   ↓
Flatten  →  784 numbers
   ↓
Dense(10, softmax)  →  10 class probabilities
```

| Setting    | Value                    |
| ---------- | ------------------------ |
| Optimizer  | SGD                      |
| Loss       | Categorical Crossentropy |
| Epochs     | 10                       |
| Batch size | 32                       |

### CNN

```
Input (28×28×1)
   ↓
Conv2D(32, 3×3, relu, padding=same)  →  28×28×32
MaxPooling2D(2×2)                    →  14×14×32
Dropout(0.25)
   ↓
Conv2D(64, 3×3, relu, padding=same)  →  14×14×64
MaxPooling2D(2×2)                    →   7×7×64
Dropout(0.25)
   ↓
Flatten  →  3136 numbers
Dense(128, relu)
Dropout(0.5)
Dense(10, softmax)  →  10 class probabilities
```

| Setting    | Value                    |
| ---------- | ------------------------ |
| Optimizer  | Adam                     |
| Loss       | Categorical Crossentropy |
| Epochs     | 10                       |
| Batch size | 64                       |

---

## Preprocessing

1. Separate pixel columns (`X`) from the label column (`y`)
2. Normalise pixel values from 0–255 to 0.0–1.0 (divide by 255)
3. Split: 80% training / 20% validation (`random_state=42` for reproducibility)
4. Reshape for Perceptron: `(N, 784)` → `(N, 28, 28)`  
   Reshape for CNN: `(N, 784)` → `(N, 28, 28, 1)` — the `1` is the grayscale channel
5. One-hot encode labels: e.g. digit `3` → `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

---

## Generated Outputs

All plots are saved to the `images/` folder when `cnn.py` is run:

| File                                       | Description                                           |
| ------------------------------------------ | ----------------------------------------------------- |
| `images/perceptron_training_history.png`   | Accuracy & loss over 10 epochs — Perceptron           |
| `images/perceptron_confusion_matrix.png`   | 10×10 heatmap — which digits the Perceptron confuses  |
| `images/perceptron_sample_predictions.png` | 10 sample digits with true vs predicted labels        |
| `images/cnn_training_history.png`          | Accuracy & loss over 10 epochs — CNN                  |
| `images/cnn_confusion_matrix.png`          | 10×10 heatmap — which digits the CNN confuses         |
| `images/cnn_sample_predictions.png`        | 10 sample digits with true vs predicted labels        |
| `images/model_comparison_accuracy.png`     | Bar chart comparing final validation accuracy of both |

---

## How to Run

```bash
python cnn.py
```

The script will:

1. Load `train.csv` and `test.csv`
2. Preprocess the data
3. Train the Perceptron (Model 1)
4. Train the CNN (Model 2)
5. Print accuracy, classification reports, and confusion matrices to the terminal
6. Save all plots to `images/`
7. Print a final side-by-side accuracy comparison

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow / keras
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## Notes

Full theory notes covering CNNs from first principles are in [notes.md](notes.md), including:

- Why ANNs fail on images
- How grayscale and colour images work
- Convolutional layers, filters, and feature maps
- Edge detection — worked mathematical examples
- Pooling (max, average, min)
- Padding and strides
- Flattening and fully connected layers
- Dropout and the Adam optimiser
- How each layer in this project maps to the theory
