# Activation Functions — Notes

Activation functions decide what a neuron does with its input before passing the result to the next layer. Without them, a neural network — no matter how many layers it has — would just behave like a single straight line. That's useless for learning complex things.

Think of an activation function as a filter: it decides how much of the signal gets through, and in what shape.

---

## 1. Linear Activation Function

### What it does

The neuron just passes the value straight through. No transformation. Whatever goes in, comes out.

### Formula

```
f(x) = x
```

### Graph

```
output
  |         /
  |        /
  |       /
  |      /
  |     /
  |____/____________ input
       0
```

### When to use it

- Output layer for **regression tasks** (predicting a number, not a category)
- Examples: house price, temperature, salary

### The big problem

Even if you stack 100 layers all using linear activation, the whole network still behaves like a single linear equation. All those layers collapse into one. You gain nothing from depth.

```
Linear × Linear × Linear = still just Linear
```

That's why we use non-linear functions (ReLU, Sigmoid, Tanh) in hidden layers — they're what actually give the network the ability to learn complex patterns.

---

## 2. Sigmoid Activation Function

### What it does

Takes any number — no matter how large or small — and squishes it into a value between 0 and 1. Great for yes/no outputs.

### Formula

```
sigmoid(x) = 1 / (1 + e^(-x))
```

### Graph (S-curve)

```
output
  1 |              ___________
    |            /
 0.5|          * (at x=0)
    |        /
  0 |_______/
    |__________________________ input
         -5     0     +5
```

### When to use it

- Output layer for **binary classification** (predict 0 or 1, yes or no)
- Example: will a customer churn? (yes/no)

### The big problem — Vanishing Gradient

When the input is very large or very small (far from 0), the sigmoid curve becomes almost flat. That means the gradient (the signal used to update weights) becomes nearly zero. When you multiply near-zero gradients across many layers, the early layers stop learning almost completely.

This is the **Vanishing Gradient Problem** — and it's why sigmoid is avoided in hidden layers of deep networks.

---

## 3. Tanh Activation Function

### What it does

Similar to sigmoid but the output range is **-1 to +1** instead of 0 to 1. This means it's zero-centered — positive inputs push toward +1, negative inputs push toward -1.

### Formula

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

### Graph

```
output
 +1 |            ___________
    |          /
  0 |--------*-------------- input
    |       /
 -1 |______/
    |__________________________ input
         -5     0     +5
```

### Why it's better than sigmoid

- Zero-centered output means the gradients don't all push in the same direction — learning is more balanced and faster
- Works well when data has both positive and negative values

### Still has a problem

Still suffers from the vanishing gradient problem for very large or very small inputs (the curve still flattens out at the ends).

---

## 4. ReLU — Rectified Linear Unit

### What it does

If the input is negative → output is 0. If the input is positive → output is the same number. Dead simple.

### Formula

```
f(x) = max(0, x)
```

### Graph

```
output
  |              /
  |             /
  |            /
  |           /
  |__________/_____________ input
             0
```

### Why it's the most popular activation function

- **Fast** — just a comparison, no exponential math
- **Avoids vanishing gradient** — for positive inputs the gradient is always 1, so the signal flows cleanly through
- Works brilliantly in CNNs and deep networks

### The problem — Dying ReLU

If a neuron gets a negative input and outputs 0, the gradient is also 0. The weight never updates. The neuron is essentially dead — it contributes nothing forever.

This can happen when learning rates are too high or weights are initialized badly.

---

## 5. Leaky ReLU

### What it does

Fixes the dying neuron problem in ReLU. Instead of outputting exactly 0 for negative inputs, it allows a tiny slope.

### Formula

```
f(x) = x          if x > 0
f(x) = 0.01 × x   if x < 0
```

### Graph

```
output
  |              /
  |             /
  |            /
  |           /
 -|----------/-------------- input
  |         / (tiny slope 0.01)
  |        /
```

### Why it helps

Even for negative inputs, there's a small gradient (0.01) — so the neuron stays alive and can still learn. Small leak, big fix.

### Downside

The slope value (0.01) is fixed manually. You have to decide it yourself — there's no guarantee 0.01 is the best choice for your problem.

---

## 6. PReLU — Learnable Slope ReLU (Parametric ReLU)

### What it does

Same idea as Leaky ReLU, but instead of a fixed slope like 0.01, the model **learns the best slope during training** automatically.

### Formula

```
f(x) = x       if x > 0
f(x) = a × x   if x < 0
```

Where `a` is a number the model tunes on its own during training — you don't set it manually.

### Why it's better

The network figures out the ideal negative slope for each neuron on its own. No manual tuning needed.

### Downside

There are extra numbers to learn — slightly higher chance of the model over-memorising the training data, especially when the dataset is small.

---

## 7. Swish Activation Function

### What it does

A smooth, modern alternative to ReLU introduced by Google. It multiplies the input by the sigmoid of the input.

### Formula

```
f(x) = x × sigmoid(x)
     = x × (1 / (1 + e^(-x)))
```

### Graph

```
output
  |              /
  |             /
  | ___________/
  |/  (slight dip below 0 for small negatives)
 -|----------------------------- input
  |
```

Unlike ReLU which hard-cuts at 0, Swish has a smooth curve that lets tiny negative values through — this means the learning signal passes through more cleanly instead of getting cut off.

### Why it can beat ReLU

- Smooth curve → the learning signal travels through the network more cleanly
- In very deep networks, that smoothness adds up and improves training
- Used in newer architectures like EfficientNet

### Downside

Uses sigmoid internally → slightly more computation than plain ReLU. Not always worth the extra cost on simpler problems.

---

## 8. Softmax Activation Function

### What it does

Softmax is used in the **output layer when you have more than two categories** to predict. It takes a bunch of raw numbers (one per class) and converts them into **probabilities that all add up to 1**.

Think of it like a voting system: each class gets a score, and Softmax turns those scores into percentage votes so you know how confident the model is about each option.

```
Example: classifying a photo as cat, dog, or bird

Raw scores (before Softmax):   After Softmax:
  cat  →  2.0                    cat  →  70%
  dog  →  1.0         →          dog  →  26%
  bird →  0.1                    bird →   4%
                                          ↑ always adds up to 100%
```

The highest probability wins — in this case the model says "it's a cat, 70% sure".

### Formula

For each class `i` out of `n` total classes:

```
softmax(x_i) = e^(x_i) / (e^(x_1) + e^(x_2) + ... + e^(x_n))
```

`e` is just a fixed math number (roughly 2.718) — raising it to a power turns any score into a positive number, which is what we need before dividing. Don't worry about why this specific number; just know that the formula converts raw scores into probabilities that add up to 1.

### Worked example with numbers

```
Raw scores: cat=2.0, dog=1.0, bird=0.1

Step 1 — raise e to each score:
  e^2.0 = 7.39  (cat)
  e^1.0 = 2.72  (dog)
  e^0.1 = 1.11  (bird)

Step 2 — sum them all:
  total = 7.39 + 2.72 + 1.11 = 11.22

Step 3 — divide each by the total:
  cat  = 7.39 / 11.22 = 0.659  →  65.9%
  dog  = 2.72 / 11.22 = 0.242  →  24.2%
  bird = 1.11 / 11.22 = 0.099  →   9.9%

Check: 65.9 + 24.2 + 9.9 = 100%  ✓

Model says: cat (65.9% confident)
```

Notice how the highest raw score (2.0 for cat) becomes the dominant probability.
Softmax amplifies the winner — the gap between scores gets bigger after transformation.

### Graph — how raw scores become probabilities

```
Before Softmax (raw scores):       After Softmax (probabilities):

Score                              Probability
  |  *           (cat=2.0)           |  *           (cat=66%)
  |                                  |
  |     *        (dog=1.0)           |     *        (dog=24%)
  |                                  |
  |        *     (bird=0.1)          |        *     (bird=10%)
  |__________________                |__________________
     cat  dog  bird                     cat  dog  bird

Shape is similar but everything now sits between 0 and 1, summing to 1.
```

What happens when one score is much larger:

```
Scores: cat=10.0, dog=1.0, bird=0.1

After Softmax:
  cat  →  ~99.99%   ← almost all probability mass
  dog  →   ~0.008%
  bird →   ~0.002%

Softmax is winner-takes-most — a clear winner gets nearly all the probability.
```

### How Softmax works with Categorical Cross-Entropy

Softmax and Categorical Cross-Entropy always go together at the output layer:

```
Raw scores  →  Softmax  →  Probabilities  →  Categorical Cross-Entropy  →  Loss

  [2.0,              →        [0.66,              →  how far is this from
   1.0,                        0.24,                  the correct answer?
   0.1]                        0.10]
```

The correct answer is a **one-hot vector** — all zeros except a 1 for the true class:

```
True label: dog    →  one-hot: [0, 1, 0]   (cat=0, dog=1, bird=0)
Predictions:       →           [0.66, 0.24, 0.10]

Loss = -log(prediction for true class)
     = -log(0.24)
     = 1.43   ← model was only 24% sure, big loss

After training, ideally:
  Predictions = [0.02, 0.95, 0.03]
  Loss = -log(0.95) = 0.05  ← model is right and confident, tiny loss
```

The loss decreases as the model gets better at pushing probability toward the correct class.

### When to use Softmax

```
Number of output classes:          Activation to use:
  2 classes (yes/no, spam/not)  →  Sigmoid
  3+ classes (cat/dog/bird/...)  →  Softmax
```

### Code example

```python
from tensorflow import keras
from tensorflow.keras import layers

# Multi-class: classify 3 animals (cat, dog, bird)
model = keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')   # 3 output neurons, one per class
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',        # always paired with Softmax
    metrics=['accuracy']
)
```

If your labels are plain integers (0, 1, 2) instead of one-hot vectors, swap the loss:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # use this when labels are plain numbers (0, 1, 2) instead of lists
    metrics=['accuracy']
)
```

### Pros

- Gives you a proper probability distribution — easy to see how confident the model is
- All outputs sum to 1 — clean and meaningful
- Works naturally with Categorical Cross-Entropy

### Cons

- Only used at the **output layer** — never in hidden layers
- Can look overconfident — if one score is much higher than others, Softmax assigns it nearly all the probability even when the gap was caused by random weight initialisation

**What overconfidence looks like:**

```
Raw scores: cat=5.0, dog=0.1, bird=0.1

After Softmax:
  cat  →  98.7%   ← model looks very confident
  dog  →   0.7%
  bird →   0.7%

The model may not truly "know" it's a cat — the high score might just reflect
how the weights happen to be set. Softmax amplifies confidence regardless.
```

---

## Where to Use Each Activation Function

```
Input Layer  →  No activation function (just raw data)
Hidden Layers  →  ReLU (default), Leaky ReLU, PReLU, Swish
Output Layer  →  depends on your task (see table below)
```

### Output Layer — Quick Reference

| Task                                      | Activation to Use |
| ----------------------------------------- | ----------------- |
| Regression (predict a number)             | Linear            |
| Binary Classification (yes/no)            | Sigmoid           |
| Multi-class Classification (cat/dog/bird) | Softmax           |

---

## Side-by-Side Comparison

| Function   | Output Range | Vanishing Gradient?     | Dying Neurons? | Best For                        |
| ---------- | ------------ | ----------------------- | -------------- | ------------------------------- |
| Linear     | (-∞, +∞)     | No                      | No             | Regression output               |
| Sigmoid    | (0, 1)       | Yes                     | No             | Binary classification output    |
| Tanh       | (-1, +1)     | Yes (less than sigmoid) | No             | Hidden layers (older networks)  |
| ReLU       | [0, +∞)      | No                      | Yes            | Hidden layers (default choice)  |
| Leaky ReLU | (-∞, +∞)     | No                      | No             | Hidden layers (when ReLU fails) |
| PReLU      | (-∞, +∞)     | No                      | No             | Deep networks (adaptive)        |
| Swish      | (-∞, +∞)     | No                      | No             | Very deep modern networks       |
| Softmax    | (0, 1) each  | No                      | No             | Multi-class output layer        |

---

## Summary

- **Linear** — simple pass-through, only for regression outputs, useless in hidden layers
- **Sigmoid** — was the go-to for years, now mostly only used at the output for binary problems
- **Tanh** — better than sigmoid (zero-centered), but still has vanishing gradient
- **ReLU** — the workhorse of modern deep learning, fast and simple, use this by default
- **Leaky ReLU / PReLU** — when your ReLU neurons are dying, switch to these
- **Swish** — smooth, modern, worth trying in very deep networks
- **Softmax** — converts raw scores to probabilities, only used at the output layer for 3+ class problems, always paired with Categorical Cross-Entropy

**Default rule of thumb:** Use **ReLU** in all hidden layers. At the output layer: **Sigmoid** for yes/no, **Softmax** for 3+ categories, **Linear** for predicting a number.
