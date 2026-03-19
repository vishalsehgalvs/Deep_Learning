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

## 6. PReLU — Parametric ReLU

### What it does

Same idea as Leaky ReLU, but instead of a fixed slope like 0.01, the model **learns the best slope during training** automatically.

### Formula

```
f(x) = x       if x > 0
f(x) = α × x   if x < 0
```

Where `α` is a parameter the model learns (not set by you).

### Why it's better

The network figures out the ideal negative slope for each neuron on its own. No manual tuning needed.

### Downside

More parameters to learn → slightly higher risk of overfitting, especially on small datasets.

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

Unlike ReLU which hard-cuts at 0, Swish has a smooth curve that allows small negative values through — this makes gradients flow more smoothly.

### Why it can beat ReLU

- Smooth curve → better gradient flow through the network
- In very deep networks, that smoothness adds up and improves training
- Used in newer architectures like EfficientNet

### Downside

Uses sigmoid internally → slightly more computation than plain ReLU. Not always worth the extra cost on simpler problems.

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

---

## Summary

- **Linear** — simple pass-through, only for regression outputs, useless in hidden layers
- **Sigmoid** — was the go-to for years, now mostly only used at the output for binary problems
- **Tanh** — better than sigmoid (zero-centered), but still has vanishing gradient
- **ReLU** — the workhorse of modern deep learning, fast and simple, use this by default
- **Leaky ReLU / PReLU** — when your ReLU neurons are dying, switch to these
- **Swish** — smooth, modern, worth trying in very deep networks

**Default rule of thumb:** Use **ReLU** in all hidden layers unless you have a specific reason not to.
