# Artificial Neural Network (ANN) — Notes

---

## 1. Basic Structure of an ANN

An ANN is just a bunch of connected neurons arranged in layers. Think of it like an assembly line — data goes in one end, gets processed in the middle, and an answer comes out the other end.

```
Input Layer  →  Hidden Layer(s)  →  Output Layer
  (X1, X2, X3)    (does the work)       (Y = 0 or 1)
```

- **Input Layer** — where you feed your data (age, salary, temperature — whatever your features are)
- **Hidden Layer** — where the actual learning happens (math is applied here)
- **Output Layer** — spits out the final answer

### Example we will use throughout these notes

| Feature           | Value |
| ----------------- | ----- |
| X1                | 2.0   |
| X2                | 1.0   |
| X3                | 3.0   |
| Actual Y (target) | 1     |

---

## 2. Forward Propagation

Forward propagation means: **send your input through the network and get a prediction out.**

Nothing is being learned here yet — we are just calculating.

### Step 1 — Weighted Sum

Each input gets multiplied by a weight. Everything gets added up along with a bias.

```
h1 = (X1 × w1) + (X2 × w2) + (X3 × w3) + bias
```

Plugging in our example (weights: 0.6, 0.2, 0.1 | bias: 0.1):

```
h1 = (2.0 × 0.6) + (1.0 × 0.2) + (3.0 × 0.1) + 0.1
h1 = 1.2 + 0.2 + 0.3 + 0.1
h1 = 1.6
```

### Step 2 — Activation Function (Sigmoid)

We pass that raw number through the **Sigmoid function** to squish it into a value between 0 and 1.

```
sigmoid(x) = 1 / (1 + e^(-x))

sigmoid(1.6) ≈ 0.8320
```

So the hidden neuron outputs **0.8320**. (Same calculation happens for h2 → 0.8176)

### Step 3 — Final Output

The output from the hidden layer now flows into the output neuron (same process):

```
output = (0.8320 × 1.2) + (0.8176 × 0.7) + 0.05
output ≈ 0.4761

sigmoid(0.4761) ≈ 0.6168
```

**Final prediction = 0.6168**

---

## 3. Decision Rule

Once we have the output, we convert it to a yes/no answer:

```
If output >= 0.5  →  predict 1  (Yes / Positive)
If output <  0.5  →  predict 0  (No  / Negative)
```

Our output was 0.6168 → **predict 1** — which matches the actual Y = 1. Correct!

---

## 4. Loss Function — How Wrong Were We?

Even when we get the right class, we need to measure _how confident_ we were and how far off the raw number was from perfect. This is what the **Loss Function** does.

We use **Binary Cross-Entropy Loss** (also called Log Loss):

```
Loss = -[ y × log(ŷ) + (1 - y) × log(1 - ŷ) ]
```

Where:

- `y` = actual value (1 in our example)
- `ŷ` = predicted value (0.6168)

Plugging in:

```
Loss = -[ 1 × log(0.6168) + 0 × log(0.3832) ]
Loss = -log(0.6168)
Loss ≈ 0.494
```

**The goal is always: make this loss number as small as possible.**

---

## 5. Backward Propagation — How the Network Learns

Forward pass told us _what we predicted_. Loss told us _how wrong we were_. Now **backward propagation goes back through the network and adjusts the weights** to reduce that error.

### Weight Update Formula

```
new_weight = old_weight - (learning_rate × gradient)

w = w - η × (∂L/∂w)
```

- `η` (eta) = **learning rate** — how big of a correction step to take (usually something small like 0.01)
- `∂L/∂w` = **gradient** — tells us in which direction and how much to adjust the weight

### Chain Rule

To calculate the gradient for a weight deep inside the network, we use the **chain rule** from calculus. We trace the effect of that weight all the way to the final loss:

```
∂L/∂w = (∂L/∂output) × (∂output/∂hidden) × (∂hidden/∂w)
```

Think of it as asking: "If I tweak this one weight by a tiny bit, how does the final loss change?" — we trace each small effect backwards through every layer.

---

## 6. One Full Training Cycle

```
Step 1: Forward Pass  →  get a prediction
Step 2: Calculate Loss  →  how wrong were we?
Step 3: Backward Pass  →  figure out how to fix the weights
Step 4: Update Weights  →  apply the corrections
```

One full cycle like this = **one Epoch**.

You repeat this for hundreds or thousands of epochs until the loss stops going down.

### Loss Going Down Over Training

```
Epoch 1:  Loss = 0.494
Epoch 2:  Loss = 0.432
Epoch 3:  Loss = 0.349
...
Epoch N:  Loss ≈ 0.01  ← model has learned
```

---

## 7. Activation Functions

### Sigmoid

```
f(x) = 1 / (1 + e^(-x))
```

Graph shape (S-curve):

```
output
  1 |              __________
    |            /
 0.5|          /
    |        /
  0 |_______/
    |_________________________ input
         -5     0     +5
```

- Output range: **(0, 1)**
- Good for: **binary classification** (yes/no predictions at the output layer)
- Problem: causes **vanishing gradient** in deep networks (explained below)

---

### Linear Activation

```
f(x) = x
```

Just passes the value through unchanged — no squishing, no transformation. Used in output layers when predicting **continuous values** like house prices or temperatures.

---

## 8. Vanishing Gradient Problem (VGP)

This is one of the most important problems to understand in deep learning.

### What happens?

During backpropagation, gradients get multiplied together as we move back through the layers. Sigmoid squishes everything between 0 and 1, so its gradients are always small numbers (like 0.2, 0.1).

When you multiply many small numbers together layer after layer:

```
0.2 × 0.2 × 0.2 × 0.2 = 0.0016
```

By the time we reach the first layers of the network, the gradient becomes **almost zero** — so those layers update their weights by almost nothing. They stop learning.

### Analogy — Chinese Whispers

Imagine whispering a message through 10 people in a line. By the time it reaches the last person, the message is barely recognizable. That is exactly what happens to the gradient signal — it gets weaker and weaker as it travels back through the layers.

### Why it matters

- Early layers are supposed to learn basic patterns
- If they don't update → the whole network learns poorly
- **Solution:** use **ReLU** (Rectified Linear Unit) instead of Sigmoid in hidden layers — it doesn't have this problem

---

## 9. Key Concepts — Quick Reference

| Concept             | What it does                                                                            |
| ------------------- | --------------------------------------------------------------------------------------- |
| Weights             | Control how much each input influences the output                                       |
| Bias                | Shifts the output — stops the model from being stuck at zero                            |
| Activation Function | Adds non-linearity so the network can learn complex patterns                            |
| Loss Function       | Measures how wrong the prediction was                                                   |
| Gradient Descent    | Method to reduce loss by adjusting weights in the right direction                       |
| Learning Rate (η)   | Controls how big each correction step is                                                |
| Epoch               | One full forward + backward pass through all training data                              |
| Vanishing Gradient  | When gradients become too small in deep networks, causing early layers to stop learning |

---

## 10. Summary

1. **Data goes in** through the input layer
2. **Weighted sums + activation functions** run and produce a prediction (Forward Propagation)
3. **Loss is calculated** — how far off was the prediction?
4. **Gradients are computed** using the chain rule and weights are corrected (Backward Propagation)
5. Repeat until loss is minimal → network has learned

The whole thing boils down to: **make a guess → see how wrong you are → correct yourself → repeat.**

---

## Coming Next

- [x] ReLU and other activation functions — see `activation_functions.md`
- [ ] Dropout (how to prevent the model from memorizing instead of learning)
- [ ] Different optimizers — Adam, RMSProp vs plain Gradient Descent
- [ ] Building a full ANN in Python
