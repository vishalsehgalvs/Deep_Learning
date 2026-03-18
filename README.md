# Deep Learning — My Learning Notes

This repo is where I'm documenting everything I learn about Deep Learning — in plain English, no fancy jargon. Think of it as notes written by someone who is learning, for someone who is learning.

---

## What's in here

| Folder                       | What it covers      |
| ---------------------------- | ------------------- |
| `Artificial_Neural_Network/` | ANN basics and code |

---

## The Basics — Starting with a Perceptron

### What even is a Perceptron?

Think of a perceptron as a single brain cell (neuron). It takes some numbers as input, does a simple calculation, and spits out an answer.

That's it. Super simple.

### A Real Example

Say we have patient data and we want to predict if someone has heart disease or not.

| Age | Cholesterol | Blood Pressure | Has Heart Disease? |
| --- | ----------- | -------------- | ------------------ |
| 28  | 150         | 110            | Yes (1)            |
| 36  | 120         | 90             | No (0)             |
| 42  | 180         | 160            | Yes (1)            |

- **Inputs** — Age, Cholesterol, Blood Pressure (these are the clues we give the model)
- **Output** — 0 or 1 (No disease or Yes disease)

---

## How a Perceptron Actually Works

### Step 1 — Multiply inputs by weights

Each input gets multiplied by a number called a **weight**. The weight decides how important that input is.

```
result = (Age × weight1) + (Cholesterol × weight2) + (BP × weight3) + bias
```

- **Weights** — how much importance to give each input
- **Bias** — a small extra number added so the model isn't stuck at zero

In math notation this looks like:

```
h(x) = x1*w1 + x2*w2 + x3*w3 + bias
```

This whole step — feeding inputs in and getting a result out — is called **Forward Propagation**.

### Step 2 — Pass through an Activation Function

The raw number we get from Step 1 could be anything — a huge positive number, a negative number, anything. We need to squeeze that into something useful.

That's what an **activation function** does.

**Sigmoid** is a common one. It takes any number and converts it to a value between 0 and 1. Perfect for yes/no predictions.

```
If result is very high  → sigmoid gives ~1 (Yes, heart disease)
If result is very low   → sigmoid gives ~0 (No heart disease)
```

It looks like an S-curve on a graph.

---

## How the Model Learns — Loss & Gradient Descent

When the model makes a prediction, it's probably wrong at first. That's fine. Here's how it fixes itself:

1. **Loss Function** — measures how wrong the prediction was. Higher the loss, worse the prediction.
2. **Gradient Descent** — a technique that nudges the weights in the right direction to reduce the loss.
3. Repeat many times → model gets better.

The goal is always: **make the loss as small as possible.**

---

## From Perceptron to Neural Network

A single perceptron can only do so much. It's like having just one brain cell.

When you stack multiple perceptrons in layers, you get an **Artificial Neural Network (ANN)**.

```
Input Layer  →  Hidden Layer(s)  →  Output Layer
```

- **Input layer** — receives the raw data
- **Hidden layers** — does the heavy lifting, finds patterns
- **Output layer** — gives the final answer

The more hidden layers, the deeper the network — that's why it's called **Deep Learning**.

---

## Limitation of a Basic Perceptron

- Uses a simple step function — output is either 0 or 1, nothing in between
- Can't predict continuous values like house prices or temperatures
- Only works well for very simple, linearly separable problems

That's why we move to full ANNs with better activation functions.

---

## Progress

- [x] Perceptron basics
- [x] Forward propagation
- [x] Activation functions (Sigmoid)
- [x] Loss function & Gradient Descent
- [x] ANN concept
- [ ] Backpropagation (coming next)
- [ ] ANN code implementation

🔹 Quick Summary

Perceptron = basic neuron

Works using weighted sum + bias

Uses activation function

Learns using gradient descent

Limitation → only binary output

🧠 Deep Learning – Basic Notes (with simple diagrams)
🔹 What is a Perceptron?

A Perceptron is the simplest type of neuron

It takes inputs → processes them → gives output

🔹 Example Dataset
Age Cholesterol BP Heart Disease
28 150 110 1
36 120 90 0
42 180 160 1

👉 Inputs = Age, Cholesterol, BP
👉 Output = 0 or 1

🔹 Perceptron Structure (Dot Diagram)
x1 (Age) --------\
 \
 x2 (Chol) ------- ( • ) -----> Output (0/1)
/
x3 (BP) ---------/

                 + bias

👉 Each line has a weight (w1, w2, w3)

🔹 Mathematical Idea

Simple line:

y = mx + b

For multiple inputs:

h(x) = θ0 + θ1x1 + θ2x2 + θ3x3

👉 θ = weights
👉 θ0 = bias

🔹 How It Works (Step-by-step)
Inputs → Multiply by weights → Add bias → Output
x1*w1 \
 \
x2*w2 ----> SUM ----> Output
/
x3\*w3 /

        + bias

🔹 Forward Propagation
h(x) = x1*w1 + x2*w2 + x3\*w3 + bias

👉 Just calculating output → nothing fancy

🔹 Activation Function
Sigmoid (S-shape)
Output
1 | ----
| /
| /
| /
0 |\_/****\_\_****
Input

👉 Converts value → between 0 and 1

🔹 Loss Function + Gradient Descent
Loss Curve (error)
Error
|
| \ /
| \ /
| \ /
| \/
|****\_\_****\_**\_\_**
min

👉 Goal = reach the lowest point (minimum error)

🔹 Neural Network (ANN)
Input Layer Hidden Layer Output

x1 ----\
 ( • ) ----\
 x2 ----/ ( • ) ----> Output
( • ) ----/
x3 ----/

👉 Adding hidden layers → makes it powerful

🔹 Limitation of Perceptron
Output
1 |---------
|
|
0 |****\_****
Input

Only gives 0 or 1

❌ Not good for continuous values

🔹 Quick Summary

Perceptron = basic neuron

Works using weights + bias

Uses activation function

Learns using gradient descent

Limitation → only binary output
