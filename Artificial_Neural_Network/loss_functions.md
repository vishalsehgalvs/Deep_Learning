# Loss Functions — Notes

A loss function is just a way to measure **how wrong your model's prediction is**.

Think of it like a score for mistakes — the higher the loss, the worse the model is doing. The whole point of training is to keep pushing this number down.

```
Lower loss = fewer mistakes = better model
```

There are two big categories depending on what you're trying to predict:

- **Regression** — predicting a number (price, temperature, salary)
- **Classification** — predicting a category (spam or not, cat or dog)

---

## Part 1 — Regression Loss Functions

These are used when the answer is a number, not a category.

---

### 1. MSE — Mean Squared Error

**The idea:** take the difference between what you predicted and what the actual answer was, square it, then average across all examples.

```
MSE = (1/n) × Σ (actual - predicted)²
```

**Why square it?**
- Squaring makes all errors positive (no negatives cancelling out)
- Big errors get punished much more than small ones

**Worked example:**

```
Actual = 10,  Predicted = 8
Error  = 10 - 8 = 2
Squared = 2² = 4

Actual = 10,  Predicted = 3
Error  = 10 - 3 = 7
Squared = 7² = 49   ← much bigger punishment
```

**When to use:** When you want the model to really avoid large errors.

**Downside:** One outlier (a really bad prediction) can throw the whole loss way up.

---

### 2. MAE — Mean Absolute Error

**The idea:** take the difference, make it positive (absolute value), average it. No squaring.

```
MAE = (1/n) × Σ |actual - predicted|
```

**Worked example:**

```
Actual = 10,  Predicted = 8
Error  = |10 - 8| = 2

Actual = 10,  Predicted = 3
Error  = |10 - 3| = 7   ← same scale, no extra punishment
```

**When to use:** When all errors should be treated equally, regardless of size. More robust to outliers than MSE.

**Downside:** Less sensitive to big mistakes — the model might not try hard enough to fix them.

---

### 3. Huber Loss

**The idea:** the best of both MSE and MAE. It switches behaviour depending on how big the error is.

```
If error is small  →  behaves like MSE (squares it)
If error is large  →  behaves like MAE (takes absolute value)
```

**Why this helps:**
- MSE is great for small errors but explodes with outliers
- MAE is stable but less smooth to optimise
- Huber gives you smooth behaviour for small errors AND stability for large ones

**Simple mental model:**

```
Small mistake  → square it (MSE style)
Big mistake    → just take the absolute value (MAE style)
```

**When to use:** When your data has some outliers but you still want the model to care about precision on normal cases.

---

### 4. MSLE — Mean Squared Log Error

**The idea:** instead of comparing raw values, compare their logarithms.

```
MSLE = (1/n) × Σ (log(actual + 1) - log(predicted + 1))²
```

The `+1` is there so you don't take log of zero (log(0) is undefined).

**Why logarithms?**

When values grow exponentially (like stock prices, population, viral video views), the *relative* difference matters more than the *absolute* difference.

```
Predicting 10 vs 20    → big relative error (off by 100%)
Predicting 1000 vs 1010 → tiny relative error (off by 1%)
```

MSLE treats both proportionally — it doesn't panic just because one involves bigger numbers.

**When to use:** Predicting sales, web traffic, anything that grows exponentially.

---

## Part 2 — Classification Loss Functions

These are used when the answer is a category, not a number.

---

### 5. Binary Cross-Entropy

**The idea:** used for yes/no, 0/1, true/false problems — exactly two possible answers.

```
Loss = -[ y × log(p) + (1 - y) × log(1 - p) ]
```

Where:
- `y` = actual label (0 or 1)
- `p` = predicted probability (between 0 and 1)

**How to read this:**

```
Confident and correct   → loss is very low  ✓
Confident and wrong     → loss is very high  ✗
Unsure (around 0.5)     → moderate loss
```

**Worked example:**

```
Actual = 1 (yes, spam)

Predicted = 0.9  →  log(0.9) ≈ -0.10  →  Low loss   ← good prediction
Predicted = 0.1  →  log(0.1) ≈ -2.30  →  High loss  ← terrible prediction
```

The model gets heavily punished when it is very confident but completely wrong.

**When to use:** Any binary prediction — disease yes/no, fraud yes/no, email spam yes/no.

---

### 6. Categorical Cross-Entropy

**The idea:** same concept as binary cross-entropy, but for more than 2 classes.

```
Loss = - Σ y_i × log(p_i)
```

Where:
- `y_i` = 1 for the correct class, 0 for everything else
- `p_i` = predicted probability for each class

**The model outputs a probability for every class. Only the probability assigned to the correct class affects the loss.**

**Worked example:**

```
Classes: Cat, Dog, Car
Actual answer: Dog → one-hot = [0, 1, 0]

Prediction A: [0.1, 0.8, 0.1]  → high probability on Dog → low loss  ✓
Prediction B: [0.6, 0.1, 0.3]  → low probability on Dog  → high loss ✗
```

**When to use:** Any multi-class problem — image classification, sentiment (positive/neutral/negative), etc.

---

## Quick Reference Table

| Type | Loss Function | When to use |
|------|--------------|-------------|
| Regression | MSE | Punish big errors heavily |
| Regression | MAE | Treat all errors equally, robust to outliers |
| Regression | Huber | Balanced — handles both normal errors and outliers |
| Regression | MSLE | Data that grows exponentially, relative error matters |
| Classification | Binary Cross-Entropy | Two classes (yes/no, 0/1) |
| Classification | Categorical Cross-Entropy | Three or more classes |

---

## The Simple Way to Remember This

```
Predicting a NUMBER?   → Regression loss   → MSE / MAE / Huber / MSLE
Predicting a CATEGORY? → Classification loss → Binary or Categorical Cross-Entropy
```

And within regression:
```
Normal data, care about big mistakes?  → MSE
Outliers in data?                      → MAE or Huber
Values grow exponentially?             → MSLE
```

---

## Coming Next

- [ ] Optimizers — how the model actually uses the loss to update weights
- [ ] Overfitting and regularization techniques
