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

**Graph — how loss grows as error gets bigger:**

```
loss
  |                        *
  |                     *
  |                  *
  |             *
  |       *
  |  *
  |* (small error = small loss)
  |__________________________ error size
  0     1     2     3     4
```

Notice how the curve shoots up steeply — a small increase in error causes
a big jump in loss. That's the squaring effect.

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

**Pros:**

- Forces the model to focus hard on fixing big mistakes
- Smooth curve — easy for gradient descent to work with
- Very commonly used, well understood

**Cons:**

- One single bad data point (outlier) can blow up the entire loss
- Not great when your dataset has noisy or incorrect entries

---

### 2. MAE — Mean Absolute Error

**The idea:** take the difference, make it positive (absolute value), average it. No squaring.

```
MAE = (1/n) × Σ |actual - predicted|
```

**Graph — loss grows in a straight line:**

```
loss
  |                    *
  |                 *
  |              *
  |           *
  |        *
  |     *
  |  *
  |* (zero error = zero loss)
  |__________________________ error size
  0     1     2     3     4
```

Straight line — every extra unit of error adds the same amount of loss.
No acceleration, no panic, just steady and fair.

**Worked example:**

```
Actual = 10,  Predicted = 8
Error  = |10 - 8| = 2

Actual = 10,  Predicted = 3
Error  = |10 - 3| = 7   ← same scale, no extra punishment
```

**When to use:** When all errors should be treated equally, regardless of size. More robust to outliers than MSE.

**Pros:**

- Outliers don't ruin your training — one bad data point won't dominate
- Easy to interpret — the loss value is in the same unit as your prediction
- Treats a small error and a big error proportionally

**Cons:**

- Not smooth at zero (the gradient jumps) — can make optimisation slightly harder
- Doesn't push the model hard enough to fix large mistakes

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

**Graph — smooth near zero, straight line for large errors:**

```
loss
  |                    *
  |                 *          ← straight line (MAE zone)
  |              *
  |           * *
  |       * *               ← curve (MSE zone)
  |   * *
  |**
  |__________________________ error size
  0     δ    2     3     4
              ↑
         switch point (delta)
```

Left of delta: behaves like MSE (curved).
Right of delta: behaves like MAE (straight).

**When to use:** When your data has some outliers but you still want the model to care about precision on normal cases.

**Pros:**

- Best of both worlds — smooth for small errors (like MSE), stable for big errors (like MAE)
- Works well in real-world messy data where some outliers exist
- More reliable training compared to using plain MSE

**Cons:**

- You need to set the threshold (called delta) that decides when to switch from MSE to MAE behaviour — one extra thing to tune
- Slightly more complex to understand and explain compared to MSE or MAE

---

### 4. MSLE — Mean Squared Log Error

**The idea:** instead of comparing raw values, compare their logarithms.

```
MSLE = (1/n) × Σ (log(actual + 1) - log(predicted + 1))²
```

The `+1` is there so the number inside the log is never zero — taking the log of zero has no answer (you can't do it), so adding 1 keeps it safe.

**Why logarithms?**

When values grow very fast (like stock prices, population, viral video views), the _relative_ difference matters more than the _absolute_ difference.

```
Predicting 10 vs 20    → big relative error (off by 100%)
Predicting 1000 vs 1010 → tiny relative error (off by 1%)
```

MSLE treats both proportionally — it doesn't panic just because one involves bigger numbers.

**Graph — flattens out as values grow larger:**

```
loss
  |  *
  |    *
  |      *
  |         *
  |              *
  |                    *
  |                              *
  |__________________________________ predicted value
  0   10   100  1000  10000
```

As the predicted value gets bigger, the loss grows much more slowly.
That's the log effect — it cares less about huge absolute differences
and more about proportional ones.

**When to use:** Predicting sales, web traffic, anything that grows exponentially.

**Pros:**

- Great when the target values vary a lot in scale (e.g. 10 vs 10,000)
- Cares about relative accuracy, not just absolute numbers
- Naturally handles exponential growth patterns

**Cons:**

- Only works with positive values — can't use it if predictions or actuals can be zero or negative (without extra workaround)
- Harder to interpret — the loss value is in log scale, not the original unit
- Underestimates large errors compared to MSE

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

**Graph — loss vs predicted probability (when actual = 1):**

```
loss
  |*
  | *
  |  *
  |    *
  |       *
  |            *
  |                   *
  |                              *
  |___________________________________ predicted probability
  0    0.1   0.3   0.5   0.7   0.9   1.0
  ↑                                   ↑
confident                         confident
& wrong                           & correct
(huge loss)                       (tiny loss)
```

When the model says "I'm 90% sure it's spam" and it IS spam → almost zero loss.
When the model says "I'm 10% sure it's spam" and it IS spam → very high loss.

**Worked example:**

```
Actual = 1 (yes, spam)

Predicted = 0.9  →  log(0.9) ≈ -0.10  →  Low loss   ← good prediction
Predicted = 0.1  →  log(0.1) ≈ -2.30  →  High loss  ← terrible prediction
```

The model gets heavily punished when it is very confident but completely wrong.

**When to use:** Any binary prediction — disease yes/no, fraud yes/no, email spam yes/no.

**Pros:**

- Works perfectly for yes/no problems
- Heavily punishes overconfident wrong predictions — keeps the model honest
- Probabilities are well-calibrated (0.9 really means 90% confident)

**Cons:**

- Only works for two classes — can't use it for cat/dog/car type problems
- If your dataset is heavily imbalanced (e.g. 99% not-fraud, 1% fraud), the loss can be misleading

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

**Pros:**

- Handles any number of classes cleanly
- Only cares about the correct class — clean and focused signal
- Standard choice for almost all classification tasks with 3+ categories

**Cons:**

- Like binary cross-entropy, struggles with heavily imbalanced datasets
- Assumes each prediction belongs to only one category — doesn't work if something can belong to multiple categories at the same time (e.g. a photo tagged as both "beach" and "sunset")

---

## Quick Reference Table

| Type           | Loss Function             | When to use                                           |
| -------------- | ------------------------- | ----------------------------------------------------- |
| Regression     | MSE                       | Punish big errors heavily                             |
| Regression     | MAE                       | Treat all errors equally, robust to outliers          |
| Regression     | Huber                     | Balanced — handles both normal errors and outliers    |
| Regression     | MSLE                      | Data that grows exponentially, relative error matters |
| Classification | Binary Cross-Entropy      | Two classes (yes/no, 0/1)                             |
| Classification | Categorical Cross-Entropy | Three or more classes                                 |

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
