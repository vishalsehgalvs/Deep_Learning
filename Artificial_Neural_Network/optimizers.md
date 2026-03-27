# Optimizers — Notes

When a model makes a prediction and gets it wrong, something needs to decide **how to fix the weights** to do better next time. That's what an optimizer does.

Think of it like this: the loss function tells you _how wrong_ you are, and the optimizer is the strategy you use to _fix it_.

```
Loss function  →  tells you the mistake
Optimizer      →  decides how to correct the weights
```

The goal of every optimizer is the same: **minimize the loss**.

---

## The Core Idea — Gradient Descent

Before diving into specific optimizers, the underlying idea is always **gradient descent**.

Imagine you're blindfolded on a hilly landscape and you want to reach the lowest point (minimum loss). You feel the slope under your feet and take a step downhill. That's gradient descent — you use the slope (gradient) to figure out which direction to move the weights.

```
Loss
  |  *
  |    *
  |      *
  |        *           ← taking steps downhill
  |           *
  |               *
  |                    *  ← minimum loss (goal)
  |_________________________________ weights
```

Each step = one weight update = learning rate × gradient.

---

## 1. Batch Gradient Descent

### The idea

Look at **every single row** in your dataset, compute the total loss, then update the weights once.

```
Step 1: Feed ALL data through the model
Step 2: Compute total loss across all rows
Step 3: Compute gradients
Step 4: Update weights → one update
Step 5: Repeat for next epoch
```

### How many updates per epoch?

```
Dataset = 100,000 rows
Updates per epoch = 1   ← just once, after seeing everything
100 epochs = 100 total weight updates
```

### Graph — smooth, steady path to the minimum

```
Loss
  |*
  | *
  |  *
  |   *
  |    *
  |     *
  |      *  ← clean, no noise, steady descent
  |_________________________________ epochs
```

Very clean curve — each step is perfectly calculated using all the data.

### Pros

- Very stable — the gradient is as accurate as it can possibly be
- Smooth learning curve, easy to monitor
- Converges steadily

### Cons

- **Extremely slow** for large datasets — imagine loading 1 million rows just to take one step
- Needs everything in memory at once — not feasible for real-world big data
- Not practical for large datasets — it just takes too long when you have millions of rows

**What the drawback looks like:**

```
Time taken per epoch (Batch GD vs Mini-Batch):

Batch GD:    [============================]  1 update  ← slow
Mini-Batch:  [==]  [==]  [==]  [==]  ...  1000 updates ← much faster
```

For the same number of epochs, Batch GD has seen far fewer total updates.
By the time Batch GD takes 100 steps, Mini-Batch might have taken 100,000.
Time-wise, Batch GD just can't compete on large data.

---

## 2. Stochastic Gradient Descent (SGD — random one-at-a-time updates)

### The idea

Go to the complete opposite extreme. Update the weights **after every single data point**.

```
Take 1 row → compute loss → update weights
Take next row → compute loss → update weights
... repeat for every row
```

### How many updates per epoch?

```
Dataset = 100,000 rows
Updates per epoch = 100,000  ← one update per row
```

### Formula

The weight update rule is simple:

```
new_weight = old_weight - (learning_rate × gradient)

w = w - η × ∂L/∂w
```

Where:

- `w` = the weight being updated
- `η` (eta) = learning rate — controls how big each step is (e.g. 0.01)
- `∂L/∂w` = gradient — slope of the loss at the current weight

**Example with numbers:**

```
Current weight  = 0.5
Learning rate   = 0.01
Gradient        = 2.0   (loss is going up steeply here)

New weight = 0.5 - (0.01 × 2.0)
           = 0.5 - 0.02
           = 0.48   ← small step in the right direction
```

Do this for every weight, after every single row. That's SGD.

---

### Graph — fast but very noisy path

```
Loss
  | *  *
  |  *   *  *
  |    *   *   *
  |      *   *   *
  |            *   *
  |              *   *  ← bounces around but trends downward
  |_________________________________ updates
```

It gets to the bottom area fast but bounces around wildly because each single row might be noisy or unusual.

---

### Why is it so noisy? — The Noise Problem Explained

When you update weights based on just ONE row, that one row might be unusual or an outlier. The gradient calculated from it might be pointing in the wrong direction.

```
Row 1  →  gradient says "go left"   → update
Row 2  →  gradient says "go right"  → update (opposite!)
Row 3  →  gradient says "go left"   → update
...
```

This is what the path looks like compared to Batch GD:

```
Batch GD (smooth):          SGD (noisy):

Loss                        Loss
  |*                          |*  *   *
  | *                         |  *  *   *   *
  |  *                        |        *  *
  |   *                       |          *   *
  |    *                      |            *  *
  |     *  ← clean            |              *  * ← chaotic but gets there
  |_______ weights            |_________________ weights
```

SGD is like stumbling downhill in the dark — you're falling in the general right direction but it's messy.

Despite the messiness, this randomness is sometimes **actually useful**. It can shake the model out of bad flat spots or local dips where a smooth optimizer would get stuck.

---

### Learning Rate — the most important thing to tune

With SGD, the learning rate matters a lot:

```
Too high:                     Too low:
Loss                          Loss
  |  *                          |*
  |     *                       | *
  |  *     *                    |  *
  |    *  *  *  ← overshoots    |   *
  |  *   *   *    and diverges  |    *
  |________________________________   |    *  *  *  ← takes forever
                                      |________________
```

Common starting value: `learning_rate = 0.01`

---

### Library and Code

**In Keras / TensorFlow:**

```python
from tensorflow import keras
from tensorflow.keras import layers

# Build model
model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Use SGD optimizer
# lr = learning rate
optimizer = keras.optimizers.SGD(learning_rate=0.01)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, batch_size=1)  # batch_size=1 → true SGD
```

**In scikit-learn (for simpler models):**

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01)
model.fit(X_train, y_train)
```

**SGD with learning rate decay (gets smaller over time — good practice):**

```python
# Learning rate starts high (fast learning) then slows down (fine-tuning)
optimizer = keras.optimizers.SGD(learning_rate=0.1, decay=1e-4)
```

---

### Pros

- Very fast — weights update constantly, learning happens quickly
- Can handle massive datasets — doesn't need to load everything at once
- The randomness can actually help escape bad spots (local minima)

### Cons

- Noisy — one weird data point can send the weights in the wrong direction
- The loss curve looks chaotic, hard to tell if it's actually improving
- Can overshoot the minimum and miss it
- Learning rate is very sensitive — hard to tune

**What overshooting looks like:**

```
Loss
  |                ← minimum is here
  |        *   *
  |      *   *   *   ← keeps jumping over the minimum
  |    *         *
  |  *             *
  |*                 *  ← diverging, getting worse!
  |_________________________________ updates
```

If the learning rate is too high, SGD keeps jumping right over the
bottom of the valley and never settles. It's like running downhill
so fast you can't stop at the bottom — you fly right past it.

---

## 3. Mini-Batch Gradient Descent

### The idea

The sweet spot between Batch GD and SGD. Instead of using all the data or just one row, use **small chunks (batches)** at a time.

```
Dataset = 100,000 rows
Batch size = 100

Updates per epoch = 100,000 / 100 = 1,000
```

This is what almost every real model uses in practice.

### Graph — fast but much smoother than SGD

```
Loss
  |*
  | *  *
  |   *  *
  |      *  *
  |         *  *
  |            *  *
  |               *  ← mostly downward, minor wobbles
  |_________________________________ updates
```

Not as smooth as Batch GD, not as chaotic as SGD — right in the middle.

### Pros

- Faster than Batch GD — updates happen every 100 rows, not 100,000
- More stable than SGD — one weird row can't derail the whole update
- Memory efficient — only load a small batch at a time
- **This is the default for almost all deep learning models**

### Cons

- Still has some noise (though much less than SGD)
- You need to choose a batch size (common choices: 32, 64, 128) — wrong choice can slow things down

**What the wrong batch size looks like:**

```
Batch size too small (like 4):     Batch size too large (like 10,000):

Loss                               Loss
  |*  *  *  *                        |*
  |  *  * *  *  ← noisy,             | *
  |    *   *      acts like SGD      |  *
  |      *   * *                     |   *   ← very slow,
  |____________                      |    *     few updates
                                     |_________
```

Small batch → lots of noise (like SGD).
Large batch → fewer updates, slower learning (like Batch GD).
Sweet spot is usually 32 or 64.

---

## 4. Momentum

### The idea

Regular SGD takes small, independent steps. Momentum adds **memory of past steps** so the model builds up speed in the right direction — like a ball rolling downhill gaining momentum.

```
Regular SGD:    small step, small step, small step...
With Momentum:  small step → gains speed → bigger steps in same direction
```

### Formula

```
velocity = β × previous_velocity + (1 - β) × gradient
weight   = weight - learning_rate × velocity
```

`β` (beta) is usually 0.9 — it says "remember 90% of the previous direction".

### Graph — cuts through oscillations

```
Without Momentum (SGD):        With Momentum:
Loss                           Loss
  |  * * * * * *                 |*
  |* * * * * * *                 | *
  |  *  * * * *                  |  *
  |    * * * *                   |   *
  |________ weights              |    *  ← smoother, faster
                                 |________ weights
```

### Pros

- Faster convergence — builds up speed in consistent directions
- Reduces the zig-zagging that SGD suffers from
- Helps push through flat regions where gradients are tiny

### Cons

- Can overshoot the minimum if momentum is too high
- One extra parameter (β) to tune

**What overshooting with momentum looks like:**

```
β = 0.9 (good):           β = 0.99 (too much momentum):

Loss                      Loss
  |*                        |*
  | *                       | *
  |  *                      |  *
  |   *                     |   *
  |    * *  ← settles       |       *    *   ← swings past
  |________ weights         |    *     *   *   the bottom
                            |  *         *   *  keeps going
                            |_____________
```

Too much momentum is like a heavy ball rolling downhill — it builds up
so much speed it rolls right past the lowest point and up the other side.
It eventually settles but wastes time oscillating.

---

## 5. RMSProp

### The idea

Instead of using the same learning rate for every weight, **adapt the learning rate for each weight individually**.

- Weights that have been getting large gradients → slow them down (smaller steps)
- Weights that have been getting small gradients → speed them up (larger steps)

Think of it like traffic: if one lane is moving fast, slow it down. If another lane is nearly stopped, give it a boost.

### Formula

```
running_total  = β × running_total + (1 - β) × gradient²
weight = weight - (learning_rate / √running_total) × gradient
```

Big gradient → running_total gets big → learning rate shrinks → smaller step
Small gradient → running_total stays small → learning rate stays bigger → larger step

The `running_total` is just a memory of how large the gradients have been recently for each weight. Big past gradients → slow down. Small past gradients → keep going faster.

### Graph — adapts to the terrain

```
Loss
  |*
  |  *
  |    *
  |      *
  |        *   ← steady, adjusts automatically based on each weight
  |          *
  |            *
  |_________________________________ updates
```

### Pros

- Adapts automatically — no need to manually slow down or speed up different weights
- Works well for complex problems and data that changes over time
- Good for recurrent networks (RNNs)

### Cons

- Can sometimes stop progressing too early (especially with very large or very small data)
- Still needs a learning rate to start with

**What early stopping (dying) looks like:**

```
Loss
  |*
  | *
  |  *
  |   *
  |    * * * * * * * * *  ← stuck, barely moving
  |                         (running_total has grown too large,
  |                          step size has shrunk to almost nothing)
  |_________________________________ updates
```

Over time the running_total keeps growing and never shrinks. This makes the step size shrink towards zero — the model
slows down and can get stuck before reaching the true minimum.
This is called the **dying learning rate** problem.

---

## 6. Adam — Adaptive Moment Estimation (the go-to optimizer for most people)

### The idea

Adam combines the best of **Momentum** and **RMSProp**.

- From Momentum: remembers which direction the weights have been moving
- From RMSProp: adjusts the step size per weight based on how large the recent gradients have been

The name "Adaptive Moment Estimation" is just the technical way of saying it adapts its step sizes based on a memory of past gradients. Ignore the name — just know it does both things at once.

This makes it the most powerful and the most widely used optimizer in deep learning.

Think of it like a GPS that not only knows the current slope (gradient) but also remembers where it has been going (momentum) AND adjusts its speed based on how bumpy the road has been (RMSProp). That combination makes it both fast AND smart.

```
Momentum alone:   remembers direction  →  fast but fixed step size
RMSProp alone:    adapts step size     →  smart but can forget direction
Adam:             does BOTH            →  fast + smart = best of everything
```

### Formula

```
Step 1 — track the direction (momentum part):
  m = β1 × m + (1 - β1) × gradient

Step 2 — track the size of gradients (RMSProp part):
  v = β2 × v + (1 - β2) × gradient²

Step 3 — bias correction (fixes the cold start problem):
  m_hat = m / (1 - β1^t)
  v_hat = v / (1 - β2^t)

Step 4 — update the weight:
  weight = weight - (learning_rate / (√v_hat + ε)) × m_hat
```

Plain English version:

- `m` = running average of which direction gradients point
- `v` = running average of how big the gradients are
- `m_hat` and `v_hat` = corrected versions of m and v (the bias correction stops Adam from taking tiny steps at the very start)
- `ε` (epsilon) = tiny number like 1e-8 to avoid dividing by zero

Default values that work well for most problems:

- `β1 = 0.9` (momentum memory — remember 90% of past direction)
- `β2 = 0.999` (gradient size memory — remember 99.9% of past sizes)
- `learning_rate = 0.001`
- `ε = 1e-8`

### Why bias correction? — The cold start explained

At the very beginning (t=1), m and v start at zero. Without correction, the first few updates are too small because the averages haven't warmed up yet.

```
Without bias correction:
  Step 1: m ≈ 0.1  ,  v ≈ 0.001   ← both are tiny (just started)
  Update is almost zero even though gradient is meaningful
  Model barely moves at the start

With bias correction:
  m_hat = m / (1 - 0.9^1) = m / 0.1  →  scales it up to proper size
  v_hat = v / (1 - 0.999^1) = v / 0.001  →  scales it up too
  First few steps are now correct size  ← model learns right away
```

This correction only matters at the start. After a while, β1^t and β2^t get so close to zero that the correction becomes 1 and disappears naturally.

### Worked example with numbers

Let's trace through one Adam update:

```
Current weight  = 0.5
learning_rate   = 0.001
β1 = 0.9,  β2 = 0.999,  ε = 1e-8
Current gradient = 0.3

--- Previous values (from last step) ---
m (old) = 0.0  (first step, starts at zero)
v (old) = 0.0  (first step, starts at zero)
t = 1   (first time step)

--- Step 1: Update m (momentum) ---
m = 0.9 × 0.0 + (1 - 0.9) × 0.3
  = 0.0 + 0.1 × 0.3
  = 0.03

--- Step 2: Update v (RMSProp) ---
v = 0.999 × 0.0 + (1 - 0.999) × 0.3²
  = 0.0 + 0.001 × 0.09
  = 0.00009

--- Step 3: Bias correction ---
m_hat = 0.03 / (1 - 0.9^1) = 0.03 / 0.1 = 0.3
v_hat = 0.00009 / (1 - 0.999^1) = 0.00009 / 0.001 = 0.09

--- Step 4: Update weight ---
weight = 0.5 - (0.001 / (√0.09 + 1e-8)) × 0.3
       = 0.5 - (0.001 / 0.3) × 0.3
       = 0.5 - 0.001
       = 0.499   ← tiny step in the right direction
```

The weight moved by exactly the learning rate (0.001) in this case.
As training continues, m and v accumulate history and the steps become smarter.

### Graph — fast, smooth, reliable

```
Loss
  |*
  | *
  |  *
  |   *
  |    *   ← very fast drop, then fine-tunes to minimum
  |     *
  |      * * *  ← settles smoothly
  |_________________________________ updates
```

Compare with SGD:

```
SGD (noisy):                 Adam (smooth):

Loss                         Loss
  |*  *   *                    |*
  |   *  *   *                 | *
  |     *  *   *               |  *
  |         *   *              |   *
  |            *   *           |    *
  |              *   *         |     * * *  ← settles cleanly
  |__________________          |________________
```

### Library and Code

**In Keras / TensorFlow (default — just one line):**

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Adam with all defaults — this is what 90% of models use
model.compile(
    optimizer='adam',                  # simplest way
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, batch_size=32)
```

**With custom settings:**

```python
# Fine-tune Adam if needed (rarely necessary)
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,   # default, usually don't change
    beta_1=0.9,            # momentum memory
    beta_2=0.999,          # gradient size memory
    epsilon=1e-8           # avoid divide by zero
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

**For our plant watering predictor (replacing SGD with Adam):**

```python
# Original used SGD:
# optimizer = keras.optimizers.SGD(learning_rate=0.01)

# Switch to Adam — usually better performance:
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Pros

- Fast convergence — learns quickly from the start
- Works out of the box on most problems with default settings
- Less tuning needed compared to plain SGD or Momentum
- Bias correction makes the first few steps correct (no cold start waste)
- **Default choice in almost all modern deep learning**

### Cons

- Can sometimes generalise slightly worse than plain SGD on some specific tasks (rare)
- Uses a little more memory (stores both m and v for every weight)

**What poor generalisation looks like:**

```
Training vs Validation loss (Adam sometimes):

Loss
  |*                   ← training loss
  | *
  |  *
  |   *  *  *  *  *  ← training keeps improving
  |
  |      *  *  *  *  *  *  ← validation loss stops improving
  |         (model memorised training data)
  |_________________________________ epochs
```

Adam is so good at fitting training data that it can sometimes overfit —
learn the training examples too perfectly without learning the underlying pattern.
In these cases, SGD with momentum can actually generalise better because
its noise acts like a natural brake.

### When to use Adam vs SGD

```
Use Adam when:                      Use SGD + Momentum when:
  - Starting a new project            - Model is overfitting and you want
  - You don't want to tune a lot        some natural noise to regularise
  - Image classification              - You have time to tune carefully
  - NLP tasks                         - Research / competition fine-tuning
  - Almost everything else            - Final polish on a nearly-done model
```

Rule of thumb: **Start with Adam. If it overfits, switch to SGD + Momentum.**

---

## 7. AdaGrad — Adaptive Gradient Algorithm

### Why was it needed?

Before AdaGrad, every weight in the network used the same learning rate.
That's a problem because different weights need different amounts of adjustment.

Imagine teaching two students:

- One already knows 90% of the material (frequent, large gradients)
- One is completely new (rare, small gradients)

Giving them the same amount of homework doesn't make sense. AdaGrad fixes this.

```
Same learning rate for all:      AdaGrad (different per weight):

Weight A (frequent updates):     Weight A  →  smaller steps (already trained enough)
  learning_rate = 0.01           Weight B  →  bigger steps  (still needs more learning)
Weight B (rare updates):
  learning_rate = 0.01  ← unfair
```

### The idea

**Adapt the learning rate for each weight separately.**

Weights that get updated a lot → slow them down (smaller steps).
Weights that rarely get updated → speed them up (bigger steps).

This is especially useful in NLP (text) and sparse data where most features
appear rarely but a few appear all the time.

### Formula

```
G = G + gradient²             ← accumulate squared gradients over time

weight = weight - (learning_rate / √(G + ε)) × gradient
```

Where:

- `G` = accumulated sum of all past squared gradients for this weight
- `ε` (epsilon) = tiny number (like 1e-8) to avoid dividing by zero
- As G grows → learning rate shrinks → smaller and smaller steps

**Example with numbers:**

```
Weight A (high gradient history):
  G = 100
  effective_lr = 0.01 / √100 = 0.01 / 10 = 0.001  ← very small step

Weight B (rarely updated):
  G = 1
  effective_lr = 0.01 / √1 = 0.01 / 1 = 0.01  ← normal step
```

### Graph — starts well, then slows down

```
Loss
  |*
  | *
  |  *
  |   *
  |    *  *  *  *  *  *  * ← learning rate dying out,
  |                          barely moving after a while
  |_________________________________ updates
```

Starts fast but slows down significantly over time because G keeps growing
and never shrinks. Eventually the effective learning rate becomes nearly zero.

### Worked example — where it shines (NLP / text data)

```
Vocabulary of 10,000 words. Most sentences use:
  "the", "is", "a"  → appear constantly → frequent updates → AdaGrad slows them
  "serendipity"     → appears rarely    → rare updates    → AdaGrad keeps learning it

Without AdaGrad, "the" would dominate the gradient and rare words would
barely learn anything. AdaGrad balances this automatically.
```

### Pros

- Great for sparse data — rare features get to learn properly
- No manual tuning of learning rate per feature — it adapts automatically
- Works very well for NLP and recommendation systems

### Cons

- **Dying learning rate** — G only grows, never shrinks. Eventually the effective
  learning rate goes to almost zero and the model stops learning entirely
- Not great for deep networks or long training runs
- RMSProp and Adam fix this problem (they use a decaying average of G instead of the full sum)

**What the dying learning rate looks like:**

```
Effective learning rate over time:

lr
  |*
  | *
  |  *
  |    *
  |       *
  |           *
  |                *
  |                        *  ← almost zero, model freezes
  |_________________________________ updates
```

---

## 8. AdaBoost — A Quick Note (Not a Neural Network Optimizer)

> **Important:** AdaBoost is NOT an optimizer for neural networks.
> It's a completely different technique from Machine Learning called
> an **ensemble method**. It's listed here because it sounds similar
> to AdaGrad but they are completely unrelated.

### What is AdaBoost then?

AdaBoost (Adaptive Boosting) is a method that combines many **weak learners**
(simple models that are only slightly better than random guessing) into one
**strong learner** that performs really well.

Think of it like a courtroom jury — no single juror is always right, but
together they make better decisions than any one person alone.

### The idea

1. Train a simple model (e.g. a tiny decision tree)
2. Look at which examples it got wrong
3. Give those wrong examples more importance (boost their weight)
4. Train the next model — it now focuses harder on the hard examples
5. Repeat many times
6. Final answer = weighted vote of all models together

```
Round 1: Model A  →  gets rows 3, 7, 12 wrong  →  boost those rows
Round 2: Model B  →  focuses on rows 3, 7, 12  →  gets them right but fails on 5, 9
Round 3: Model C  →  focuses on rows 5, 9 ...

Final prediction = weighted vote of Model A + B + C + ...
```

### Graph — how accuracy improves with more models

```
Accuracy
  |                              * * * * *  ← plateaus (adding more doesn't help)
  |                    *  *  *
  |            *  *
  |      *  *
  |  *
  |_________________________________ number of models (estimators)
  0    10   20   30   40   50
```

Each additional model improves accuracy until it plateaus.
More isn't always better — after a point you're just wasting compute.

### Example — classifying emails as spam

```
Simple Model 1: "if email has 'free' → spam"       → 60% accurate
Simple Model 2: "if email has 'win' → spam"         → 62% accurate
Simple Model 3: "if email has no greeting → spam"   → 58% accurate

AdaBoost combines all three:
Final model → 89% accurate  ← much better than any single rule
```

### Why is it called "Adaptive"?

Because each new model _adapts_ by focusing on the mistakes of the previous ones.
The hard examples get more attention, the easy ones get less.

### Pros

- Often achieves very high accuracy with simple base models
- Less prone to overfitting than a single complex model
- Interpretable — you can see what each weak learner is doing
- Works great for tabular/structured data

### Cons

- Sensitive to noisy data and outliers — wrong examples get boosted, including garbage data
- Slower than a single model — you're training many models in sequence
- Not used inside neural networks — its home is traditional ML (scikit-learn, etc.)

### Code example (scikit-learn)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Weak learner = tiny decision tree (depth 1, called a "stump")
weak_learner = DecisionTreeClassifier(max_depth=1)

# AdaBoost combines 100 of these stumps
model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=100,      # number of weak models to combine
    learning_rate=0.5      # how much each model contributes
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

### AdaGrad vs AdaBoost — don't mix them up

|                | AdaGrad                    | AdaBoost                              |
| -------------- | -------------------------- | ------------------------------------- |
| Type           | Neural network optimizer   | Ensemble ML method                    |
| Used in        | Deep learning              | Traditional ML                        |
| What it adapts | Learning rate per weight   | Which training examples to focus on   |
| Fixes          | Unfair learning rates      | Weak individual models                |
| Library        | `keras.optimizers.Adagrad` | `sklearn.ensemble.AdaBoostClassifier` |

---

| Optimizer     | Speed     | Stability   | Memory | Best For                  |
| ------------- | --------- | ----------- | ------ | ------------------------- |
| Batch GD      | Very slow | Very stable | High   | Small datasets, research  |
| SGD           | Fast      | Noisy       | Low    | Large datasets            |
| Mini-Batch GD | Good      | Balanced    | Low    | Almost everything         |
| Momentum      | Faster    | Smoother    | Low    | Deep networks, noisy data |
| RMSProp       | Adaptive  | Good        | Medium | Complex problems, RNNs    |
| Adam          | Very fast | Very good   | Medium | **Default choice**        |
| AdaGrad       | Adaptive  | Good early  | Medium | Sparse data, NLP          |
| AdaBoost      | N/A       | N/A         | Low    | Ensemble ML (not ANN)     |

---

## The Simple Way to Remember

```
Batch GD    →  uses ALL data  →  slow but perfect steps
SGD         →  uses 1 row     →  fast but chaotic
Mini-Batch  →  uses a chunk   →  best balance (use this by default)

Momentum    →  adds memory of past direction  →  builds up speed
RMSProp     →  adapts learning rate per weight  →  smart step sizes
Adam        →  Momentum + RMSProp together  →  best of everything
```

**Rule of thumb:** Start with **Adam**. If your model isn't generalising well on a specific problem, try **SGD with Momentum** instead.

---

## Coming Next

- [ ] Learning rate schedules (how to reduce learning rate over time)
- [x] Dropout regularisation — see `Convolutional_Neural_Network/notes.md` Section 17
- [ ] L1 and L2 regularisation techniques
- [ ] Batch normalization
