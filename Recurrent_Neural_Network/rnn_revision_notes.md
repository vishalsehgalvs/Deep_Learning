# 🧠 RNN, LSTM & GRU — Revision Notes

---

## 1. What is an RNN?

**Recurrent Neural Network (RNN)** works on **sequential data** and creates a **sequential model**.

### Sequential vs Non-Sequential Data

| Non-Sequential | Sequential |
|---|---|
| Age, Location, Salary → ANN | Text, Time-series → RNN |

> **Key idea**: When data has *order* or *context* (like words in a sentence), use RNN, not ANN.

---

## 2. Why RNN Instead of ANN?

Consider 3 sentences for sentiment analysis:

```
1. I love Sheryians       → 1 (positive)
2. Sheryians provide quality content → 1 (positive)
3. I hate Sheryians       → 0 (negative)
```

### Problem with ANN (OHE approach):

Using **One-Hot Encoding** with vocabulary size = 7 words:

```
Sentence 1 → [ [1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],
                [0,0,1,0,0,0,0] ]
```

**Issues with ANN for text:**

| # | Problem |
|---|---|
| ① | Text input size may vary |
| ② | High computational power needed |
| ③ | Prediction problem — train/test input mismatch |
| ④ | **Semantic meaning lost** — sequence is NOT retained |

---

## 3. RNN Architecture

### Data Feeding Format (Keras):

```
Input shape: (batch_size, timestamp, input_features)
Example:     (3, 4, 6)
```

- **batch_size** = number of sentences
- **timestamp** = number of words per sentence
- **input_features** = vocabulary size (OHE vector length)

### ANN vs RNN:

```
ANN:  Input → [Box] → Output           (Feed Forward Network)
RNN:  Input → [Box] → Output
               ↑_______↓               (Feedback / Recurrent Network)
```

> RNN feeds its **output back as input** to the next timestep.

---

## 4. RNN Forward Pass — Step by Step

**Example sentence:** `"you are good"` → Vocabulary size = 5

```
you  → [1,0,0,0,0]
are  → [0,1,0,0,0]
good → [0,0,1,0,0]
bad  → [0,0,0,1,0]
not  → [0,0,0,0,1]
```

### Activation Function: `tanh` (default in RNN hidden layer)

**T=1 (word: "you"):**
```
h₁ = tanh( you(i) × Wᵢ + b )
   = tanh( (1,5) × (5,3) )  →  shape (1,3)
```

**T=2 (word: "are"):**
```
h₂ = tanh( (are(i) × Wᵢ) + (Wₕ × O₁) + b )
   = tanh( (1,5)×(5,3) + ((1,3)+(1,3)) + b )  →  shape (1,3)
```

**T=3 (word: "good"):**
```
h₃ = tanh( (good(i) × Wᵢ) + (Wₕ × O₂) + b )  →  shape (1,3)
ŷ  = σ( h₃ × Wₒ )
```

### Weight Count:

```
Wᵢ = (5×3) = 15 weights
Wₒ = (3×1) =  3 weights
Wₕ = (3×3) =  9 weights
Total       = 27 weights
```

---

## 5. Backpropagation Through Time (BPTT)

RNN uses **BPTT** — gradients flow backward through all timesteps.

### Forward equations:

```
O₁ = f(x₁·Wᵢ + O₀·Wₕ) + b
O₂ = f(x₂·Wᵢ + O₁·Wₕ) + b
O₃ = f(x₃·Wᵢ + O₂·Wₕ) + b
ŷ  = σ(O₃·Wₒ)
```

### Loss Function (Binary Cross-Entropy):

```
Loss = -yᵢ·log(ŷᵢ) - (1-yᵢ)·log(1-ŷᵢ)
```

**Goal:** Minimize Loss using **Gradient Descent**

### Weight Updates:

```
Wᵢ_new = Wᵢ - η · (∂L/∂Wᵢ)
Wₕ_new = Wₕ - η · (∂L/∂Wₕ)
Wₒ_new = Wₒ - η · (∂L/∂Wₒ)
```

### Chain Rule for BPTT:

```
∂Loss/∂Wₒₗd = (∂Loss/∂O₃) × (∂O₃/∂O₂) × (∂O₂/∂O₁) × (∂O₁/∂Wₒₗd)
```

---

## 6. Problems with RNN

| # | Problem | Effect |
|---|---|---|
| ① | Weak long-term memory | Forgets early words in long sequences |
| ② | Vanishing Gradient | Gradients → 0; early layers don't learn |
| ③ | Exploding Gradient | Gradients → ∞; unstable training |
| ④ | No long dependencies captured | Can't link "Rahul" to "he" 2000 words later |
| ⑤ | Sequential processing is slow | Can't parallelize |

> **Simple RNNs are rarely used today.** LSTM and GRU were invented to solve these problems.

---

## 7. LSTM (Long Short-Term Memory)

### Key Idea:

```
RNN  → Short-term memory only
LSTM → Long-term memory + Short-term memory
```

**Example:**
> "Riya is a doctor. She lives in Delhi. ... 2000 words later ... Where does **Rahul** live?"
> RNN forgets. LSTM **remembers**.

### LSTM State Variables:

| Symbol | Meaning |
|---|---|
| `Cₜ₋₁` | Previous cell state → **Long-term memory** |
| `hₜ₋₁` | Previous hidden state → **Short-term memory** |
| `xₜ` | Input at time t |
| `hₜ` | Current hidden state |
| `Cₜ` | Current cell state |

---

### LSTM Gate 1: Forget Gate 🔴

**Decides what to DISCARD from long-term memory.**

```
fₜ = σ( Wf · [hₜ₋₁, xₜ] + bf )
```

Output range: [0, 1] — closer to 0 means forget, closer to 1 means keep.

**Example ("Riya moved to London"):**
```
Gender     → 0.99  (keep)
Profession → 0.95  (keep)
Delhi      → 0.10  (forget ✗)
Hospital   → 0.85  (keep)
London     → 0.90  (keep ✓)
```

---

### LSTM Gate 2: Input Gate 🟡

**Decides what NEW info to ADD to cell state.**

```
iₜ = σ( Wᵢ · [hₜ₋₁, xₜ] + bᵢ )         ← 2nd filter
C̃ₜ = tanh( Wc · [hₜ₋₁, xₜ] + bc )       ← candidate cell state
```

**Cell State Update:**
```
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
```
- `fₜ * Cₜ₋₁` → scaled old memory
- `iₜ * C̃ₜ`   → scaled new info

---

### LSTM Gate 3: Output Gate 🟢

**Decides what to OUTPUT as hidden state.**

```
oₜ = σ( Wₒ · [hₜ₋₁, xₜ] + bₒ )
hₜ = oₜ * tanh(Cₜ)
```

`tanh(Cₜ)` squashes cell state to **[-1, 1]** before outputting.

---

### LSTM Architecture Summary:

```
     Cₜ₋₁ ──────────×──────────+──────────────► Cₜ
                     ↑          ↑
                    fₜ        iₜ*C̃ₜ
                     │          │
     hₜ₋₁, xₜ → [σ][σ][tanh][σ]
                  forget input       output
                  gate   gate        gate
                                      │
                                    tanh(Cₜ)
                                      │
                                     hₜ ──────────►
```

---

## 8. GRU (Gated Recurrent Unit)

### Why GRU? Problems with LSTM:

| # | Problem |
|---|---|
| ① | Complex architecture |
| ② | More training parameters |
| ③ | Higher time complexity |

> **LSTM has 3 gates → GRU has only 2 gates** (simpler, faster)

### GRU Gates:

| Gate | Formula | Purpose |
|---|---|---|
| **Update Gate** `zₜ` | `zₜ = σ(Wz · [hₜ₋₁, xₜ])` | How much OLD memory to keep |
| **Reset Gate** `rₜ` | `rₜ = σ(Wr · [hₜ₋₁, xₜ])` | How much past influences new info |

### GRU Equations:

```
zₜ = σ( Wz · [hₜ₋₁, xₜ] )          ← Update gate
rₜ = σ( Wr · [hₜ₋₁, xₜ] )          ← Reset gate
h̃ₜ = tanh( W · [rₜ * hₜ₋₁, xₜ] )  ← Candidate hidden state
hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ   ← Final hidden state
```

### How GRU Works (Example):

```
"Riya lives in Delhi"   → T1, T2
"Riya moved to London"  → T1, T2, T3

At T3 (London):
  Update gate zₜ = [0.1, 0.1]  → LOW value → UPDATE the past
                                → Delhi ✗ → London ✓

At T1-T2:
  Update gate zₜ = [0.9, 0.9]  → HIGH value → KEEP the past
```

### GRU Summary:

1. GRU creates new info from the current word
2. Mixes it with old memory
3. Mixing ratio is decided by the **Update Gate**

---

## 9. Activation Functions

| Function | Range | Used In |
|---|---|---|
| `tanh` | [-1, 1] | RNN/LSTM/GRU hidden layers |
| `sigmoid (σ)` | [0, 1] | All gates (forget, input, output, reset, update) |

### Element-wise Operations:

```
× → element-wise multiplication   e.g., [1,2,3] × [3,4,5] = [3,8,15]
+ → element-wise addition
```

---

## 10. RNN vs LSTM vs GRU — Quick Comparison

| Feature | RNN | LSTM | GRU |
|---|---|---|---|
| Memory type | Short-term only | Long + Short term | Long + Short term |
| Gates | None | 3 (forget, input, output) | 2 (update, reset) |
| Vanishing gradient | ✗ Suffers | ✓ Handles | ✓ Handles |
| Parameters | Few | Most | Moderate |
| Speed | Medium | Slowest | Faster than LSTM |
| Long dependencies | ✗ No | ✓ Yes | ✓ Yes |
| Use today | Rarely | Common | Common |

---

## 11. Key Formulas at a Glance

### RNN:
```
hₜ = tanh( xₜ·Wᵢ + hₜ₋₁·Wₕ + b )
ŷ  = σ( hₜ·Wₒ )
```

### LSTM:
```
fₜ = σ(Wf·[hₜ₋₁,xₜ] + bf)            ← Forget gate
iₜ = σ(Wᵢ·[hₜ₋₁,xₜ] + bᵢ)            ← Input gate
C̃ₜ = tanh(Wc·[hₜ₋₁,xₜ] + bc)         ← Candidate
Cₜ = fₜ*Cₜ₋₁ + iₜ*C̃ₜ                 ← Cell state
oₜ = σ(Wₒ·[hₜ₋₁,xₜ] + bₒ)            ← Output gate
hₜ = oₜ * tanh(Cₜ)                    ← Hidden state
```

### GRU:
```
zₜ = σ(Wz·[hₜ₋₁,xₜ])                 ← Update gate
rₜ = σ(Wr·[hₜ₋₁,xₜ])                 ← Reset gate
h̃ₜ = tanh(W·[rₜ*hₜ₋₁, xₜ])          ← Candidate
hₜ = (1-zₜ)*hₜ₋₁ + zₜ*h̃ₜ            ← Hidden state
```

### Loss (Binary Cross-Entropy):
```
L = -y·log(ŷ) - (1-y)·log(1-ŷ)
```

### Gradient Descent Update:
```
W_new = W_old - η · (∂L/∂W)
```

---

*Notes compiled from RNN lecture slides covering: Deep Learning > ANN/CNN/RNN > Sequential data, RNN architecture, BPTT, Problems, LSTM gates, GRU gates.*
