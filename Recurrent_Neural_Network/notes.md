# Recurrent Neural Network (RNN) — Notes

---

## 1. Why Neural Networks for Certain Tasks?

Not all machine learning problems are the same. Some tasks — like recognising a handwritten digit or predicting whether a plant needs water — work fine with a standard ANN. You feed in a fixed set of numbers and get an answer out.

But some tasks are completely different. The input is not a fixed table of numbers — it is a **sequence**. A sentence. A series of events over time. Something where the **order matters**.

For tasks like these, a standard ANN breaks down. That's where an RNN (Recurrent Neural Network) comes in.

---

## 2. What Is Sequential Data?

Sequential data is any data where the **order of the items matters and cannot be changed** without changing the meaning.

The clearest example is text. Take a simple sentence:

```
"The cat sat on the mat"
```

If you shuffle the words randomly:

```
"mat the on sat cat the"
```

It means nothing. The sequence — the order — is what carries the meaning.

Other examples where order matters in the same way:

| Type of data              | What makes it sequential                                         |
| ------------------------- | ---------------------------------------------------------------- |
| A sentence or review      | Words must appear in order — shuffle them and meaning is lost    |
| A person's salary history | Salary at age 22, 25, 28, 30 — the progression over time matters |
| Location tracking         | Where someone was at 9am, 10am, 11am — order tells the story     |

The key point: **you cannot treat each item in isolation**. Item 5 in the sequence means something different depending on what came before it.

---

## 3. Why a Plain ANN Cannot Handle Sequential Data

A standard ANN takes a fixed set of inputs, processes them all at once, and gives an answer. There is no concept of "what came before."

When you try to use an ANN on text or any sequential data, you immediately hit four big problems:

### Problem 1 — Variable Length Input

A review written by one person might be 10 words long. Another might be 300 words long. An ANN has a fixed number of input neurons — it cannot handle different-sized inputs. You would need a separate model for every possible sentence length, which is impossible.

```
"Good film"                        →  2 words
"This was one of the best films I have ever seen in my life"  →  15 words

ANN input layer:  [●  ●  ●  ●  ●  ●]   ← fixed size, cannot change
```

### Problem 2 — Massive Computational Load

To give an ANN any chance of understanding long sentences, you'd have to make the input layer enormous — one neuron per word, across the entire possible vocabulary. That's potentially tens of thousands of input neurons, each connected to a hidden layer. The number of weights explodes instantly.

### Problem 3 — The Prediction Problem

Even if you handled training, there's a problem at test time. When someone submits a new piece of text to your model, it might be longer than anything you trained on. The ANN has no way to deal with an input that doesn't match the exact size it was built for.

### Problem 4 — Sequence Is Not Maintained (Semantic Meaning Lost)

This is the biggest one. An ANN has no memory of what it saw before the current input. Every input is treated independently.

```
Sentence: "I did not enjoy the film at all"

ANN sees:  ["I", "did", "not", "enjoy", ...]
           → Each word processed separately, in isolation
           → "not" has no relationship to "enjoy" as far as ANN is concerned
           → Meaning completely lost
```

The word "not" completely reverses the meaning of "enjoy" — but an ANN cannot know that because it doesn't track the sequence. Remove the word order and you lose the meaning.

---

## 4. What Is an RNN?

An RNN is a type of neural network designed specifically to handle sequential data. The core idea is: **it has a memory.**

At each step, the RNN does not just look at the current input. It also looks at what it learned from all the **previous inputs in the sequence**. This memory is called the **hidden state**.

```
Normal ANN:

  Input → [Network] → Output
  (no memory, processes each input completely fresh)


RNN:

  Input 1 → [RNN] → Output 1
               ↓
           Hidden State (memory of step 1)
               ↓
  Input 2 → [RNN] → Output 2
               ↓
           Hidden State (memory of steps 1 + 2)
               ↓
  Input 3 → [RNN] → Output 3

  (each step carries forward what it learned from all previous steps)
```

Think of it like reading a book. When you reach page 50, you understand it because you remember what happened on pages 1–49. You're not reading each page with a blank mind. An RNN does the same — it carries a running memory as it processes each item in the sequence one step at a time.

### The Full Flow

```
            ┌─────────────────────────────────────────────────────────┐
            │                   RNN — Step by Step                    │
            │                                                         │
  Word 1 →  │  [Cell] → hidden state h1 ───────────────────────────► │
            │     ↑                                                   │
  Word 2 →  │  [Cell] → hidden state h2  (h2 knows about word 1+2) ► │
            │     ↑                                                   │
  Word 3 →  │  [Cell] → hidden state h3  (h3 knows about 1+2+3)   ► │
            │     ↑                                                   │
  Word N →  │  [Cell] → Final Output  (based on entire sequence)      │
            └─────────────────────────────────────────────────────────┘
```

The same cell is reused at each step — that's the "recurrent" in RNN. It loops back on itself, carrying memory forward.

---

## 5. How Text Is Fed Into an RNN — Vectorisation and One-Hot Encoding

A neural network only understands numbers — it cannot work with raw words. So before feeding any text into an RNN, you must convert every word into a number (or a list of numbers). This process is called **vectorisation**.

The most basic way to do this is called **One-Hot Encoding**.

### What One-Hot Encoding Does

First, you build a vocabulary — a list of every unique word in your dataset. Say your vocabulary has 5 words:

```
Vocabulary:
  Index 0 = "I"
  Index 1 = "love"
  Index 2 = "not"
  Index 3 = "hate"
  Index 4 = "film"
```

Each word is turned into an array (a list of numbers) where every position is 0, **except the position that matches that word's index — that one is 1**:

```
"I"     →  [1, 0, 0, 0, 0]
"love"  →  [0, 1, 0, 0, 0]
"not"   →  [0, 0, 1, 0, 0]
"hate"  →  [0, 0, 0, 1, 0]
"film"  →  [0, 0, 0, 0, 1]
```

One word = one array. Only one position is "hot" (1), the rest are "cold" (0). That's why it's called **one-hot**.

### From Words to Arrays to the Network

Once every word is encoded, the sentence becomes a sequence of arrays:

```
Sentence: "I love film"

After one-hot encoding:

  Step 1:  "I"     →  [1, 0, 0, 0, 0]  → fed into RNN cell
  Step 2:  "love"  →  [0, 1, 0, 0, 0]  → fed into RNN cell (+ memory from step 1)
  Step 3:  "film"  →  [0, 0, 0, 0, 1]  → fed into RNN cell (+ memory from steps 1+2)
                                             ↓
                                        Final output (e.g. "positive sentiment")
```

The RNN processes the sentence one word at a time, step by step, carrying the hidden state forward at each step until it has seen the entire sentence.

### Timesteps and Input Features — How the Data Is Actually Structured

When you feed a sentence into an RNN, the data is organised into **timesteps**. Each timestep is one position in the sequence — word 1 is timestep 1, word 2 is timestep 2, and so on. At each timestep, the RNN receives the **input feature vector** for that word (its one-hot encoded array) plus the **hidden state** carried forward from the previous step.

Let's walk through **"I love film"** step by step.

```
Vocabulary (5 words):
  Index:  0="I"   1="love"   2="not"   3="hate"   4="film"

Sentence:   "I      love    film"
Timestep:    t=1     t=2     t=3
```

At each timestep the RNN gets one input feature vector and produces an updated hidden state:

```
Timestep 1  (word = "I"):
  Input features:   [1, 0, 0, 0, 0]   ← one-hot for "I"
  Hidden state in:  [0, 0, 0, 0, 0]   ← starts as all zeros — no memory yet
  Hidden state out: h1                ← memory now holds "I"

Timestep 2  (word = "love"):
  Input features:   [0, 1, 0, 0, 0]   ← one-hot for "love"
  Hidden state in:  h1                ← memory of "I" carried forward
  Hidden state out: h2                ← memory now holds "I love"

Timestep 3  (word = "film"):
  Input features:   [0, 0, 0, 0, 1]   ← one-hot for "film"
  Hidden state in:  h2                ← memory of "I love" carried forward
  Hidden state out: h3                ← memory now holds "I love film"
                         ↓
                   Final output  →  "positive sentiment"
```

The block of data fed into the RNN has the shape **[timesteps × features]**:

```
Shape:  [ 3 timesteps × 5 features ]

  Row 1 (t=1)  →  [1, 0, 0, 0, 0]   ← "I"
  Row 2 (t=2)  →  [0, 1, 0, 0, 0]   ← "love"
  Row 3 (t=3)  →  [0, 0, 0, 0, 1]   ← "film"
```

The RNN reads one row at a time, top to bottom. At every row it updates the hidden state and moves to the next. By the last row it has seen the entire sentence in order.

---

## 6. Building an RNN with Keras and TensorFlow

To build and train an RNN in Python we use **Keras** — a high-level deep learning library that runs on top of **TensorFlow**.

Think of TensorFlow as the engine room — it handles all the heavyweight maths (matrix multiplications, gradients, GPU processing). Keras is the friendly layer on top of it — you describe the model in simple, readable steps without writing any of the low-level maths yourself.

```
TensorFlow  =  the engine     (fast, powerful — handles all maths under the hood)
Keras       =  the interface   (simple — you just stack layers and go)
```

Keras comes built into TensorFlow so you don't install them separately:

```python
import tensorflow as tf
from tensorflow import keras
```

### How an RNN Looks in Keras

Building an RNN in Keras follows the same pattern as the ANN and CNN projects — you stack layers inside a `Sequential` model. The new piece here is the `SimpleRNN` layer, which is the recurrent cell that processes one timestep at a time and carries the hidden state forward.

```python
model = keras.Sequential([
    keras.layers.SimpleRNN(64, input_shape=(3, 5)),
    keras.layers.Dense(1, activation='sigmoid')
])
```

Breaking this down:

```
keras.Sequential([...])
  → Stack of layers, one after another — same as ANN and CNN

SimpleRNN(64, input_shape=(3, 5))
  → The recurrent layer — loops through the sequence one timestep at a time
  → 64  = size of the hidden state (how much memory the cell carries per step)
  → input_shape = (3, 5)
                   ↑  ↑
                   |  └── 5 features (vocabulary size = 5 words)
                   └───── 3 timesteps (sentence has 3 words)

Dense(1, activation='sigmoid')
  → Final output layer — same as ANN
  → 1 neuron + Sigmoid → outputs a probability between 0 and 1
  → Good for binary tasks like sentiment (positive / negative)
```

### Batch Size — How Many Sentences at Once

When training, you don't feed sentences to the RNN one by one — that would be extremely slow. Instead you group them into **batches** and process several sentences at the same time.

```
Batch size = 32  →  32 sentences processed together in one go
Batch size = 1   →  one sentence at a time (slowest, only useful for tiny datasets)
```

The full shape of the data block passed to Keras is:

```
Shape:  [ batch_size  ×  timesteps  ×  features ]

Example with batch_size=32, 3-word sentences, vocabulary of 5:
  →  [ 32  ×  3  ×  5 ]
      ↑       ↑     ↑
      |       |     └── 5 numbers per word (one-hot vector)
      |       └──────── 3 words per sentence
      └──────────────── 32 sentences processed at once
```

If you don't set a batch size, Keras defaults to processing each sentence one at a time — which works but is much slower.

The full pipeline from raw sentence to prediction:

```
Raw text  "I love film"
        ↓
Split into words  →  ["I", "love", "film"]
        ↓
One-hot encode   →  [[1,0,0,0,0], [0,1,0,0,0], [0,0,0,0,1]]
        ↓
Organise into shape [3 × 5]  (timesteps × features)
        ↓
Feed into SimpleRNN  →  processes t=1, t=2, t=3, carries hidden state
        ↓
Dense output layer
        ↓
0.87  →  positive sentiment  ✓
```

---

## 7. Use Cases of RNNs

RNNs shine on any task where the input or output is a sequence — especially text. The two main use cases noted:

### Sentiment Analysis

Given a piece of text (a product review, a tweet, a comment), predict whether the sentiment is positive or negative.

```
Input:   "The service was absolutely terrible and I will never return"
Output:  Negative sentiment

Input:   "Had an amazing experience — will definitely come back"
Output:  Positive sentiment
```

The RNN reads the sentence word by word, building up context as it goes. By the time it reaches the last word, its hidden state carries the full meaning of the sentence and it can make a judgment.

### Sentence Completion

Given the start of a sentence, predict what word comes next — and keep going to complete the sentence.

```
Input:   "The weather today is very"
RNN predicts next word → "hot"  → then "and" → then "sunny" → ...
Output:  "The weather today is very hot and sunny"
```

The RNN uses its memory of every word it has seen so far to predict the most likely next word.

---

## 8. Where RNNs Stand Today

RNNs were the go-to approach for NLP tasks for several years. However, they have become **less common in recent times**.

The reason: **Transformers** came along and solved most of the same problems, but much better. Transformers (the architecture behind LLMs like ChatGPT) can process entire sequences at once instead of step by step, handle much longer sequences without losing track, and train far faster on modern hardware.

```
Timeline:

  ANN  →  could not handle sequences at all
  RNN  →  could handle sequences (used widely for NLP, 2010s)
  Transformers / LLMs  →  handle sequences far better (dominant today)
```

RNNs are still useful to understand because:

- They explain **why sequential processing matters**
- They are the foundation that led to more advanced models
- They are still used in some lightweight, resource-constrained settings

---

## 9. Quick Reference

| Concept             | What it means                                                                |
| ------------------- | ---------------------------------------------------------------------------- |
| Sequential data     | Data where the order of items matters and cannot be changed                  |
| ANN limitation      | Fixed input size, no memory, loses sequence — fails on text                  |
| RNN                 | Neural network with a hidden state that carries memory across the sequence   |
| Hidden state        | The running memory of the RNN — updated at each step                         |
| Timestep            | One position in the sequence — the RNN processes one timestep at a time      |
| Input features      | The feature vector (e.g. one-hot array) fed into the RNN at each timestep    |
| Vectorisation       | Converting words (or any non-numeric data) into numbers before feeding in    |
| One-hot encoding    | Representing each word as an array with a single 1 and all other positions 0 |
| TensorFlow          | The deep learning engine — handles all the low-level maths and GPU work      |
| Keras               | High-level interface on top of TensorFlow — lets you stack layers simply     |
| SimpleRNN layer     | The Keras layer that processes one timestep at a time and carries memory     |
| Sentiment analysis  | Classifying whether a piece of text is positive or negative                  |
| Sentence completion | Predicting the next word(s) given the start of a sentence                    |
| Transformers / LLMs | Newer architecture that has largely replaced RNNs for most NLP tasks         |
| Batch size          | How many sentences the RNN processes at once — if not set, one at a time     |

---

## 10. RNN Architecture — How the Recurrent Part Actually Works

### Feed Forward vs Recurrent — What's the Difference?

A regular ANN is called a **feed forward** network. Data moves in one direction only — from the input layer, through the hidden layers, to the output. No looking back, no memory.

An RNN is a **recurrent** network. At every step, the hidden layer passes information forward to the next timestep — it loops back on itself. That loop is what gives the RNN its memory.

![Feed forward vs feed backward neural network](images/feedforward%20vs%20feedbackward.jpg)

![RNN vs Feed Forward network comparison](images/recurrent_neural_network_vs_feedforward_neural_network_training_ppt_slide01.jpg)

```
Feed Forward (ANN):              Recurrent (RNN):

  Input                            Input (t=1)
    ↓                                ↓
  Hidden Layer  →  Output          Hidden Layer ──┐  (loops back to next step)
                                     ↓            │
                                   Output         │
                                                  ↓
                                   Input (t=2) + hidden from t=1
                                     ↓
                                   Hidden Layer ──┐
                                     ...         ...
```

### The Hidden Layer — The Heart of the RNN

In an RNN, the hidden layer does more than just process the current input — it also carries memory from every previous step. At every timestep it:

1. Takes the current word's vector as input
2. Takes the hidden state from the previous timestep (memory of everything seen so far)
3. Combines them, runs an activation function, and produces a new hidden state

```
new hidden state = tanh( (current input × weights_input) + (previous hidden × weights_hidden) + bias )
```

In plain terms: **new memory = tanh( what I see now + what I already remembered + bias )**

### tanh — The Activation Function Inside the Hidden Neurons

The default activation function inside an RNN's hidden neurons is **tanh** (hyperbolic tangent). It squishes any number into a range between −1 and +1:

```
tanh output range:

  Strong signal one way   →  close to  +1
  No signal               →         0
  Strong signal other way →  close to  −1
```

This −1 to +1 range works better for recurrent networks than Sigmoid (0 to 1) because it lets the hidden state carry both positive and negative signals, which helps gradients flow better during training.

### Hidden State Starts at Zero

Before the first word is processed, the RNN has no memory of anything. So the hidden state is initialised as all zeros at the very start of every sentence.

```
Before t=1:   hidden state = [0, 0, 0, ..., 0]   ← blank memory, nothing seen yet

After  t=1:   hidden state = h1                  ← memory of word 1
After  t=2:   hidden state = h2                  ← memory of words 1 + 2
...
After  t=N:   hidden state = hN                  ← full sentence stored in memory
```

### A Worked Example — "You are good"

Say we want to classify the sentiment of the sentence **"You are good"**. Here's exactly what happens step by step.

First, each word gets turned into a vector (using word embeddings — explained below):

```
"you"  →  [1, 0, 0, 0, 0]
"are"  →  [0, 1, 0, 0, 0]
"good" →  [0, 0, 1, 0, 0]
```

These vectors are fed into the hidden layer one word at a time. Say the hidden layer has 3 neurons:

```
Timestep t=1  →  "you" vector  +  hidden state [0, 0, 0]  (blank start)
                  formula:  tanh( [1,0,0,0,0] × W  +  [0,0,0] × W_hidden  +  bias )
                  output:   h1  =  [some numbers]     ← memory now holds "you"

Timestep t=2  →  "are" vector  +  hidden state h1  (memory of "you")
                  formula:  tanh( [0,1,0,0,0] × W  +  h1 × W_hidden  +  bias )
                  output:   h2  =  [some numbers]     ← memory now holds "you are"

Timestep t=3  →  "good" vector  +  hidden state h2  (memory of "you are")
                  formula:  tanh( [0,0,1,0,0] × W  +  h2 × W_hidden  +  bias )
                  output:   h3  =  [some numbers]     ← memory now holds "you are good"
```

### Output Only Comes After the Last Timestep

The hidden layer does NOT send output to the next layer after each word. It processes every word silently, updating its memory each time. Only after all timesteps are finished does it pass the final hidden state forward to the Dense output layer.

```
t=1  →  hidden layer updates silently        (no output passed forward yet)
t=2  →  hidden layer updates silently        (no output passed forward yet)
t=3  →  hidden layer updates silently        (no output passed forward yet)
                        ↓
             all timesteps finished
                        ↓
         h3 passed forward to Dense output layer
                        ↓
             0.89  →  positive sentiment  ✓
```

In Keras this is controlled by `return_sequences=False` on the `SimpleRNN` layer — "False" means: send me only the final hidden state, after the last word.

### Why It's Called "Recurrent" — The Same Weights Used Again and Again

Here is the key insight. The hidden layer uses the **exact same set of weights at every single timestep**. The same `W` and `W_hidden` that process word 1 are the same weights used at word 2, word 3, and every other word in the sentence.

These weights are applied again and again at each step, looping back — that is the **recurrence**. That's why it's called a Recurrent Neural Network.

```
t=1  →  W, W_hidden applied
t=2  →  same W, same W_hidden applied again
t=3  →  same W, same W_hidden applied again
  ↑
  "recurrent" = the same weights are reused at every step, looping back
```

### Memory Limit of a Basic RNN

A basic RNN starts to forget earlier words when the sentence gets too long — roughly beyond **10 words**, the influence of early words on the hidden state fades away. This is called the **vanishing gradient problem** and it is one of the main weaknesses of a plain RNN.

```
Short sentence (≤ ~10 words):  RNN handles it well
Long sentence   (> ~10 words):  RNN starts forgetting what it saw early on
                                 → needs LSTM or GRU to fix this (covered next)
```

---

## Coming Next

- [ ] LSTM — Long Short-Term Memory (fixes RNN's memory limitations over long sequences)
- [ ] Vanishing gradient in RNNs — why memory fades over long sequences
- [ ] RNN project — building a text model
