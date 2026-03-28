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

---

## 6. Use Cases of RNNs

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

## 7. Where RNNs Stand Today

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

## 8. Quick Reference

| Concept             | What it means                                                                |
| ------------------- | ---------------------------------------------------------------------------- |
| Sequential data     | Data where the order of items matters and cannot be changed                  |
| ANN limitation      | Fixed input size, no memory, loses sequence — fails on text                  |
| RNN                 | Neural network with a hidden state that carries memory across the sequence   |
| Hidden state        | The running memory of the RNN — updated at each step                         |
| Vectorisation       | Converting words (or any non-numeric data) into numbers before feeding in    |
| One-hot encoding    | Representing each word as an array with a single 1 and all other positions 0 |
| Sentiment analysis  | Classifying whether a piece of text is positive or negative                  |
| Sentence completion | Predicting the next word(s) given the start of a sentence                    |
| Transformers / LLMs | Newer architecture that has largely replaced RNNs for most NLP tasks         |

---

## Coming Next

- [ ] LSTM — Long Short-Term Memory (fixes RNN's memory limitations over long sequences)
- [ ] Vanishing gradient in RNNs — why memory fades over long sequences
- [ ] RNN project — building a text model
