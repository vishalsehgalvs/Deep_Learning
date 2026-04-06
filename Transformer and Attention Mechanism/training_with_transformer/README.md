# Training with Transformers

> Two simple examples showing how transformers work — one built from scratch, one using a pretrained model. Written for people who are learning, not experts.

---

## What is a Transformer?

A transformer is the type of AI model behind GPT, Claude, Gemini, and most modern AI tools. It's really good at understanding text because it can look at all the words in a sentence at once and figure out which words matter most for understanding the meaning.

For example, in the sentence **"I did not enjoy the movie"**, the word "not" completely changes the meaning — a transformer learns to pay attention to that.

---

## What's in This Folder?

| File                    | What it does                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| `train_from_scratch.py` | Builds a transformer with zero borrowed knowledge. Learns everything from your data alone.     |
| `fine_tuning.py`        | Takes DistilBERT (already trained on millions of web pages) and teaches it your specific task. |

Both use the same **spam detection task** — given an SMS message, decide if it's spam or not.

---

## Which Approach Should You Use?

Think of it like this:

- **From scratch** = hiring someone with no experience and training them fully yourself. Cheap to set up, takes longer to get good, needs a lot of data to work well.
- **Fine-tuning** = hiring an experienced person and just teaching them your company's specific rules. They already know English, grammar, and context — you just point them at your problem.

For almost every real use case, **fine-tuning wins**. But training from scratch is great for learning how things work under the hood.

---

## Dataset

Both examples use the **SMS Spam Collection dataset** from Kaggle.

1. Download it from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Place `spam.csv` in the same folder as the script you want to run

---

## How to Run

**Step 1 — Install the packages**

For `train_from_scratch.py`:

```bash
pip install pandas torch
```

For `fine_tuning.py`:

```bash
pip install pandas torch transformers datasets scikit-learn
```

**Step 2 — Place spam.csv in this folder**

**Step 3 — Run**

```bash
python train_from_scratch.py
# or
python fine_tuning.py
```

---

## What the Output Looks Like

During training you'll see something like this after each epoch (one full pass through the data):

```
Epoch 1/5 | Loss: 0.4821 | Accuracy: 87.35%
Epoch 2/5 | Loss: 0.3102 | Accuracy: 91.60%
Epoch 3/5 | Loss: 0.2201 | Accuracy: 94.12%
```

- **Loss** — how wrong the model is. Lower = better.
- **Accuracy** — how many messages it got right. Higher = better.

At the end, it tests on your own sample sentences:

```
SPAM ← Congratulations! You've won a FREE prize, call now to claim!
HAM  ← Hey, are we still on for lunch tomorrow?
SPAM ← URGENT: Your account will be suspended. Click here now.
HAM  ← Can you pick up some milk on your way home?
```

---

## What You'll Learn From These Examples

| Concept                                         | Where you'll see it                                      |
| ----------------------------------------------- | -------------------------------------------------------- |
| How words are turned into numbers               | Both files — the `encode()` function / tokenizer         |
| Why position matters in a sentence              | `train_from_scratch.py` — the positional embedding       |
| What "attention" actually does                  | `train_from_scratch.py` — the `MultiheadAttention` layer |
| What an epoch is                                | Both files — the training loop                           |
| What loss means                                 | Both files — printed after each epoch                    |
| What fine-tuning means vs training from scratch | Compare both files side by side                          |

---

## Expected Performance

| Approach                 | Expected Accuracy | Training Time (CPU) |
| ------------------------ | ----------------- | ------------------- |
| From scratch             | ~85–92%           | ~3–5 minutes        |
| Fine-tuning (DistilBERT) | ~97–99%           | ~10–30 minutes      |

Fine-tuning is slower to set up (downloads a ~250MB model) but produces much better results because the model already understands English.
