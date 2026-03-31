# Recurrent Neural Network — Sentence Completion with RNN & LSTM

A step-by-step project that trains an RNN and an LSTM model on 3,038 famous quotes to predict the next word in any sentence — just like autocomplete on your phone.

---

## What This Project Does

Two neural networks are trained to predict the next word given the start of a sentence:

- **Model 1 — SimpleRNN**: a basic recurrent network. Reads words one by one and keeps one hidden state. Fast to train but forgetful on long sentences.
- **Model 2 — LSTM**: a more powerful recurrent network that keeps two memory streams. Remembers patterns from much earlier in the sentence — the one we actually train and save.

The trained LSTM can then take any text prompt and suggest what word comes next.

---

## Dataset

A CSV of 3,038 famous quotes collected from public sources.

| File                     | Rows  | Description                                     |
| ------------------------ | ----- | ----------------------------------------------- |
| `data/qoute_dataset.csv` | 3,038 | Famous quotes with `quote` and `Author` columns |

Sample quotes:

```
"The world as we have created it is a process of our thinking..."  →  Albert Einstein
"It is our choices, Harry, that show what we truly are..."         →  J.K. Rowling
"Imperfection is beauty, madness is genius..."                     →  Marilyn Monroe
```

---

## Project Structure

```
Recurrent_Neural_Network/
├── sentence_completion_rnn.py   ← all code: preprocessing, models, training, saving
├── sentiment_classifier_rnn.py  ← earlier project: binary sentiment classification
├── next_word_prediction_ui.py   ← Streamlit UI to interact with the trained model
├── notes.md                     ← full theory notes (RNN, LSTM, GRU from scratch)
├── rnn_revision_notes.md        ← quick revision cheat sheet
├── data/
│   └── qoute_dataset.csv        ← 3,038 famous quotes
└── images/                      ← architecture diagrams and training time screenshots
```

---

## How It Works — The Full Pipeline

```
CSV file (3,038 quotes)
     ↓
Clean text → lowercase, remove all punctuation
     ↓
Tokenise → assign a number to each word (vocabulary size = 10,000)
     ↓
Build input/output pairs:
  "the world"    → predict "as"
  "the world as" → predict "we"
  ...
  → 85,271 training samples from 3,038 quotes
     ↓
Pad sequences → all inputs padded to length 745 (longest quote)
     ↓
One-hot encode targets → shape (85,271 × 10,000)
     ↓
Build model: Embedding(10000, 50) → LSTM(128) → Dense(10000, Softmax)
     ↓
Train: 10 epochs, batch=128, validation=10%
     ↓
Save: lstm_model.keras
     ↓
Predict: give a phrase → get the next word back
```

---

## Models

### SimpleRNN (commented out — kept for comparison)

```
Input (padded sequence, length 745)
   ↓
Embedding(10000 vocab, 50 dimensions)
   ↓
SimpleRNN(128 units)
   ↓
Dense(10000, Softmax)  →  probability for each word in vocabulary
```

| Setting    | Value                    |
| ---------- | ------------------------ |
| Optimizer  | Adam                     |
| Loss       | Categorical Crossentropy |
| Epochs     | 10                       |
| Batch size | 128                      |

### LSTM (the model we train and save)

```
Input (padded sequence, length 745)
   ↓
Embedding(10000 vocab, 50 dimensions)
   ↓
LSTM(128 units)   ← two memory streams: hidden state + cell state
   ↓
Dense(10000, Softmax)  →  probability for each word in vocabulary
```

| Setting          | Value                    |
| ---------------- | ------------------------ |
| Optimizer        | Adam                     |
| Loss             | Categorical Crossentropy |
| Epochs           | 10                       |
| Batch size       | 128                      |
| Validation split | 10%                      |

**Why LSTM over SimpleRNN?**
A single quote can be 50–100 words long. A SimpleRNN forgets words from the start of the sentence by the time it reaches the end. LSTM's cell state carries memory across the whole sequence without fading.

---

## Preprocessing Details

| Step              | What happens                                                             |
| ----------------- | ------------------------------------------------------------------------ |
| Lowercase         | All text converted to lowercase                                          |
| Strip punctuation | All commas, full stops, apostrophes etc. removed                         |
| Tokenise          | Each unique word gets a number; top 10,000 words kept                    |
| Sequence pairs    | Every prefix of every quote becomes one training sample                  |
| Padding           | All sequences padded with zeros at the front to length 745               |
| One-hot encode    | Each target word index → vector of length 10,000 (1 at correct position) |

---

## Training Time

![RNN training time](images/training%20time%20for%20rnn%20model.png)
_SimpleRNN — each epoch completes quickly but accuracy is lower_

![LSTM training time](images/training%20time%20for%20ltsm%20model.jpg)
_LSTM — each epoch takes longer but the model learns better patterns and remembers more context_

---

## Saved Model Files

After training, two formats are saved:

| File               | Format | Notes                                        |
| ------------------ | ------ | -------------------------------------------- |
| `lstm_model.keras` | Modern | Recommended — works with TensorFlow 2.x+     |
| `lstm_model.h5`    | Legacy | Kept for reference; may not load on newer TF |

---

## Streamlit UI

`next_word_prediction_ui.py` is a simple Streamlit web app that loads the trained LSTM and lets you type a sentence to see the predicted next word in your browser.

Run with:

```bash
streamlit run next_word_prediction_ui.py
```

> Note: make sure `lstm_model.keras` (or `.h5`) and a saved `tokenizer.pkl` and `max_len.pkl` are in the same folder before running.

---

## How to Run the Training Script

```bash
python sentence_completion_rnn.py
```

The script will:

1. Load `data/qoute_dataset.csv`
2. Clean and tokenise the quotes
3. Build 85,271 training pairs
4. Pad and one-hot encode
5. Build the LSTM model
6. Train for 10 epochs
7. Save `lstm_model.keras` and `lstm_model.h5`

> Training on CPU takes a while — this is expected. The model is processing 85,271 sequences each 745 steps long.

---

## Dependencies

```
numpy
pandas
tensorflow / keras (tf_keras for legacy Keras 2 API)
```

Install with:

```bash
pip install numpy pandas tensorflow tf_keras
```

---

## Notes

Full theory notes covering RNNs, LSTMs, and GRUs from first principles are in [notes.md](notes.md), including:

- Why ANNs fail on sequential data
- How RNNs process sequences step by step
- What vanishing gradients are and why they matter
- How LSTM gates (forget, input, output) solve the memory problem
- How GRU simplifies LSTM with just two gates
- A full walkthrough of this project in plain English
