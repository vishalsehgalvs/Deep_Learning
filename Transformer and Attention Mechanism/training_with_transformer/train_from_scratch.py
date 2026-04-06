# ============================================================
# TRAINING A TRANSFORMER FROM SCRATCH
# ------------------------------------------------------------
# This means: no borrowed knowledge, no pretrained weights.
# The model starts knowing NOTHING and learns only from
# the spam dataset you give it.
# ------------------------------------------------------------
# Dataset needed: spam.csv from Kaggle
#   https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
#   Place spam.csv in the same folder as this file.
# ============================================================

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ── Step 1: Load the spam messages ───────────────────────────
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # ham=0, spam=1
print(f"Loaded {len(df)} messages ({df['label'].sum()} spam, {(df['label']==0).sum()} ham)")

# ── Step 2: Build a word dictionary ──────────────────────────
# We look at all the words in the dataset and pick the 5000
# most common ones. Each word gets a unique number.
# e.g.  "free" → 3,  "call" → 7,  unknown word → 1
all_words = " ".join(df["text"]).lower().split()
vocab = {w: i+2 for i, (w, _) in enumerate(Counter(all_words).most_common(5000))}
vocab["<pad>"] = 0   # used to fill empty space at end of short sentences
vocab["<unk>"] = 1   # used for words we haven't seen before

def encode(text, max_len=50):
    # Turn a sentence into a list of numbers, padded/trimmed to 50
    # "free prize now" → [3, 9, 22, 0, 0, 0, ...]
    tokens = [vocab.get(w, 1) for w in text.lower().split()[:max_len]]
    return tokens + [0] * (max_len - len(tokens))

# ── Step 3: Wrap the data so PyTorch can loop through it ─────
class SpamDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor([encode(t) for t in df["text"]], dtype=torch.long)
        self.y = torch.tensor(df["label"].tolist(), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

split = int(0.8 * len(df))
train_loader = DataLoader(SpamDataset(df[:split]), batch_size=32, shuffle=True)
test_loader  = DataLoader(SpamDataset(df[split:]),  batch_size=32)

# ── Step 4: Define the transformer model ─────────────────────
class SmallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Converts each word number into 64 values the model can learn from
        self.embed = nn.Embedding(5002, 64, padding_idx=0)
        # Tells the model the position of each word (1st, 2nd, 3rd ...)
        self.pos   = nn.Embedding(50, 64)
        # The attention layer: each word looks at every other word
        # to understand the full context of the sentence
        self.attn  = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        # A small thinking layer after attention
        self.ff    = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        # Final layer: outputs 2 scores → ham score vs spam score
        self.out   = nn.Linear(64, 2)

    def forward(self, x):
        positions = torch.arange(x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.pos(positions)  # word meaning + word position
        x, _ = self.attn(x, x, x)               # words look at each other
        x = self.ff(x)                           # think about what was seen
        x = x.mean(dim=1)                        # combine all words into one summary
        return self.out(x)                       # make the final decision

# ── Step 5: Train the model ───────────────────────────────────
model     = SmallTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # updates weights after each batch
loss_fn   = nn.CrossEntropyLoss()                          # measures how wrong the model is

print("\nTraining from scratch...\n")
for epoch in range(5):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()            # forget last round's calculations
        loss = loss_fn(model(X), y)      # how wrong were we this batch?
        loss.backward()                  # figure out what to fix
        optimizer.step()                 # apply the fix
        total_loss += loss.item()

    # Check how accurate we are on unseen test data
    model.eval()
    correct = sum((model(X).argmax(1) == y).sum().item() for X, y in test_loader)
    print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {correct/len(df[split:]):.2%}")

# ── Step 6: Try it yourself ───────────────────────────────────
print("\n--- Testing on custom messages ---")
samples = [
    "Congratulations! You've won a FREE prize, call now to claim!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your account will be suspended. Click here now.",
    "Can you pick up some milk on your way home?"
]
for text in samples:
    x    = torch.tensor([encode(text)], dtype=torch.long)
    pred = model(x).argmax(1).item()
    print(f"  {'SPAM' if pred == 1 else 'HAM '} ← {text}")
