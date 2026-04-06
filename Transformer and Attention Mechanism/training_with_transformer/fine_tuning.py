# ============================================================
# FINE-TUNING A PRETRAINED TRANSFORMER
# ------------------------------------------------------------
# This means: we take a model that already learned English
# from millions of internet pages (DistilBERT), and we
# teach it specifically to detect spam vs ham.
# Think of it like hiring an experienced person and just
# training them on your company's specific rules —
# instead of training someone completely from zero.
# ------------------------------------------------------------
# Dataset needed: spam.csv from Kaggle
#   https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
#   Place spam.csv in the same folder as this file.
#
# Install dependencies:
#   pip install transformers datasets torch pandas scikit-learn
# ============================================================

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# ── Step 1: Load the spam messages ───────────────────────────
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # ham=0, spam=1
print(f"Loaded {len(df)} messages ({df['label'].sum()} spam, {(df['label']==0).sum()} ham)")

# ── Step 2: Prepare text using the pretrained model's tokenizer ──
# DistilBERT has its own way of breaking sentences into pieces.
# We must use its tokenizer (not our own word list like before).
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    # Convert raw text to numbers the way DistilBERT expects
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# Wrap data in Hugging Face Dataset format, then tokenize
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize, batched=True)

# Split: 80% train, 20% test
dataset = dataset.train_test_split(test_size=0.2)

# ── Step 3: Load the pretrained model ────────────────────────
# distilbert-base-uncased already knows English.
# We add a classification head on top (2 outputs: ham or spam).
# Only the final layer needs to learn from scratch — everything
# else is already trained on millions of documents.
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2  # 2 classes: ham and spam
)

# ── Step 4: Define how to measure accuracy ───────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ── Step 5: Set training settings ────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",      # where to save checkpoints
    num_train_epochs=3,          # train for 3 rounds
    per_device_train_batch_size=16,
    evaluation_strategy="epoch", # check accuracy after every epoch
    logging_strategy="epoch"
)

# ── Step 6: Train ─────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

print("\nFine-tuning DistilBERT on spam data...\n")
trainer.train()

# ── Step 7: Try it yourself ───────────────────────────────────
import torch

print("\n--- Testing on custom messages ---")
samples = [
    "Congratulations! You've won a FREE prize, call now to claim!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your account will be suspended. Click here now.",
    "Can you pick up some milk on your way home?"
]

for text in samples:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    pred   = model(**inputs).logits.argmax(-1).item()
    print(f"  {'SPAM' if pred == 1 else 'HAM '} ← {text}")
