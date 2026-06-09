from datasets import load_dataset
import pandas as pd
import json
import os

# ── 1. Load the IMDb dataset from HuggingFace ──────────────────
print("Loading dataset...")
dataset = load_dataset("stanfordnlp/imdb")

# ── 2. Convert to pandas DataFrames ───────────────────────────
train_df = pd.DataFrame(dataset['train'])
test_df  = pd.DataFrame(dataset['test'])

print(f"Original train size: {len(train_df)}")
print(f"Original test size:  {len(test_df)}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"Class distribution:\n{train_df['label'].value_counts()}")

# ── 3. Clean the data ──────────────────────────────────────────
# Remove duplicates
train_df = train_df.drop_duplicates(subset='text')
test_df  = test_df.drop_duplicates(subset='text')

# Remove rows with missing values
train_df = train_df.dropna(subset=['text', 'label'])
test_df  = test_df.dropna(subset=['text', 'label'])

# Basic text cleaning - strip leading/trailing whitespace
train_df['text'] = train_df['text'].str.strip()
test_df['text']  = test_df['text'].str.strip()

# Remove empty strings after stripping
train_df = train_df[train_df['text'].str.len() > 0]
test_df  = test_df[test_df['text'].str.len() > 0]

# ── 4. Sample to keep it small (Kaggle GPU limits) ─────────────
train_df = train_df.sample(5000, random_state=42)
test_df  = test_df.sample(1000, random_state=42)

print(f"\nAfter cleaning:")
print(f"Train size: {len(train_df)}")
print(f"Test size:  {len(test_df)}")
print(f"Class distribution:\n{train_df['label'].value_counts()}")

# ── 5. Save id2label mapping ───────────────────────────────────
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

with open("id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print("\nSaved id2label.json")

# ── 6. Save cleaned CSVs locally (DO NOT commit these) ─────────
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv",   index=False)

print("Saved train.csv and test.csv")
print("Done!")