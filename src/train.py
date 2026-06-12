# NOTE: This script is for reference only.
# Actual training must be run on Kaggle Notebooks with GPU.
# See: https://www.kaggle.com/ for the live notebook.

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import wandb
import json

# ── Config ─────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
RUN_NAME = "run-v2"   

# ── Load id2label ───────────────────────────────────────────────
with open("id2label.json") as f:
    id2label = json.load(f)
label2id = {v: k for k, v in id2label.items()}

# ── Load tokenizer and model ────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# ── Load cleaned data ───────────────────────────────────────────
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ── Tokenize ────────────────────────────────────────────────────


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize,  batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
train_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(
    "torch",  columns=["input_ids", "attention_mask", "labels"])

# ── Metrics ─────────────────────────────────────────────────────


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


# ── W&B init ────────────────────────────────────────────────────
wandb.init(
    project="mlops-assignment3",
    name=RUN_NAME,
    config={
        "model":         MODEL_NAME,
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "version":       RUN_NAME,
        "platform":      "Kaggle",
    }
)

# ── Training arguments ──────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name=RUN_NAME,
    logging_steps=50,
)

# ── Train ───────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
wandb.finish()
