import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = os.environ.get("HF_MODEL_NAME", "qasudershan/mlops-imdb-sentiment")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "This movie was absolutely fantastic!")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

print(f"Loading model: {MODEL_NAME}")
print(f"Input text:    {INPUT_TEXT}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)

inputs = tokenizer(
    INPUT_TEXT,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# DistilBERT doesn't use token_type_ids
inputs.pop("token_type_ids", None)

import torch

outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

pred = torch.argmax(probs, dim=-1).item()

score = probs[0][pred].item()

label = model.config.id2label[pred]

print("=" * 40)
print(f"Result:     {label}")
print(f"Confidence: {score:.4f}")
print(f"{'='*40}")