import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

MODEL_NAME = os.environ.get("HF_MODEL_NAME", "Manishrepo-bi/mlops-imdb-sentiment")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "This movie was absolutely fantastic!")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

print(f"Loading model: {MODEL_NAME}")
print(f"Input text:    {INPUT_TEXT}")

# Load tokenizer + model explicitly so we can control the inputs
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# DistilBERT does NOT use token_type_ids — force the tokenizer to skip them
tokenizer.model_input_names = ["input_ids", "attention_mask"]

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
)

result = classifier(INPUT_TEXT)
label  = result[0]["label"]
score  = result[0]["score"]

print(f"\n{'='*40}")
print(f"Result:     {label}")
print(f"Confidence: {score:.4f}")
print(f"{'='*40}")