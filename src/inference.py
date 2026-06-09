import os
from transformers import pipeline

# These come from environment variables (set via Docker or GitHub Actions)
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "qasudershan/mlops-imdb-sentiment")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "This movie was absolutely fantastic!")
HF_TOKEN   = os.environ.get("HF_TOKEN", None)

print(f"Loading model: {MODEL_NAME}")
print(f"Input text:    {INPUT_TEXT}")

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    token=HF_TOKEN
)

result = classifier(INPUT_TEXT)
label  = result[0]["label"]
score  = result[0]["score"]

print(f"\n{'='*40}")
print(f"Result:     {label}")
print(f"Confidence: {score:.4f}")
print(f"{'='*40}")
