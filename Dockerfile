FROM python:3.11-slim

WORKDIR /app

ARG HF_MODEL_NAME=qasudershan/mlops-imdb-sentiment
ENV HF_MODEL_NAME=$HF_MODEL_NAME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference.py .

CMD ["python", "inference.py"]