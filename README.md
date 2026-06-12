# End-to-End MLOps Pipeline for IMDb Sentiment Analysis

## Project Overview

This project implements a complete MLOps pipeline for sentiment analysis on the IMDb movie review dataset using DistilBERT. The solution covers data preparation, model training, experiment tracking, CI/CD automation, model deployment to Hugging Face, and Dockerized inference.

The objective is to demonstrate best practices in machine learning operations, including reproducibility, version control, automated testing, and deployment.

---

## Team Members

| Roll Number | Name               | Contribution                                              |
| ----------- | ------------------ | --------------------------------------------------------- |
| G25AIT2059  | Manish Kumar Singh | Docker, Hugging Face Deployment, Data Preparation, Report |
| G25AIT2114  | Sudershan Singh    | GitHub Setup, Kaggle Training, Model Training, Report     |

---

## Project Workflow

```text
IMDb Dataset
      │
      ▼
Data Preparation
      │
      ▼
DistilBERT Fine-Tuning
      │
      ▼
Weights & Biases Tracking
      │
      ▼
Hugging Face Deployment
      │
      ▼
Dockerized Inference
      │
      ▼
GitHub Actions CI/CD
```

---

## Dataset

Dataset Source:

https://huggingface.co/datasets/stanfordnlp/imdb

### Data Preprocessing

* Removed duplicate reviews
* Removed missing values
* Trimmed whitespace
* Removed empty records
* Sampled dataset for efficient training

### Final Dataset Size

| Dataset  | Samples |
| -------- | ------- |
| Training | 5,000   |
| Testing  | 1,000   |

### Label Mapping

```json
{
  "0": "NEGATIVE",
  "1": "POSITIVE"
}
```

---

## Model Selection

### DistilBERT

Model:

```text
distilbert-base-uncased
```

### Why DistilBERT?

* Lightweight transformer architecture
* Faster training and inference
* Approximately 97% of BERT performance
* Suitable for Kaggle GPU environment
* Lower deployment cost

---

## Training Configuration

| Parameter     | Value                   |
| ------------- | ----------------------- |
| Model         | distilbert-base-uncased |
| Epochs        | 4                       |
| Batch Size    | 32                      |
| Learning Rate | 5e-5                    |
| Max Length    | 128                     |

Training is executed on Kaggle GPU and tracked using Weights & Biases.

---

## Repository Structure

```text
mlops-group-project/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── inference.yml
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── inference.py
│
├── Dockerfile
├── requirements.txt
├── id2label.json
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/qasudershan/mlops-group-project.git
cd mlops-group-project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```cmd
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Generate cleaned datasets and label mappings:

```bash
python src/prepare_data.py
```

Output files:

```text
train.csv
test.csv
id2label.json
```

---

## Model Training

Training is intended to run on Kaggle GPU.

```bash
python src/train.py
```

Training logs and metrics are automatically tracked using Weights & Biases.

---

## Weights & Biases Dashboard

Track experiments and metrics:

https://wandb.ai/qasudershan-iit-jodhpur/mlops-assignment3

---

## Hugging Face Model

Published Model:

https://huggingface.co/qasudershan/mlops-imdb-sentiment

Example usage:

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="qasudershan/mlops-imdb-sentiment"
)

result = classifier("This movie was fantastic!")
print(result)
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t mlops-a3-inference:latest .
```

### Run Docker Container

```bash
docker run --rm \
-e INPUT_TEXT="This movie was amazing!" \
-e HF_TOKEN=<your_huggingface_token> \
mlops-a3-inference:latest
```

### Sample Output

```text
========================================
Result: POSITIVE
Confidence: 0.98
========================================
```

### Docker Hub Repository

https://hub.docker.com/r/qasudershan/mlops-a3-inference

---

## Continuous Integration

GitHub Actions automatically performs:

* Dependency installation
* Code validation
* Flake8 linting
* Pull request checks

### Workflow Triggers

```yaml
on:
  push:
    branches:
      - develop
      - master

  pull_request:
    branches:
      - master
```

---

## Dependencies

```text
transformers==4.44.0
torch==2.2.2
numpy<2
huggingface_hub
scikit-learn
datasets
wandb
flake8
```

---

## Project Resources

### GitHub Repository

https://github.com/qasudershan/mlops-group-project

### Hugging Face Model

https://huggingface.co/qasudershan/mlops-imdb-sentiment

### Docker Image

https://hub.docker.com/r/qasudershan/mlops-a3-inference

### Weights & Biases Dashboard

https://wandb.ai/qasudershan-iit-jodhpur/mlops-assignment3

---

## Future Enhancements

* Automated model deployment pipeline
* Model performance monitoring
* Automated retraining workflow
* REST API deployment using FastAPI
* Kubernetes deployment support

---

## License

This project is developed as part of the IIT Jodhpur MLOps Group Assignment.

