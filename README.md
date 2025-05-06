# Restaurant Sentiment Analysis – Model Training

This repository contains the code to train a sentiment analysis model on restaurant reviews using a machine learning pipeline. 
It includes Docker support for reproducibility and a GitHub Actions workflow to automate training and artifact generation.

---

## Features

- Trains a sentiment analysis model on TSV datasets  
- Uses `nltk`, `sklearn` 
- Dockerized
- GitHub Actions workflow to automate model training and upload artifacts  

---

## Training with Docker 

### 1. Build the Docker Image

```bash
docker build -t model-trainer .
```

### 2. Run the Docker Container

```bash
docker run --rm model-trainer
```

This will generate trained models and save them in the output/ directory.

---
## Training without Docker
### 1. Install Dependencies
The project works with Python 3.10.
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script

```bash
cd src
python train_model.py
```

## Output
After successful training, the following files are saved in the `output/` directory:

- `c1_BoW_Sentiment_Model.pkl`: Trained CountVectorizer BoW model
- `c2_Classifier_Sentiment_Model.pkl`: Trained classifier model

---

## GitHub Actions

The workflow in `.github/workflows/train.yml`:

- Runs on `push` or `pull_request` to `main`  
- Builds and runs the Docker container  
- Uploads the trained models as an artifact  

## Dependencies
Dependencies are defined in `requirements.txt` and installed inside the Docker image automatically during the build process.

