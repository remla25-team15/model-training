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

---

## Setting up service account key

For team 15 members only.

We already have a service account called "remla-dvc-project" created on our Google Cloud project. Please login to the
common remlateam15@gmail.com account and check it out.

You don't need to create a new service account, you only need to add a new key for yourselves.
Please read the docs and only follow the part from "Select your service account and go to the Keys tab..."

You can read the DVC docs [here](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-service-accounts)

After you create the key, you can download it to `secrets/sa_key.json` (you may need to create a new folder called `secrets/`)

Make sure you have the file at that location. You can add it to your local dvc config:

```zsh
dvc remote modify myremote --local gdrive_service_account_json_file_path secrets/sa_key.json
```

The local config doesn't get pushed to GitHub and not does the `secrets/` folder.

## Training with Docker

### 1. Build the Docker Image

```bash
docker build -t model-trainer .
```

### 2. Run the Docker Container

```bash
docker run --rm -v $(pwd):/app \
  -e DVC_GDRIVE_SERVICE_ACCOUNT_JSON_FILE_PATH=/app/secrets/sa_key.json \
  model-trainer
```

This will generate trained models and save them in the output/ directory.

---

## Training without Docker

### 1. Install Dependencies

The project works with Python 3.10.

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 2. Run the DVC pipeline

```bash
dvc repro
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
