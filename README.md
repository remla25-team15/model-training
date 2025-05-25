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

## Setting up service account key

For team 15 members only.

We use Google Drive as DVC remote, specifically the [remla_resources](https://drive.google.com/drive/folders/1bxRbOXRah2lb_E8Ec6X6yDO9vYUj5OFT?usp=sharing) folder.

You need to be logged in to the remlateam15 google account to access this directly.

This should already be present in our `.dvc/config` but if it isn't, you can add the remote storage to your dvc config using:

```zsh
dvc remote add myremote gdrive://1bxRbOXRah2lb_E8Ec6X6yDO9vYUj5OFT
```

Set this remote as your default remote for future push and pull operations: (again, this should already be present in the config)

```zsh
dvc remote default myremote
```

To access the remote storage using dvc, you need to set up an API key. Our preferred way of doing this
is using a service account.

We already have a service account called "remla-dvc-project" created on our Google Cloud project called "remla-team-15". Please login to the
common remlateam15@gmail.com account and check it out on the Google Cloud console dashboard [https://console.cloud.google.com/](https://console.cloud.google.com/).

You **DO NOT** need to create a new service account, you only need to add a new key for yourselves.
Please read the docs and only follow the part from "Select your service account and go to the Keys tab..."

You can read the DVC docs [here](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-service-accounts)

After you create the key, you can download it to `secrets/sa_key.json` (you may
need to create a new folder called `secrets/` at the root of this project.)

Make sure you have the file at that location.

(Optional)
You can also add the key to your local dvc config:

```zsh
dvc remote modify myremote --local gdrive_service_account_json_file_path secrets/sa_key.json
```

The local config doesn't get pushed to GitHub and nor does the `secrets/` folder.

If everything went correctly, you should be able to pull from dvc remote using:

```zsh
dvc pull
```

---

## Linting (TODO)

For now, there is a non-trivial pylintrc file but it needs to be improved to catch ML Specific code smells (refer Assignment 4 Code Quality excellent section).

You can run pylint right now:

```zsh
pylint src/
```

We also need to add more linters like flake8 and Bandit.

---

## Testing
There are some tests in the `tests/` directory following the ML Test Score methodology.
You can run the tests using:

```bash
pytest tests/
```
You can also run the tests with coverage:

```bash
pytest --cov=src tests/
```

Check cyclomatic complexity
```bash
radon cc src/ -s -a
```

--

## Training

### 1. Install Dependencies

The project works with Python 3.10.

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 2. Run the DVC pipeline

Pull from dvc remote to get the latest experiments

```bash
dvc pull
```

Run the pipeline

```bash
dvc repro
```

## Running experiments with DVC

We use **DVC experiments** to manage and track machine learning experiments.

To run the pipeline and capture experimental results, use:

```zsh
dvc exp run
```

This command executes the pipeline defined in `dvc.yaml` using the parameters from `params.yaml`.
It tracks outputs, metrics, and changes, letting you iterate quickly without committing to Git every time.

To view and compare experiments:

```zsh
dvc exp show
```

You can modify hyperparameters in `params.yaml` and rerun `dvc exp run` to test different configurations.
DVC will log each run as a separate experiment that you can compare and manage.

You can also do this through the CLI:

```zsh
dvc exp run -S train.random_state=45
```

## Pushing to remote

If you have everything setup correctly, you should also be able to push to the remote storage by running:

```zsh
dvc push
```

---

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

## Output

After successful training, the following files are saved in the `output/` directory:

- `c1_BoW_Sentiment_Model.pkl`: Trained CountVectorizer BoW model
- `c2_Classifier_Sentiment_Model.pkl`: Trained classifier model

---

## GitHub Actions

The workflow in `.github/workflows/train.yml`:

- Runs on `push` or `pull_request` to `main`
- Runs the dvc pipeline
- Uploads the trained models as an artifact

## Dependencies

Dependencies are defined in `requirements.txt` and installed inside the Docker image automatically during the build process.
