# Titanic Survival Predictor

A small end-to-end machine learning app: train a model on tabular data,
track the run, save the model, load it in a web app, and get interactive
predictions — with CI/CD shipping the containers automatically.

> Educational project. Predictions reflect patterns in a historical
> dataset, not a factual claim about any individual.

## How it works

```
data/titanic.csv
       |
       v
train.py -> clean/impute -> one-hot encode -> CV -> train random forest -> log to MLflow
       |
       v
model/titanic_model.pkl
       |
       v
validation.py -> reject unrealistic inputs
       |
       v
app.py (Gradio) or streamlit_app.py -> pipeline.predict_proba -> result in browser
```

Preprocessing and the classifier are bundled into one scikit-learn
`Pipeline`, so the exact same transformations used at training time are
applied automatically at prediction time.

## Inputs

| Input | Meaning | Valid range |
|---|---|---|
| Passenger class | Ticket class: 1st, 2nd, 3rd | 1–3 |
| Sex | male / female | — |
| Age | Age in years | 0–100 |
| Siblings / spouses aboard | SibSp | 0–10 |
| Parents / children aboard | Parch | 0–10 |
| Fare | Ticket fare paid | 0–600 |
| Port of embarkation | Southampton (S), Cherbourg (C), Queenstown (Q) | — |

Both UIs constrain these with sliders/radios, and `validation.py` rejects
out-of-range values a second time before they reach the model (see
`tests/test_train.py` for the validation test cases).

## Model

Random forest classifier (100 trees). Missing numeric values are filled
with the training median; missing categories with the most frequent
value; categorical fields are one-hot encoded.

- **5-fold stratified cross-validation accuracy:** ~80.2% (± 2.4%)
- **Held-out 20% test accuracy:** ~80.4%
- **Held-out test F1:** ~0.73

Cross-validation gives a more stable estimate than the single test split
alone; both are logged for every run.

## Experiment tracking (MLflow)

`train.py` logs each run to MLflow: hyperparameters, the git commit,
CV and test metrics, and the model artifact itself.

```bash
python train.py                 # logs to a local sqlite file, ./mlflow.db, by default
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs at http://localhost:5000
```

To point at a shared tracking server instead of the local sqlite file,
set `MLFLOW_TRACKING_URI` before training. The app itself never talks to
MLflow — it only loads the plain `model/titanic_model.pkl` file that
`train.py` writes alongside the MLflow run, so inference has no runtime
dependency on a tracking server being up.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python app.py                  # Gradio UI at http://localhost:8760
# or
streamlit run streamlit_app.py # Streamlit UI at http://localhost:8501
```

A trained model is already included, so retraining isn't required. To
retrain from `data/titanic.csv` (also logs an MLflow run):

```bash
pip install -r requirements-dev.txt
python train.py
```

## Tests and linting

```bash
pip install -r requirements-dev.txt
pytest -v        # trains a tiny in-memory pipeline + checks validation edge cases; no MLflow server needed
ruff check .      # lint
```

## Run with Docker

Gradio UI:

```bash
docker build -t titanic-gradio .
docker run --rm -p 8760:8760 titanic-gradio
```

Streamlit UI:

```bash
docker build -f Dockerfile.streamlit -t titanic-streamlit .
docker run --rm -p 8501:8501 titanic-streamlit
```

## Deploy to Kubernetes

`k8s/deployment.yaml` and `k8s/service.yaml` deploy the Streamlit image
(2 replicas, a `LoadBalancer` Service, and health probes against
Streamlit's built-in `/_stcore/health` endpoint), following the pattern
in [Streamlit's Kubernetes deployment guide](https://docs.streamlit.io/deploy/tutorials/kubernetes):

```bash
docker build -f Dockerfile.streamlit -t <your-registry>/titanic-streamlit:latest .
docker push <your-registry>/titanic-streamlit:latest
# update the image field in k8s/deployment.yaml to match, then:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## CI/CD

- **`.github/workflows/ci.yml`** — runs on every pull request and push to
  `main`: install deps, `ruff check`, `pytest`.
- **`.github/workflows/cd.yml`** — runs on every push to `main`: re-runs
  the tests, then builds and pushes both the Gradio and Streamlit images
  to Docker Hub, tagged `latest` and with the commit SHA. It ships
  whatever `model/titanic_model.pkl` is committed in the repo — training
  and promotion are a deliberate, separate, manual step (`python
  train.py`, review the MLflow run, commit the updated model) rather
  than something CD retrains automatically.

CD needs two repository secrets:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Read & Write) |

## Dataset

891 passenger records from Kaggle's *Titanic: Machine Learning from
Disaster* competition (Cukierski, Will. 2012.
https://www.kaggle.com/competitions/titanic).

## Project structure

```
.
├── app.py                    # Gradio UI + prediction logic
├── streamlit_app.py          # Streamlit UI + prediction logic
├── train.py                  # Preprocessing, CV, training, MLflow logging
├── validation.py             # Shared input validation for both UIs
├── data/titanic.csv          # Training data
├── model/titanic_model.pkl   # Serialized trained pipeline
├── tests/test_train.py       # Pipeline + validation tests (pytest)
├── k8s/
│   ├── deployment.yaml       # Kubernetes Deployment (Streamlit)
│   └── service.yaml          # Kubernetes Service (LoadBalancer)
├── .github/workflows/
│   ├── ci.yml                # Lint + test
│   └── cd.yml                # Build & push Docker images
├── requirements.txt          # Runtime deps (pinned)
├── requirements-dev.txt      # + mlflow, pytest, ruff
├── Dockerfile                # Gradio image
├── Dockerfile.streamlit      # Streamlit image
├── pyproject.toml            # Ruff config
└── README.md
```

## Possible next steps

- MLflow model registry with champion/challenger promotion (only
  promote a new run if it beats the current champion on a chosen metric)
- Push-based EC2/cloud deploy in `cd.yml`, or have the cluster poll the
  registry instead so CD never holds deploy credentials
- Model explainability (e.g. SHAP)
- Confusion matrix / precision-recall breakdown, not just accuracy and F1
