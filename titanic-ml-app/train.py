"""
Train a Titanic survival classifier, track the run with MLflow, and
persist the model as a single, reusable scikit-learn Pipeline.

By default MLflow logs to a local ./mlruns folder, so this runs with no
server required. Point it at a real tracking server by setting the
MLFLOW_TRACKING_URI environment variable before running.

Usage:
    python train.py
"""

import os
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/titanic.csv"
MODEL_PATH = "model/titanic_model.pkl"
RANDOM_STATE = 42
N_ESTIMATORS = 100
CV_FOLDS = 5

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"


def build_pipeline() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()

    # Cross-validation on the training split gives a more stable estimate
    # of generalization than a single train/test split alone.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"CV accuracy: {cv_scores.mean():.3%} (+/- {cv_scores.std():.3%}) over {CV_FOLDS} folds")
    print(f"Held-out test accuracy: {test_accuracy:.3%} on {len(y_test)} records")
    print(f"Held-out test F1: {test_f1:.3f}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("titanic-survival-predictor")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": N_ESTIMATORS,
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
                "test_size": 0.2,
                "features": ",".join(FEATURES),
            }
        )
        mlflow.set_tag("git_commit", get_git_commit())

        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")

        run_id = mlflow.active_run().info.run_id
        print(f"Logged MLflow run: {run_id}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
