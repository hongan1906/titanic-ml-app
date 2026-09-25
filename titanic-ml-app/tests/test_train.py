"""
Fast, hermetic tests: train a small pipeline directly on a tiny sample
instead of hitting a real MLflow server or the full dataset.
"""

import pandas as pd
import pytest

from train import build_pipeline
from validation import validate_passenger_input


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "Pclass": [1, 3, 2, 3, 1, 2, 3, 1],
            "Sex": ["female", "male", "female", "male", "male", "female", "male", "female"],
            "Age": [29, 22, 35, None, 54, 4, 28, 38],
            "SibSp": [0, 1, 0, 3, 0, 1, 0, 1],
            "Parch": [0, 0, 0, 1, 0, 2, 0, 0],
            "Fare": [211.3, 7.25, 26.0, 21.07, 51.86, 16.7, 8.05, 71.28],
            "Embarked": ["S", "S", "S", "S", "S", "S", None, "C"],
        }
    )


@pytest.fixture
def sample_target():
    return pd.Series([1, 0, 1, 0, 0, 1, 0, 1])


def test_pipeline_fits_and_predicts(sample_data, sample_target):
    pipeline = build_pipeline()
    pipeline.fit(sample_data, sample_target)

    predictions = pipeline.predict(sample_data)
    probabilities = pipeline.predict_proba(sample_data)

    assert len(predictions) == len(sample_data)
    assert probabilities.shape == (len(sample_data), 2)


def test_pipeline_handles_missing_values(sample_data, sample_target):
    # Row 3 has a missing Age and row 6 has a missing Embarked; the
    # pipeline's imputers should handle both without raising.
    pipeline = build_pipeline()
    pipeline.fit(sample_data, sample_target)
    pipeline.predict(sample_data)


def test_validate_passenger_input_accepts_valid_values():
    result = validate_passenger_input(
        pclass=1, sex="female", age=29, sibsp=0, parch=0, fare=100, embarked="S"
    )
    assert result.is_valid
    assert result.errors == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("pclass", 5),
        ("sex", "unknown"),
        ("age", 150),
        ("age", -1),
        ("sibsp", -1),
        ("parch", 20),
        ("fare", -10),
        ("embarked", "X"),
    ],
)
def test_validate_passenger_input_rejects_invalid_values(field, value):
    kwargs = {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 100,
        "embarked": "S",
    }
    kwargs[field] = value

    result = validate_passenger_input(**kwargs)

    assert not result.is_valid
    assert len(result.errors) == 1
