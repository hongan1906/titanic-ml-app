"""
Input validation shared by all front ends (Gradio, Streamlit).

Keeps unrealistic values (negative fares, 300-year-old passengers, etc.)
from reaching the model, and returns a plain-language error message the
UI can display instead of a raw exception.
"""

from dataclasses import dataclass

VALID_PCLASS = {1, 2, 3}
VALID_SEX = {"male", "female"}
VALID_EMBARKED = {"S", "C", "Q"}

AGE_RANGE = (0, 100)
SIBSP_RANGE = (0, 10)
PARCH_RANGE = (0, 10)
FARE_RANGE = (0, 600)  # historical max fare in the training data is ~512


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_passenger_input(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
) -> ValidationResult:
    errors = []

    if pclass not in VALID_PCLASS:
        errors.append(f"Passenger class must be one of {sorted(VALID_PCLASS)}.")

    if sex not in VALID_SEX:
        errors.append(f"Sex must be one of {sorted(VALID_SEX)}.")

    if not (AGE_RANGE[0] <= age <= AGE_RANGE[1]):
        errors.append(f"Age must be between {AGE_RANGE[0]} and {AGE_RANGE[1]}.")

    if not (SIBSP_RANGE[0] <= sibsp <= SIBSP_RANGE[1]):
        errors.append(
            f"Siblings/spouses aboard must be between {SIBSP_RANGE[0]} and {SIBSP_RANGE[1]}."
        )

    if not (PARCH_RANGE[0] <= parch <= PARCH_RANGE[1]):
        errors.append(
            f"Parents/children aboard must be between {PARCH_RANGE[0]} and {PARCH_RANGE[1]}."
        )

    if not (FARE_RANGE[0] <= fare <= FARE_RANGE[1]):
        errors.append(f"Fare must be between {FARE_RANGE[0]} and {FARE_RANGE[1]}.")

    if embarked not in VALID_EMBARKED:
        errors.append(f"Embarked must be one of {sorted(VALID_EMBARKED)}.")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
