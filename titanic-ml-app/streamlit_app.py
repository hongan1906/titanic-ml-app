"""
Streamlit interface for the Titanic survival predictor.

Same pipeline and validation logic as app.py (the Gradio version) —
this is the V2 UI layer recommended alongside Docker/Kubernetes
deployment (see README).
"""

import joblib
import pandas as pd
import streamlit as st

from validation import validate_passenger_input

MODEL_PATH = "model/titanic_model.pkl"

PCLASS_LABELS = {"1st class": 1, "2nd class": 2, "3rd class": 3}
EMBARKED_LABELS = {
    "Southampton (S)": "S",
    "Cherbourg (C)": "C",
    "Queenstown (Q)": "Q",
}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
    st.title("🚢 Titanic Survival Predictor")
    st.caption(
        "Estimates whether a passenger would have survived, based on patterns a "
        "random forest model learned from the historical Titanic dataset. "
        "Educational project — reflects correlations in historical data, not a "
        "factual judgment about any individual."
    )

    pipeline = load_pipeline()

    col1, col2 = st.columns(2)
    with col1:
        pclass_label = st.radio("Passenger class", list(PCLASS_LABELS.keys()), index=2)
        sex = st.radio("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 30)
        sibsp = st.slider("Siblings / spouses aboard", 0, 10, 0)
    with col2:
        parch = st.slider("Parents / children aboard", 0, 10, 0)
        fare = st.slider("Fare paid", 0, 600, 32)
        embarked_label = st.radio("Port of embarkation", list(EMBARKED_LABELS.keys()))

    if st.button("Predict", type="primary"):
        pclass = PCLASS_LABELS[pclass_label]
        embarked = EMBARKED_LABELS[embarked_label]

        result = validate_passenger_input(
            pclass=pclass, sex=sex, age=age, sibsp=sibsp, parch=parch, fare=fare, embarked=embarked
        )
        if not result.is_valid:
            for error in result.errors:
                st.error(error)
            return

        row = pd.DataFrame(
            [
                {
                    "Pclass": pclass,
                    "Sex": sex,
                    "Age": age,
                    "SibSp": sibsp,
                    "Parch": parch,
                    "Fare": fare,
                    "Embarked": embarked,
                }
            ]
        )
        proba = pipeline.predict_proba(row)[0]
        survival_probability = float(proba[1])
        label = "Likely survived" if survival_probability >= 0.5 else "Likely did not survive"

        st.metric("Prediction", label)
        st.metric("Estimated survival probability", f"{survival_probability:.1%}")


if __name__ == "__main__":
    main()
