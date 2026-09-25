"""
Gradio interface for the Titanic survival predictor.

Loads the pipeline saved by train.py and turns passenger details entered
in the browser into a prediction + probability.
"""

import gradio as gr
import joblib
import pandas as pd

from validation import validate_passenger_input

MODEL_PATH = "model/titanic_model.pkl"

pipeline = joblib.load(MODEL_PATH)

PCLASS_LABELS = {"1st class": 1, "2nd class": 2, "3rd class": 3}
EMBARKED_LABELS = {
    "Southampton (S)": "S",
    "Cherbourg (C)": "C",
    "Queenstown (Q)": "Q",
}


def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    pclass_value = PCLASS_LABELS[pclass]
    embarked_value = EMBARKED_LABELS[embarked]

    result = validate_passenger_input(
        pclass=pclass_value,
        sex=sex,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare,
        embarked=embarked_value,
    )
    if not result.is_valid:
        raise gr.Error(" ".join(result.errors))

    row = pd.DataFrame(
        [
            {
                "Pclass": pclass_value,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked_value,
            }
        ]
    )

    proba = pipeline.predict_proba(row)[0]
    survival_probability = float(proba[1])
    label = "Likely survived" if survival_probability >= 0.5 else "Likely did not survive"

    return label, f"{survival_probability:.1%}"


with gr.Blocks(title="Titanic Survival Predictor") as demo:
    gr.Markdown(
        "# Titanic Survival Predictor\n"
        "Estimates whether a passenger would have survived, based on patterns "
        "a random forest model learned from the historical Titanic dataset.\n\n"
        "*Educational project — this reflects correlations in a historical "
        "dataset, not a factual judgment about any individual.*"
    )

    with gr.Row():
        with gr.Column():
            pclass = gr.Radio(
                list(PCLASS_LABELS.keys()), label="Passenger class", value="3rd class"
            )
            sex = gr.Radio(["male", "female"], label="Sex", value="male")
            age = gr.Slider(0, 80, value=30, step=1, label="Age")
            sibsp = gr.Slider(0, 8, value=0, step=1, label="Siblings / spouses aboard")
            parch = gr.Slider(0, 6, value=0, step=1, label="Parents / children aboard")
            fare = gr.Slider(0, 512, value=32, step=1, label="Fare paid")
            embarked = gr.Radio(
                list(EMBARKED_LABELS.keys()), label="Port of embarkation", value="Southampton (S)"
            )
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction")
            probability_output = gr.Textbox(label="Estimated survival probability")

    predict_btn.click(
        fn=predict_survival,
        inputs=[pclass, sex, age, sibsp, parch, fare, embarked],
        outputs=[prediction_output, probability_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8760)
