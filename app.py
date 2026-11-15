"""Streamlit interface for the supplied HSLS:09 dropout model.

Run: python -m streamlit run app.py
Keep dropout_model.pkl beside this file. Install requirements.txt first.

The pickle contains a fitted ColumnTransformer and LogisticRegression. We send
one row of original feature values to it, in its recorded feature order. We do
not fit, rescale, or one-hot encode in this application. Blank inputs become
NaN, which the saved imputers replace using their training statistics.

Target meaning confirmed by the model owner: 1 = dropout/high risk, 0 = lower
risk. Dropdown labels follow public NCES codebooks linked in the application.
This application assumes the original NCES category codes were not recoded.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.utils.validation import check_is_fitted


MODEL_PATH = Path(__file__).with_name("dropout_model.pkl")
# Small typography/layout adjustments complement .streamlit/config.toml.
# Use local system fonts so the design requires no external font service.
PAGE_STYLE = """
<style>
    .stApp { background-color: #FAF9F6; }
    .stMainBlockContainer {
        max-width: 1060px; padding-top: 3rem;
        border-top: 4px solid #217A75;
    }
    [data-testid="stHeading"] h1 {
        font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
        font-size: clamp(2.1rem, 4vw, 3rem);
        font-weight: 650; letter-spacing: -0.035em;
        color: #243E53; line-height: 1.15;
    }
    [data-testid="stHeading"] h3 {
        font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
        font-size: 1.3rem; font-weight: 600; color: #243E53;
        margin-top: .4rem;
    }
    .study-label {
        font-family: Arial, Helvetica, sans-serif;
        font-size: .72rem; font-weight: 600;
        letter-spacing: .14em; color: #217A75;
        margin: 0 0 .8rem 0;
    }
    [data-testid="stWidgetLabel"] p { font-size: .88rem; }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { color: #596975; }
    [data-testid="stForm"] { padding: 0; }
    [data-testid="stMetricValue"] {
        font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
        font-weight: 600; color: #217A75;
    }
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button { border-radius: 3px; }
    hr { border-color: #D2D6D7; }
    @media (max-width: 640px) {
        .stMainBlockContainer { padding-top: 1.5rem; }
    }
</style>
"""
EDUCATION = {
    1: "Less than high school", 2: "High school diploma / GED",
    3: "Begin associate degree", 4: "Finish associate degree",
    5: "Begin bachelor degree", 6: "Finish bachelor degree",
    7: "Begin master degree", 8: "Finish master degree",
    9: "Begin doctorate / professional degree",
    10: "Finish doctorate / professional degree", 11: "Respondent unsure",
}
FREQUENCY = {1: "Never", 2: "1–2 times", 3: "3–4 times", 4: "5+ times"}
CATEGORIES = {
    "X1STUEDEXPCT": EDUCATION, "X1PAREDEXPCT": EDUCATION,
    "P1SUSPEND": {0: "No", 1: "Yes"},
    "P1BEHAVE": FREQUENCY, "P1ATTEND": FREQUENCY,
    "X1IEPFLAG": {0: "No", 1: "Yes"},
    "X1PARPATTERN": {
        1: "Two biological/adoptive parents",
        2: "Biological/adoptive mother + another guardian",
        4: "Biological/adoptive father + another guardian",
        6: "Two other guardians", 7: "Biological/adoptive mother only",
        8: "Biological/adoptive father only", 9: "One other female guardian",
        10: "One other male guardian", 11: "With parents less than half the time",
    },
    "X1DUALLANG": {1: "English first", 2: "Another language first",
                    3: "English and another language together"},
    "X1RACE": {
        1: "American Indian / Alaska Native, non-Hispanic",
        2: "Asian, non-Hispanic", 3: "Black, non-Hispanic",
        4: "Hispanic, race unspecified", 5: "Hispanic, race specified",
        6: "Multiple races, non-Hispanic",
        7: "Native Hawaiian / Pacific Islander, non-Hispanic",
        8: "White, non-Hispanic",
    },
    "X1SEX": {1: "Male", 2: "Female"},
}
LABELS = {
    "X1SCHOOLENG": "School engagement score",
    "X1SCHOOLBEL": "School belonging score",
    "X1TXMTSCOR": "Mathematics standardized test score",
    "X1SES": "Socioeconomic composite score",
    "X1MTHEFF": "Mathematics self-efficacy score",
    "X1MTHINT": "Mathematics interest score",
    "X1MTHID": "Mathematics identity score",
    "X1MTHUTI": "Mathematics utility score",
    "X1SCIEFF": "Science self-efficacy score",
    "X1SCIINT": "Science interest score",
    "X1HHNUMBER": "Household members (11 means 11+)",
    "P1CHANGESCH": "School changes since kindergarten",
    "X3TGPA9TH": "Ninth-grade GPA",
    "X1STUEDEXPCT": "Student's expected education",
    "X1PAREDEXPCT": "Parent's expected education for student",
    "P1SUSPEND": "Ever suspended or expelled since kindergarten?",
    "P1BEHAVE": "Family contacted about behavior (prior school year)",
    "P1ATTEND": "Family contacted about attendance (prior school year)",
    "X1IEPFLAG": "Individualized Education Plan (IEP)?",
    "X1PARPATTERN": "Household parent / guardian arrangement",
    "X1DUALLANG": "First language background",
    "X1RACE": "NCES race / ethnicity category", "X1SEX": "NCES recorded sex",
}
GROUPS = {
    "Academic history and school experience": [
        "X3TGPA9TH", "X1TXMTSCOR", "X1SCHOOLENG", "X1SCHOOLBEL",
        "P1CHANGESCH", "P1SUSPEND", "P1BEHAVE", "P1ATTEND"],
    "Educational expectations and subject scores": [
        "X1STUEDEXPCT", "X1PAREDEXPCT", "X1MTHEFF", "X1MTHINT",
        "X1MTHID", "X1MTHUTI", "X1SCIEFF", "X1SCIINT"],
    "Household and student background": [
        "X1SES", "X1HHNUMBER", "X1PARPATTERN", "X1DUALLANG",
        "X1RACE", "X1SEX", "X1IEPFLAG"],
}


@st.cache_resource
def load_model():
    """Load only the developer-supplied pickle, once per server process."""
    if sklearn.__version__ != "1.6.1":
        raise RuntimeError("Install requirements.txt: this model needs scikit-learn 1.6.1.")
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    check_is_fitted(model.named_steps["model"])
    check_is_fitted(model.named_steps["preprocessor"])
    if set(model.feature_names_in_) != set(LABELS):
        raise ValueError("This model's inputs do not match this application.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected the confirmed class labels 0 and 1.")
    return model


def set_example(model):
    """Fill a synthetic example using training medians and categorical modes.

    These values do not describe a real student. Do not present them as a
    representative student or use them to measure model performance.
    """
    for name, branch, columns in model.named_steps["preprocessor"].transformers_:
        if name in {"numeric", "categorical"}:
            statistics = branch.named_steps["imputer"].statistics_
            for column, value in zip(columns, statistics):
                st.session_state[column] = float(value) if name == "numeric" else int(value)


def input_widget(column, allowed_categories):
    """Render raw numeric scores or a label that preserves the learned code."""
    label = LABELS[column]
    if column in CATEGORIES:
        options = [None] + [int(code) for code in allowed_categories[column]]
        value = st.selectbox(
            label, options, key=column,
            format_func=lambda code: "Not provided" if code is None
            else CATEGORIES[column].get(code, f"NCES code {code}"),
            help=f"{column}. The selected NCES numeric code goes to the saved pipeline.",
        )
    else:
        kwargs = {}
        help_text = f"{column}. Enter the NCES score in its original units; leave blank if unavailable."
        if column == "X3TGPA9TH":
            kwargs = {"min_value": 0.0, "max_value": 4.0, "step": 0.01}
        elif column == "P1CHANGESCH":
            kwargs = {"min_value": 0.0, "max_value": 9.0, "step": 1.0}
            help_text += " Exclude routine moves to the next school level."
        elif column == "X1HHNUMBER":
            kwargs = {"min_value": 2.0, "max_value": 11.0, "step": 1.0}
        value = st.number_input(label, value=None, key=column, help=help_text, **kwargs)
    return np.nan if value is None else float(value)


def main():
    """Collect inputs, predict once on submission, and explain missing inputs."""
    st.set_page_config(page_title="Dropout Risk Explorer", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.markdown('<p class="study-label">STUDENT OUTCOMES / HSLS:09</p>', unsafe_allow_html=True)
    st.title("Dropout Risk Explorer")
    st.write("A student profile, assessed through logistic regression.")
    st.caption("Enter the original NCES scores and select the applicable categories. Leave unavailable values blank.")
    try:
        model = load_model()
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    with st.expander("How to use"):
        st.button("Load synthetic example", key="load_example", on_click=set_example, args=(model,))
        st.caption("The example combines training medians and modes. It is not a real student.")
        st.write("Use original NCES scores. Composite scores are not questionnaire ratings. Missing inputs are filled by the saved model's imputers; enter a blank rather than a code such as -9.")

    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    category_columns = preprocessor.transformers_[1][2]
    allowed = dict(zip(category_columns, encoder.categories_))
    values = {}
    with st.form("student_profile", border=False):
        for number, (group, columns) in enumerate(GROUPS.items(), start=1):
            st.divider()
            st.subheader(f"{number:02d}  {group}")
            left, right = st.columns(2, gap="large")
            for index, column in enumerate(columns):
                with left if index % 2 == 0 else right:
                    values[column] = input_widget(column, allowed)
        st.divider()
        submitted = st.form_submit_button("Predict risk status", type="primary")

    if submitted:
        row = pd.DataFrame([values], columns=model.feature_names_in_, dtype=float)
        missing = int(row.isna().sum().sum())
        provided = len(values) - missing
        if provided == 0:
            st.warning("Enter at least one value, or load the synthetic example.")
        elif not np.isfinite(row.to_numpy()[~row.isna().to_numpy()]).all():
            st.error("Inputs must be finite numbers.")
        elif any(values[c] % 1 != 0 for c in ["P1CHANGESCH", "X1HHNUMBER"] if not np.isnan(values[c])):
            st.error("Household size and number of school changes must be whole numbers.")
        else:
            try:
                prediction = int(model.predict(row)[0])
                positive_index = list(model.classes_).index(1)
                score = float(model.predict_proba(row)[0, positive_index])
            except Exception as error:
                st.error(f"Prediction could not be completed: {error}")
            else:
                st.divider()
                st.subheader("Prediction result")
                st.write("**Higher dropout risk**" if prediction == 1 else "**Lower dropout risk**")
                st.caption(f"Predicted class {prediction}")
                # Follow the saved classifier's decision, rather than inventing
                # additional risk bands. Retain the text label for accessibility.
                score_color = "#94600C" if prediction == 1 else "#217A75"
                st.markdown(
                    f'<style>[data-testid="stMetricValue"] {{ color: {score_color}; }}</style>',
                    unsafe_allow_html=True,
                )
                st.metric("Model score for dropout (class 1)", f"{score:.1%}")
                st.caption("This is the model's predict_proba output, not a validated individual probability. The classifier uses balanced class weights.")
                st.write(f"{provided} of 23 inputs provided. {missing} filled by the saved training imputers.")
                if missing:
                    st.warning("Some inputs were imputed. Interpret this result with the missing information in mind.")
                with st.expander("View submitted values"):
                    review = row.T.rename(columns={0: "Value"})
                    review["Input"] = [LABELS[c] for c in review.index]
                    st.dataframe(review[["Input", "Value"]], width="stretch")

    with st.expander("About the model and data"):
        st.write("The saved pipeline applies median imputation and scaling to 13 numeric inputs, and mode imputation and one-hot encoding to 10 categorical inputs. Logistic regression then predicts class 0 or 1.")
        st.markdown("Input labels: [NCES base-year codebook](https://nces.ed.gov/surveys/hsls09/pdf/2011328_2.pdf), [follow-up codebook](https://nces.ed.gov/pubs2014/2014361_AppendixI.pdf), and [2013 update codebook](https://nces.ed.gov/pubs2015/2015036_student.pdf).")
        st.caption("Scores and category codes must match those used in training. This prototype does not collect names or save form submissions to a database.")


if __name__ == "__main__":
    main()
