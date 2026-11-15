# Dropout Risk Explorer

A documented Streamlit application for the supplied HSLS:09 logistic regression
pipeline. It accepts the model's 23 original features, preserves NCES category
codes behind readable dropdowns, and uses the fitted pipeline for all model
preprocessing and predictions.

## Run on your computer

Use Python 3.13. Open a terminal in this folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```



## What the app does

- One form groups academic history, subject scores/expectations, and background.
- Numeric fields start blank. Missing categorical fields also start unselected.
- Blank values become `numpy.nan`. The saved training imputers fill them.
- The synthetic example combines the fitted imputers' medians and modes.
  It is not an observed student, a representative profile, or a test set.
- A submission displays the saved classifier's class and class-1 model score.
- At least one input is required. Missing-input counts accompany predictions.

## Input meaning

Use original NCES composite values for engagement, belonging, subject attitudes,
SES, and the mathematics test score. Do not substitute everyday 1–5 ratings.
GPA uses a 0–4 scale. School changes exclude routine grade-level transitions.
For the public-use household-size variable, 11 means 11 or more people.
The dropdowns use the codes observed in the fitted one-hot encoder. Labels
assume you retained the original NCES category coding during training.
Do not enter numeric missing-value indicators such as -9; leave fields blank.
The app does not reproduce any external recoding used before model training.

Public references for these definitions and mappings:

- Base-year codebook: https://nces.ed.gov/surveys/hsls09/pdf/2011328_2.pdf
- Follow-up codebook: https://nces.ed.gov/pubs2014/2014361_AppendixI.pdf
- 2013 update codebook: https://nces.ed.gov/pubs2015/2015036_student.pdf
- Composite score construction: https://nces.ed.gov/surveys/hsls09/pdf/2011328_1.pdf

## Model and interpretation

The exported object is a fitted scikit-learn pipeline with steps `preprocessor`
and `model`. It contains 13 numeric and 10 categorical inputs, producing 69
transformed features. The classifier is LogisticRegression(C=1,
class_weight='balanced', max_iter=1000). Class 1 means
dropout/high risk and class 0 means lower risk.

The model was saved with scikit-learn 1.6.1. Other package versions in
requirements.txt are the versions used for local app verification, not a
reconstruction of every version used in the original training environment.


The displayed score is `predict_proba` for class 1.  Missing inputs increase dependence on training summaries.
The model includes demographic variables,
so use with students would require evaluation of subgroup performance and
appropriate human review.

