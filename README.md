# 🧩 Autism Spectrum Disorder (ASD) Prediction — Machine Learning

A machine learning model that predicts whether someone is likely to have Autism Spectrum Disorder, based on a short behavioral screening questionnaire plus a few demographic fields. Trained and compared 3 different models, tuned the best one, and evaluated it properly on unseen test data.

**Best model:** Random Forest — **93% cross-validation accuracy**, **81.9% accuracy on the held-out test set**.

---

## What this is actually doing

The dataset is based on the AQ-10 screening tool — 10 yes/no behavioral questions (`A1_Score` through `A10_Score`) that are commonly used as an initial autism screening step — plus age, gender, ethnicity, family history, and a few other fields. The target is whether the person was ultimately diagnosed with ASD (`Class/ASD`).

This is **not** a diagnostic tool — real ASD diagnosis needs a clinician. What this project demonstrates is the full ML workflow around a real, moderately messy healthcare dataset: cleaning it, handling class imbalance properly, comparing models fairly, tuning the winner, and reporting honest test performance instead of just cross-validation numbers.

## 📊 Results

| Step | Result |
|---|---|
| Decision Tree (default params, 5-fold CV) | 86% accuracy |
| Random Forest (default params, 5-fold CV) | 92% accuracy |
| XGBoost (default params, 5-fold CV) | 90% accuracy |
| **Random Forest, tuned (RandomizedSearchCV)** | **93% CV accuracy** ← best model |
| **Final model on held-out test set** | **81.9% accuracy** |

Test set confusion matrix (160 people held out, never seen during training):

| | Predicted: No ASD | Predicted: ASD |
|---|---|---|
| **Actual: No ASD** | 108 | 16 |
| **Actual: ASD** | 13 | 23 |

The gap between the 93% CV score and 81.9% test score is the honest part — it's a good reminder that cross-validation accuracy (measured after oversampling with SMOTE) is optimistic compared to real, untouched test data. I reported both instead of just leading with the bigger number.

Recall on the ASD class is 64% — the model catches about 2 out of every 3 actual ASD cases in this test set. Given the class imbalance in the original data, that's the number worth watching more than overall accuracy.

## 🔧 What I actually did

1. **Cleaned the data** — fixed inconsistent country names (e.g. "Viet Nam" → "Vietnam"), handled `?` placeholder values in `ethnicity` and `relation`, dropped an `age_desc` column that only had 1 unique value, converted `age` to integer
2. **Explored it** — distribution plots, boxplots for outliers, count plots for every categorical field, a full correlation heatmap
3. **Handled outliers** — replaced age/result outliers (found via IQR method) with the median rather than dropping rows
4. **Encoded categoricals** — label-encoded every text column, saved the encoders (`encoders.pkl`) so the same transformations can be reused on new data
5. **Fixed class imbalance** — the target classes were imbalanced, so applied **SMOTE** (oversampling) on the training set only, after the train/test split — to avoid leaking synthetic data into the test set
6. **Compared 3 models fairly** — Decision Tree, Random Forest, and XGBoost, all with 5-fold cross-validation
7. **Tuned the winner** — `RandomizedSearchCV` across all 3 models' hyperparameters, picked whichever came out on top (Random Forest)
8. **Evaluated honestly** — final accuracy, confusion matrix, and full classification report on the untouched test set

## 🖼️ Screenshots

**Age distribution** — most people in the dataset are younger (teens to late 20s), with a long tail out to 80+, which is part of why age outliers got capped rather than dropped:

![Age distribution](screenshots/age-distribution.png)

**Correlation heatmap** — none of the features are dangerously correlated with each other, but a few of the AQ-10 questions (A6, A5, A4) correlate reasonably well with the actual ASD outcome, which is a good sign the screening questions are doing their job:

![Correlation heatmap](screenshots/correlation-heatmap.png)

**Final test set evaluation** — accuracy, confusion matrix, and full classification report on data the model never saw during training or tuning:

![Classification report](screenshots/classification-report.png)

## 📁 What's in this repo

```
├── autism_prediction.ipynb   → the full notebook, EDA through evaluation
├── train.csv                  → the dataset (800 records, AQ-10 + demographics)
├── requirements.txt
├── screenshots/
└── README.md
```

## 🚀 How to run it

```bash
pip install -r requirements.txt
jupyter notebook autism_prediction.ipynb
```

One thing to fix before running: the notebook reads the data from a Colab path (`/content/train.csv`). If you're running it locally, change that line to just `"train.csv"` (assuming the CSV sits next to the notebook).

## 🛠️ Built with

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · seaborn / matplotlib

## 💡 What I'd add next

- Try SHAP values to explain individual predictions — which questionnaire answers drove a specific prediction
- Push recall on the ASD class higher, even at some cost to precision, since a missed case matters more than a false positive in a screening context
- Wrap the trained model + encoders into a small Streamlit app for a live predictive demo

## 👤 About

Built by Gurvinder — mining engineering background, building out ML and data science skills.
