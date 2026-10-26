# 🧩 Autism Spectrum Disorder (ASD) Prediction & Interactive Dashboard

A machine learning model and interactive web dashboard that predicts whether someone is likely to have Autism Spectrum Disorder (ASD), based on the AQ-10 behavioral screening questionnaire plus demographic indicators. Trained and compared 3 different models, tuned the best one, evaluated it on held-out test data, and deployed a production-grade Streamlit application.

**Best model:** Random Forest (Tuned) — **93% cross-validation accuracy**, **81.9% test accuracy**.

---

## 🌟 Interactive Streamlit Dashboard

The project includes a web dashboard built with Streamlit and Plotly:

- **🏠 Overview**: Executive summary with KPI metric cards, top-10 feature importance chart, confusion matrix, and class distributions.
- **🔮 Live Predictor**: Interactive AQ-10 questionnaire + demographic inputs with real-time inference, probability gauge, risk classification, and guidance.
- **📊 Data Explorer**: Deep-dive dataset exploration across 4 tabs — Raw Data, Distribution Histograms & Boxplots, Correlation Matrix Heatmap, and AQ-10 Question breakdowns.
- **🎯 Model Performance**: Comprehensive evaluation suite with Confusion Matrix, ROC Curve (AUC = 0.88), Precision-Recall Curve, and full Classification Report.
- **ℹ️ About**: Methodology details, pipeline overview, and future roadmap.

---

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

Recall on the ASD class is 64% — the model catches about 2 out of every 3 actual ASD cases in this test set. Given the class imbalance in the original data, that's the number worth watching more than overall accuracy.

---

## 📁 Repository Structure

```
├── .streamlit/
│   └── config.toml           → Dark theme, server, and headless settings
├── app.py                    → Streamlit dashboard application
├── best_model.pkl            → Pre-trained Random Forest model
├── encoders.pkl              → Fitted LabelEncoders for all categorical features
├── train.csv                 → Dataset (800 records, AQ-10 + demographics)
├── Autism_Preidiction_using_machine_Learning.ipynb → Model development notebook
├── requirements.txt          → Python package dependencies
├── Dockerfile                → Container build file for Docker / Cloud Run / HF
├── .dockerignore             → Files excluded from Docker image
├── Procfile                  → Process file for Heroku / Render / Railway
├── runtime.txt               → Python runtime configuration
├── .gitignore                → Git ignore rules
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone the repository and install dependencies
```bash
git clone <your-repo-url>
cd "ML-Powered Autism Screening Analysis"
pip install -r requirements.txt
```

### 2. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker

Build and run the containerized app with a single command:

```bash
# Build the Docker image
docker build -t asd-screening-app .

# Run the container
docker run -p 8501:8501 asd-screening-app
```
Then navigate to `http://localhost:8501`.

---

## ☁️ Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended — Free & 1-Click)
1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"**, select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **"Deploy!"**. Your dashboard will be live with a public URL in ~2 minutes.

### Option 2: Hugging Face Spaces (Free)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** (or **Docker**) as the SDK.
3. Push your repository or upload the files (`app.py`, `best_model.pkl`, `encoders.pkl`, `train.csv`, `requirements.txt`).
4. The space will automatically install dependencies and launch the app.

### Option 3: Render / Railway
- **Render**: Create a new **Web Service**, connect your GitHub repo, select **Python 3**, set Build Command to `pip install -r requirements.txt`, and Start Command to `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`.
- **Railway**: Simply link your repo; Railway automatically detects the `Procfile` and deploys the container.

---

## 🛠️ Built With

- **Framework**: Streamlit
- **Visualizations**: Plotly Express & Plotly Graph Objects
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Data Manipulation**: pandas, numpy
- **Containerization**: Docker

---

## ⚠️ Medical Disclaimer

This application is developed strictly for educational, informational, and academic research purposes. It is **not** a clinical diagnostic tool. Autism Spectrum Disorder (ASD) can only be diagnosed by licensed healthcare professionals and developmental specialists.
