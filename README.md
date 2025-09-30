# 📊 Tabular Churn Prediction (Baseline)

An end-to-end machine learning pipeline that predicts **customer churn** (whether a customer is likely to leave a service).
Built with **scikit-learn + FastAPI** and using the **Telco Customer Churn dataset**.

---

## 🚀 Project Structure

```
tabular-churn-ml/
│
├── data/
│   ├── raw/         <- Original dataset (Telco churn CSV)
│   └── processed/   <- Train/val/test splits
│
├── models/          <- Saved trained model (churn_model.pkl)
│
├── notebooks/       <- Jupyter notebooks for exploration
│
├── src/
│   ├── download_data.py   # Download dataset
│   ├── data.py            # Split data into train/val/test
│   ├── train.py           # Train baseline model
│   └── app.py             # FastAPI app serving predictions
│
├── tests/           <- Unit tests (future)
│
├── requirements.txt <- Python dependencies
└── README.md        <- Project documentation
```

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone https://github.com/<your-username>/tabular-churn-ml.git
cd tabular-churn-ml
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Data

Download Telco Churn dataset from Kaggle automatically:

```bash
python3 src/download_data.py
```

Split into train/val/test:

```bash
python3 src/data.py
```

---

## 🏋️ Training

Train a baseline Logistic Regression model:

```bash
python3 src/train.py
```

This will:

* Preprocess numeric & categorical features
* Train a logistic regression model
* Print evaluation metrics (AUC, F1, precision, recall)
* Save model → `models/churn_model.pkl`

---

## 🌐 API (Serving Predictions)

Run FastAPI locally:

```bash
uvicorn src.app:app --reload
```

* Open docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example request (JSON):

```json
{
  "customerID": "1234-ABCD",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 445.0
}
```

Example response:

```json
{
  "prob": 0.734,
  "label": 1
}
```

---

## ✅ Definition of Done

* Data pipeline (`download_data.py`, `data.py`)
* Baseline churn model trained (`train.py`)
* Model served via FastAPI (`app.py`)
* Example request/response tested
* Documentation (`README.md`)

---

## 📌 Next Steps (Future Sprints)

* Add Streamlit/Gradio demo UI
* Add SHAP explainability (`top_factors`)
* Try tree-based models (XGBoost, LightGBM)
* Add Dockerfile for deployment
