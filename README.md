# Customer Churn Prediction: Multi-Algorithm Classification Showdown
Supervised Learning Project for Telecom Subscriber Retention (Jio / Airtel Case Study)

---

## Project Overview
Customer Churn occurs when subscribers discontinue their telecom services or switch to a competing provider. In the telecom industry (such as Jio or Airtel), Customer Acquisition Cost (CAC) is significantly higher than customer retention costs. Predicting churn enables the retention team to proactively intervene before subscribers leave, protecting overall Customer Lifetime Value (CLV).

This project uses the benchmark IBM Telco Customer Churn dataset (7,043 customers, 21 columns) to train, tune, and evaluate four Supervised Learning classification models:
* K-Nearest Neighbours (KNN)
* Naive Bayes (Gaussian NB)
* Support Vector Machine (SVM)
* Decision Tree Classifier

Key project workflows:
* Data cleaning: Handling whitespace entries in `TotalCharges` (filled with median).
* Exploratory Data Analysis (EDA): Univariate distributions and bivariate churn relationships.
* Feature engineering: `tenure_group`, `num_services`, and `AutoPay`.
* Class imbalance handling: Applying SMOTE on the training set only.
* Model evaluation: Focus on Recall for the Churn class to minimize costly False Negatives.
* Production pipeline: End-to-end scikit-learn Pipeline saved as `churn_model.pkl`.

---

## Dataset Link
* Dataset: IBM Telco Customer Churn Dataset
* Kaggle Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* File: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 columns)

---

## Video Section
* Video URL: [PASTE YOUR GOOGLE DRIVE OR UNLISTED YOUTUBE VIDEO LINK HERE]
* Format: MP4 or MOV (maximum 500 MB)
* Duration: 5 to 10 minutes
* Screen content: Full computer screen demonstrating `CustomerChurn_SupervisedLearning.ipynb` and `churn_model.pkl`
* Face visibility: Webcam picture-in-picture overlay visible throughout the full recording

---

## Model Comparison Results

All four models evaluated on the untouched 20% test set (1,409 customers), sorted by **Recall (descending)**:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | AUC-ROC | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbours (k=15)** | 0.7182 | 0.4817 | **0.8075** | 0.6034 | 0.8159 | 0.0030s |
| **Gaussian Naive Bayes** | 0.7239 | 0.4876 | **0.7914** | 0.6035 | 0.8165 | 0.0039s |
| **Decision Tree (depth=7)** | 0.7537 | 0.5249 | **0.7620** | 0.6216 | 0.8267 | 0.0180s |
| **SVM (RBF, C=100)** | 0.7275 | 0.4908 | **0.7139** | 0.5817 | 0.7886 | 14.1870s |

**Recommendation:** Deploy the Decision Tree Classifier. It achieves strong Recall (76.20%), the highest Precision (52.49%), and the highest F1-Score (0.6216). Its clear if-else decision rules allow frontline retention agents to understand and explain why a customer was flagged, and its sub-second inference latency supports real-time scoring.

---

## How to Run

### 1. Install Requirements
Ensure Python 3.10+ is installed, then install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open and run all cells in Jupyter Notebook:
```bash
jupyter notebook CustomerChurn_SupervisedLearning.ipynb
```

### 3. Test the Saved Pipeline
Verify the saved pipeline (`churn_model.pkl`) using Python:
```python
import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load('churn_model.pkl')

# Load sample data and predict
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
sample = df.head(5).drop(columns=['Churn'])

# Predict probability and class label
probs = pipeline.predict_proba(sample)[:, 1]
preds = pipeline.predict(sample)

for i in range(len(sample)):
    status = "Churn" if preds[i] == 1 else "No Churn"
    print(f"Customer {i+1}: Churn Probability = {probs[i]:.4f} | Label = {status}")
```

---

## Deliverables
* `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Benchmark IBM Telco dataset
* `CustomerChurn_SupervisedLearning.ipynb` - Fully executed notebook with all outputs and figures
* `churn_model.pkl` - Saved scikit-learn Pipeline
* `summary_report.md` - Executive summary report (~500 words)
* `requirements.txt` - Minimal dependencies
* `README.md` - Project documentation and video link
