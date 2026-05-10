# 💳 Task 2: Credit Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> A machine-learning project that predicts whether a loan applicant will be **approved** or **denied / default** using demographic and financial features.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Objective](#-objective)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Installation & Setup](#%EF%B8%8F-installation--setup)
6. [How to Run](#-how-to-run)
7. [Workflow](#-workflow)
8. [Exploratory Data Analysis](#-exploratory-data-analysis)
9. [Modelling](#-modelling)
10. [Results](#-results)
11. [Key Insights](#-key-insights)
12. [Skills Demonstrated](#-skills-demonstrated)
13. [Future Improvements](#-future-improvements)
14. [Author](#-author)

---

## 🎯 Project Overview

Lending institutions face a fundamental question every day: **"Will this applicant repay the loan?"**
Wrong decisions are costly — approving a bad loan leads to default losses, while denying a good loan loses revenue and customer goodwill.

This project builds a **binary classification model** that learns from historical loan applications and predicts loan approval status (`Y` = approved / repaid, `N` = denied / defaulted).

---

## 🎯 Objective

- Train a model to **predict loan approval status** based on applicant attributes.
- Identify the **key features** that drive the approval decision.
- Compare the performance of two interpretable classifiers — **Logistic Regression** vs **Decision Tree**.
- Evaluate performance using **accuracy** and a **confusion matrix**.

---

## 📊 Dataset

**Source:** [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) — Analytics Vidhya / Kaggle

| Property | Value |
|---|---|
| Rows | 614 |
| Columns | 13 |
| Target | `Loan_Status` (binary: `Y` / `N`) |
| Type | Tabular, mixed (numeric + categorical) |

### Feature Dictionary

| Feature | Type | Description |
|---|---|---|
| `Loan_ID` | ID | Unique identifier (dropped during modelling) |
| `Gender` | Categorical | `Male` / `Female` |
| `Married` | Categorical | `Yes` / `No` |
| `Dependents` | Categorical | `0`, `1`, `2`, `3+` |
| `Education` | Categorical | `Graduate` / `Not Graduate` |
| `Self_Employed` | Categorical | `Yes` / `No` |
| `ApplicantIncome` | Numeric | Applicant's monthly income |
| `CoapplicantIncome` | Numeric | Co-applicant's monthly income |
| `LoanAmount` | Numeric | Loan amount in thousands |
| `Loan_Amount_Term` | Numeric | Loan term in months |
| `Credit_History` | Binary | 1 = good credit, 0 = bad credit |
| `Property_Area` | Categorical | `Urban` / `Semiurban` / `Rural` |
| **`Loan_Status`** | **Target** | `Y` = approved, `N` = denied |

---

## 📂 Project Structure

```
Task_2_Credit_Risk/
│
├── 📓 Task_2_Credit_Risk.ipynb     # Main notebook with full analysis
├── 📊 loan_data.csv                # Loan Prediction dataset (614 rows)
├── 📄 README.md                    # You are here
└── 📋 requirements.txt             # Python dependencies (optional)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip / conda

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or use a `requirements.txt`:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter
```

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Launch Jupyter
jupyter notebook Task_2_Credit_Risk.ipynb
```

Then **Run All Cells** ( `Cell → Run All` ).
Make sure `loan_data.csv` is in the same directory as the notebook.

---

## 🔄 Workflow

```
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   1. Load Data    │ ─► │  2. Handle Nulls  │ ─► │   3. EDA + Plots  │
└───────────────────┘    └───────────────────┘    └─────────┬─────────┘
                                                            │
┌───────────────────┐    ┌───────────────────┐    ┌─────────▼─────────┐
│  6. Compare &     │ ◄─ │ 5. Train Models   │ ◄─ │  4. Encode + Split│
│    Conclude       │    │ (LR + Decision Tr)│    │     (80 / 20)     │
└───────────────────┘    └───────────────────┘    └───────────────────┘
```

### Pipeline steps
1. **Load** the `loan_data.csv` file with pandas.
2. **Handle missing values** — categorical columns filled with the **mode**, numeric with the **median**.
3. **Explore the data** — distributions, target balance, key feature relationships.
4. **Encode categoricals** with `LabelEncoder` and split the data 80 / 20 (stratified on the target).
5. **Train two models**: Logistic Regression and Decision Tree (`max_depth=5`).
6. **Evaluate** with accuracy, classification report, and confusion matrix.

---

## 📈 Exploratory Data Analysis

Highlights from the EDA section of the notebook:

- **Class imbalance:** approval rate ≈ **77 %** — the dataset is moderately imbalanced toward approvals.
- **Missing values** present in `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History` — all imputed before modelling.
- **Credit history** stands out as the single strongest visual predictor — good-credit applicants are approved at a dramatically higher rate.
- **Property area** matters: *Semiurban* properties have the highest approval rate, *Rural* the lowest.
- **Income** alone is a surprisingly weak signal — many high-income applicants are denied if other criteria fail.

Visualisations included:
- Bar chart of missing values
- Target distribution (`Loan_Status`)
- Loan amount distribution + box plot by status
- `Education` vs `Loan_Status` count plot
- `ApplicantIncome` box plot by status
- `Credit_History` vs `Loan_Status` count plot
- `Property_Area` vs `Loan_Status` count plot

---

## 🧠 Modelling

| Step | Detail |
|---|---|
| Encoding | `LabelEncoder` on every categorical column |
| Split | `train_test_split(test_size=0.2, random_state=42, stratify=y)` |
| Model 1 | `LogisticRegression(max_iter=1000, random_state=42)` |
| Model 2 | `DecisionTreeClassifier(max_depth=5, random_state=42)` |
| Metrics | `accuracy_score`, `confusion_matrix`, `classification_report` |

---

## 🏆 Results

| Model | Accuracy | Notes |
|---|:---:|---|
| **Logistic Regression** | **≈ 0.87** | Linear, highly interpretable via coefficients |
| **Decision Tree (depth 5)** | **≈ 0.87** | Captures non-linear splits, easy to visualise |

> Both models score similarly because *Credit_History* is so dominant that even a simple linear boundary captures most of the signal.

### Confusion matrix interpretation

|  | Predicted N | Predicted Y |
|---|---|---|
| **Actual N** | True Negatives | False Positives ❗ (risky — bad loans approved) |
| **Actual Y** | False Negatives (lost revenue) | True Positives |

The notebook prints the classification report (precision, recall, F1) for both classes so you can inspect the trade-off directly.

---

## 💡 Key Insights

1. **`Credit_History` is the dominant predictor.** Applicants with good credit history are approved overwhelmingly more often — this is the single most important variable for the lender.
2. **`Property_Area = Semiurban`** has the highest approval rate, *Rural* the lowest.
3. **Income alone is a weak predictor** — combining it with `LoanAmount` (debt-to-income) is more informative.
4. **Education** has only a small marginal effect once other variables are controlled for.
5. The dataset's class imbalance (~77 % approvals) means **accuracy alone is not sufficient** — recall on the **denied** class is the more business-critical metric.

---

## 🛠 Skills Demonstrated

- 🐍 **Python** — pandas, NumPy
- 📊 **Data visualisation** — matplotlib, seaborn
- 🧹 **Data cleaning** — missing-value imputation strategies
- 🔢 **Feature encoding** — `LabelEncoder`
- 🧠 **Machine learning** — Logistic Regression, Decision Trees
- 📐 **Model evaluation** — accuracy, confusion matrix, classification report
- 🧪 **Train/test methodology** — stratified splitting
- 📝 **Communication** — interpreting results in business language

---

## 🚀 Future Improvements

- ➕ Add **feature engineering**: total income (`ApplicantIncome + CoapplicantIncome`), loan-to-income ratio, EMI estimate.
- 🌲 Try **ensemble models**: Random Forest, Gradient Boosting, XGBoost.
- ⚖️ Handle **class imbalance** with `class_weight='balanced'`, SMOTE, or threshold tuning.
- 🔍 **Hyperparameter tuning** with `GridSearchCV` / `RandomizedSearchCV`.
- 📈 Add **ROC-AUC** and Precision-Recall curves for a fuller picture.
- 🧪 Cross-validation instead of a single train/test split.
- 🌐 Deploy as a small **Streamlit / Flask app** that scores a new applicant.
- 🔬 Use **SHAP** values for per-applicant explanation.

---

## 👤 Author

- Mohammad Ahzam Hassan

---

## 🙏 Acknowledgements

- **Analytics Vidhya / Kaggle** — for the original Loan Prediction dataset.
- **scikit-learn** — for the modelling and evaluation tools used throughout.

---

<p align="center">⭐ If you found this project helpful, consider giving it a star! ⭐</p>
