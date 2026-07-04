# 💼 Job Salary Prediction using Multiple Linear Regression

Predicting employee **salary** based on experience, skillset, education, industry, company size, location, and remote-work status using a Multiple Linear Regression model trained on a large-scale dataset of 250,000 records.

This project is part of my ongoing **Machine Learning learning series**, focused on mastering end-to-end regression workflows on real-world, high-cardinality datasets.

---

## 📌 Problem Statement

Salary is influenced by a mix of numerical and categorical factors — years of experience, certifications, skills, and softer attributes like industry, location, and job title. This project builds a regression model that captures these relationships to predict salary, while handling high-cardinality categorical features at scale.

---

## 📊 Dataset

- **Source:** `job_salary_prediction_dataset.csv`
- **Rows:** 250,000
- **Features:**
  | Column | Type | Description |
  |---|---|---|
  | job_title | Categorical | 12 unique roles (e.g. Data Scientist, AI Engineer) |
  | experience_years | Numerical | Years of professional experience |
  | education_level | Categorical | High School, Diploma, Bachelor, PhD, etc. |
  | skills_count | Numerical | Number of skills listed |
  | certifications | Numerical | Number of certifications held |
  | industry | Categorical | 10 unique industries |
  | company_size | Categorical | Small, Medium, Large |
  | location | Categorical | 10 unique locations |
  | remote_work | Categorical | Remote/Hybrid/On-site |
  | **salary** | Numerical | Target variable |

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, Seaborn, Scikit-learn
- **Model:** Multiple Linear Regression
- **Preprocessing:** MinMaxScaler, OneHotEncoder

---

## 🔍 Project Workflow

1. **Data Loading & Exploration** — Loaded 250K records and inspected structure
2. **Feature-Target Split** — Separated independent features (`X`) and target (`salary`)
3. **Column Segregation** — Identified 3 numerical and 6 categorical columns
4. **Train-Test Split** — 80/20 split with `random_state=42`
5. **Outlier Detection** — Used boxplots to confirm no significant outliers were present
6. **Correlation Analysis** — Checked numerical feature relationships with salary
7. **Feature Scaling** — Applied `MinMaxScaler` on numerical columns
8. **Cardinality Check** — Verified unique value counts across all categorical columns before encoding
9. **Categorical Encoding** — Applied `OneHotEncoder` across 6 categorical columns, expanding to 45 encoded features
10. **Model Training** — Trained a `LinearRegression` model on the fully processed dataset
11. **Model Evaluation** — Assessed performance using R², MAE, and RMSE on both train and test sets

---

## 📈 Results

| Metric | Train Set | Test Set |
|---|---|---|
| **R² Score** | 0.9634 | 0.9635 |
| **MAE** | 5,470.96 | 5,436.10 |
| **RMSE** | 7,166.55 | 7,125.52 |

Train and test metrics are nearly identical, indicating the model generalizes well with no signs of overfitting — even at 250K rows and 45+ engineered features.

---

## 💡 Key Insights

- **Experience Years** had the strongest correlation with salary (0.44)
- **Skills Count** and **Certifications** contributed moderately (0.13 and 0.07)
- One-Hot Encoding of 6 categorical columns expanded the feature space from 9 → 48 columns
- Despite high dimensionality from encoding, the model remained stable and well-generalized across train/test splits

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/job-salary-prediction-linear-regression.git
cd job-salary-prediction-linear-regression
pip install -r requirements.txt
jupyter notebook jobSalPred.ipynb
```

---

## 📂 Repository Structure

```
Job-Salary-Prediction/
│
├── jobSalPred.ipynb
├── job_salary_prediction_dataset.csv
├── README.md
└── images/
```

---

---

## 🔗 Connect

If you found this project useful, feel free to ⭐ the repo and connect with me on [LinkedIn](#).

---

*Part of my Linear Regression learning series — building a strong foundation in ML through hands-on, real-world datasets.*