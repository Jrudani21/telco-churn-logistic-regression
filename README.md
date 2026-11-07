# 📞 Will They Stay or Will They Go?  Telco Customer Churn Analysis

> **Can we predict which customers will cancel their telecom subscription  and why?**  
> Logistic regression in both R and Python on 7,043 customer records.

Plots are written to `figures/` when you run the scripts; they are not committed.

---

## 📋 Table of Contents
- [Problem](#problem)
- [Data](#data)
- [Methods](#methods)
- [Key Results](#key-results)
- [Business Insights](#business-insights)
- [How to Reproduce](#how-to-reproduce)
- [What I'd Do Next](#what-id-do-next)

---

## Problem

Customer churn is one of the most costly problems in subscription businesses. This project builds a **logistic regression model** to:

1. Predict whether a customer will churn (binary outcome: Yes / No)
2. Identify the strongest drivers of churn using odds ratios
3. Evaluate model performance using AUC, confusion matrix, and classification report
4. Deliver actionable business recommendations

The same analysis is done in **both R and Python**  comparing coefficients across both implementations.

---

## Data

| Property | Detail |
|---|---|
| **Source** | [Kaggle  Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Size** | 7,043 customers × 21 columns (11 blanks dropped → 7,032 clean rows) |
| **Target** | `Churn`  Yes (26.6%) / No (73.4%) |
| **License** | IBM Sample Data via Kaggle |

### Key Variables

| Variable | Type | Description |
|---|---|---|
| `tenure` | numeric | Months as a customer |
| `MonthlyCharges` | numeric | Monthly bill ($) |
| `TotalCharges` | numeric | Total amount billed ($) |
| `Contract` | factor | Month-to-month / One year / Two year |
| `InternetService` | factor | DSL / Fiber optic / No |
| `TechSupport` | factor | Yes / No / No internet |
| `OnlineSecurity` | factor | Yes / No / No internet |
| `PaperlessBilling` | factor | Yes / No |
| `SeniorCitizen` | binary | 1 = senior citizen |
| `Churn` | **target** | Yes / No |

---

## Methods

### R (`churn_analysis.R`)
| Step | Function | Package |
|---|---|---|
| Data cleaning | `drop_na()`, `as.numeric()` | tidyverse |
| EDA | `ggplot2` bar + box + histogram | ggplot2 |
| Train/test split | `createDataPartition()` | caret |
| Logistic regression | `glm(family = binomial)` | base R |
| Odds ratios + CI | `tidy(exponentiate = TRUE)` | broom |
| Confusion matrix | `confusionMatrix()` | caret |
| ROC / AUC | `roc()`, `auc()` | pROC |
| Hypothesis test | `Anova(type = "II")` | car |

### Python (`churn_analysis.py`)
| Step | Function | Package |
|---|---|---|
| Data cleaning | `pd.to_numeric(errors="coerce")` | pandas |
| EDA | `boxplot`, `hist` | matplotlib |
| Preprocessing | `ColumnTransformer` + `StandardScaler` + `OneHotEncoder` | scikit-learn |
| Logistic regression | `LogisticRegression(class_weight="balanced")` | scikit-learn |
| Cross-validation | `cross_val_score(scoring="roc_auc")` | scikit-learn |
| ROC curve | `RocCurveDisplay` | scikit-learn |
| Odds ratios | `sm.Logit().fit()` + `np.exp(params)` | statsmodels |

---

## Key Results

### Model Performance

| Metric | R | Python (sklearn) |
|---|---|---|
| **AUC** | ~0.84 | ~0.84 |
| Accuracy | ~0.81 | ~0.73 |
| Precision (Churn) | ~0.64 | ~0.50 |
| Recall (Churn) | ~0.56 | ~0.80 |

> Both implementations reach AUC ≈ 0.84 (R 0.845, Python 0.835). The Python model (`class_weight="balanced"`) catches more churners (recall 0.80) at the cost of precision (0.50); the R model is the opposite. Both were re-run on R 4.6 and scikit-learn 1.9 with the IBM copy of the dataset.

### Top Churn Drivers (Odds Ratios)

| Predictor | Direction | Interpretation |
|---|---|---|
| Two-year contract | ↓ churn | ~74% lower odds vs month-to-month (OR 0.25 R, 0.26 Python) |
| One-year contract | ↓ churn | ~45% lower odds vs month-to-month (OR 0.55 R, 0.52 Python) |
| Fiber optic internet | ↑ churn | Higher churn vs DSL |
| No TechSupport | ↑ churn | Customers without support churn more |
| No OnlineSecurity | ↑ churn | Security add-on linked to retention |
| Tenure (per month) | ↓ churn | Longer customers are more loyal |
| Paperless billing | ↑ churn | About 45% higher odds (OR 1.46, Python) |

### 6-Step Hypothesis Test: Contract Effect
1. **α = 0.05**
2. **H₀:** β_One-year = β_Two-year = 0 (no contract effect on churn)
3. **Decision rule:** Reject H₀ if p-value ≤ α
4. **Test statistic:** χ²(2) = 58.8 (Type II likelihood-ratio test, R `car::Anova`)
5. **p-value < 0.001** (1.7e-13)
6. **Conclusion:** Reject H₀  contract type significantly predicts churn

---

## Business Insights

1. **Promote long-term contracts at signup.** Month-to-month customers churn at ~43% vs ~3% for two-year customers. Even a small incentive to sign a one-year contract could meaningfully reduce attrition.

2. **Bundle TechSupport and OnlineSecurity.** These services are among the strongest protective factors. Consider offering them free for the first 3 months.

3. **Target new customers aggressively.** Churn risk is highest in the first 12 months of tenure. A proactive outreach program in months 1–6 could significantly improve lifetime value.

4. **Fiber optic customers are high-risk.** Despite paying more, fiber customers churn at higher rates  possibly due to price sensitivity or competition. Investigate service quality and pricing.

---

## How to Reproduce

### Download the data
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in the `data/` folder

### R
```r
install.packages(c("tidyverse","broom","caret","pROC","car"))
source("churn_analysis.R")
```

### Python
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
python churn_analysis.py
```

---

## What I'd Do Next
- Try **Lasso logistic regression** (`glmnet` in R / `LogisticRegression(penalty="l1")` in Python) for automatic variable selection
- Compare against a **Random Forest** baseline to see if non-linear methods outperform logistic regression
- Build a **Shiny / Streamlit dashboard** where a business user can input customer attributes and get a churn probability score
- Address **class imbalance** more rigorously with SMOTE (`themis` in R / `imbalanced-learn` in Python)

---

*Data: IBM Sample Dataset via Kaggle (blastchar). 7,043 customers, 21 variables.*
