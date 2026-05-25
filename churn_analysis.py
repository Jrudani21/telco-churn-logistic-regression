# ============================================================
# Telco Customer Churn: Logistic Regression
# STAT 4000 Portfolio Project 2  Python Implementation
# Author: Jrudani21
# ============================================================

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay,
                              ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Load & Clean Data

# %%
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix TotalCharges blanks
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Binary target
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

print(f"Shape: {df.shape}")
print(f"Churn rate: {df['Churn_bin'].mean()*100:.1f}%")
df.head()

# %% [markdown]
# ## 2. EDA

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Churn rate by Contract
churn_contract = df.groupby("Contract")["Churn_bin"].mean().reset_index()
axes[0].bar(churn_contract["Contract"], churn_contract["Churn_bin"],
            color=["#2196F3", "#4CAF50", "#F44336"])
axes[0].set_title("Churn Rate by Contract Type")
axes[0].set_ylabel("Churn Rate")
axes[0].set_xlabel("Contract")
for i, v in enumerate(churn_contract["Churn_bin"]):
    axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)

# Monthly charges by churn
df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[1],
           boxprops=dict(color="steelblue"),
           medianprops=dict(color="red"))
axes[1].set_title("Monthly Charges by Churn")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Monthly Charges ($)")
plt.sca(axes[1])
plt.title("Monthly Charges by Churn")

# Tenure histogram
for label, grp in df.groupby("Churn"):
    axes[2].hist(grp["tenure"], bins=20, alpha=0.6, label=label,
                 color="#F44336" if label == "Yes" else "#2196F3")
axes[2].set_title("Tenure Distribution by Churn")
axes[2].set_xlabel("Tenure (months)")
axes[2].set_ylabel("Count")
axes[2].legend()

plt.tight_layout()
plt.savefig("figures/py_01_eda.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Feature Engineering & Train/Test Split

# %%
features = ["tenure", "MonthlyCharges", "TotalCharges",
            "Contract", "InternetService", "TechSupport",
            "OnlineSecurity", "PaperlessBilling", "PaymentMethod",
            "SeniorCitizen", "Partner", "Dependents"]

X = df[features]
y = df["Churn_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Identify column types
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [c for c in features if c not in num_cols]

# %% [markdown]
# ## 4. Logistic Regression Pipeline

# %%
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("lr",   LogisticRegression(max_iter=1000, random_state=42,
                                 class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

# Cross-validated AUC
cv_auc = cross_val_score(pipeline, X_train, y_train,
                          cv=5, scoring="roc_auc")
print(f"5-Fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# %% [markdown]
# ## 5. Evaluation

# %%
y_pred      = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Churn", "Churn"]))
print(f"AUC = {roc_auc_score(y_test, y_pred_prob):.4f}")

# %%
# Confusion matrix + ROC curve side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["No Churn", "Churn"],
    colorbar=False, ax=axes[0],
    cmap="Blues"
)
axes[0].set_title("Confusion Matrix")

RocCurveDisplay.from_predictions(
    y_test, y_pred_prob, ax=axes[1],
    name=f"Logistic Regression (AUC={roc_auc_score(y_test,y_pred_prob):.3f})",
    color="steelblue"
)
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_title("ROC Curve")

plt.tight_layout()
plt.savefig("figures/py_02_evaluation.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Odds Ratios via Statsmodels (for Interpretability)

# %%
# Use statsmodels for interpretable coefficients
# One-hot encode categoricals first
df_model = pd.get_dummies(
    df[features + ["Churn_bin"]], drop_first=True
)
df_model.columns = [c.replace(" ", "_").replace("-", "_")
                    for c in df_model.columns]

X_sm = sm.add_constant(df_model.drop("Churn_bin", axis=1))
y_sm = df_model["Churn_bin"]

logit_sm = sm.Logit(y_sm, X_sm).fit(disp=False)
print(logit_sm.summary())

# Odds ratios
or_df = pd.DataFrame({
    "OR":      np.exp(logit_sm.params),
    "CI_low":  np.exp(logit_sm.conf_int()[0]),
    "CI_high": np.exp(logit_sm.conf_int()[1]),
    "p_value": logit_sm.pvalues
}).drop("const").sort_values("OR", ascending=False)

print("\nTop 10 predictors by Odds Ratio:")
print(or_df[or_df["p_value"] < 0.05].head(10).round(3))

# %%
# Odds ratio forest plot
sig = or_df[or_df["p_value"] < 0.05].copy()
sig = sig.sort_values("OR")

fig, ax = plt.subplots(figsize=(9, max(6, len(sig) * 0.35)))
ax.scatter(sig["OR"], range(len(sig)), color="steelblue", s=60, zorder=3)
for i, (_, row) in enumerate(sig.iterrows()):
    ax.plot([row["CI_low"], row["CI_high"]], [i, i],
            color="steelblue", lw=1.5)
ax.axvline(1, color="red", linestyle="--", lw=1.5)
ax.set_yticks(range(len(sig)))
ax.set_yticklabels(sig.index, fontsize=8)
ax.set_xlabel("Odds Ratio (95% CI)")
ax.set_title("Significant Churn Predictors  Odds Ratios\n(OR > 1 = higher churn risk)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/py_03_odds_ratios.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Results Summary  R vs Python Comparison

# %%
print("=" * 60)
print(f"{'Metric':<35} {'Python (sklearn)'}")
print("-" * 60)
auc_val = roc_auc_score(y_test, y_pred_prob)
report  = classification_report(y_test, y_pred, output_dict=True)
print(f"{'AUC':<35} {auc_val:.4f}")
print(f"{'Accuracy':<35} {report['accuracy']:.4f}")
print(f"{'Precision (Churn)':<35} {report['1']['precision']:.4f}")
print(f"{'Recall (Churn)':<35} {report['1']['recall']:.4f}")
print(f"{'F1 (Churn)':<35} {report['1']['f1-score']:.4f}")
print("=" * 60)
print("\nBusiness Insights:")
print("• Month-to-month → Two-year contract reduces churn odds dramatically")
print("• Adding TechSupport reduces churn  bundle it at onboarding")
print("• Target retention campaigns at new customers (tenure < 12 months)")
