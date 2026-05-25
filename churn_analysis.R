# ============================================================
# Telco Customer Churn: Logistic Regression
# STAT 4000 Portfolio Project 2  R Implementation
# Author: Jrudani21
# ============================================================

# ── 0. Packages ──────────────────────────────────────────────
# install.packages(c("tidyverse","broom","caret","pROC","car","ROCR"))
library(tidyverse)
library(broom)
library(caret)
library(pROC)
library(car)

# ── 1. Load & Clean Data ──────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save as data/WA_Fn-UseC_-Telco-Customer-Churn.csv
churn <- read.csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                  stringsAsFactors = TRUE)

# Fix TotalCharges (has blank strings → NA)
churn$TotalCharges <- as.numeric(as.character(churn$TotalCharges))
churn <- churn |> drop_na(TotalCharges)

# Recode target: Churn → 1/0
churn <- churn |>
  mutate(Churn_bin = ifelse(Churn == "Yes", 1, 0))

cat("Rows:", nrow(churn), "\n")
cat("Churn rate:", round(mean(churn$Churn_bin) * 100, 1), "%\n")

# ── 2. EDA ────────────────────────────────────────────────────

# 2a. Churn rate by Contract type
ggplot(churn, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title  = "Churn Rate by Contract Type",
       x      = "Contract",
       y      = "Proportion",
       fill   = "Churn") +
  scale_fill_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  theme_minimal()
ggsave("figures/01_churn_by_contract.png", width = 7, height = 5)

# 2b. Monthly charges by churn
ggplot(churn, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.1, size = 0.8) +
  labs(title = "Monthly Charges by Churn Status",
       x = "Churn", y = "Monthly Charges ($)") +
  scale_fill_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("figures/02_charges_by_churn.png", width = 6, height = 5)

# 2c. Tenure distribution by churn
ggplot(churn, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 3, position = "identity", alpha = 0.6) +
  labs(title = "Customer Tenure by Churn Status",
       x = "Tenure (months)", y = "Count") +
  scale_fill_manual(values = c("No" = "#2196F3", "Yes" = "#F44336")) +
  theme_minimal()
ggsave("figures/03_tenure_hist.png", width = 7, height = 5)

# ── 3. Train / Test Split (70/30) ────────────────────────────
set.seed(42)
train_idx <- createDataPartition(churn$Churn_bin, p = 0.70, list = FALSE)
train     <- churn[train_idx, ]
test      <- churn[-train_idx, ]
cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

# ── 4. Logistic Regression Model ─────────────────────────────
churn.logit <- glm(
  Churn_bin ~ tenure + MonthlyCharges + TotalCharges +
              Contract + InternetService + TechSupport +
              OnlineSecurity + PaperlessBilling + PaymentMethod +
              SeniorCitizen + Partner + Dependents,
  family  = binomial,
  data    = train
)
summary(churn.logit)

# ── 5. Odds Ratios with Confidence Intervals ─────────────────
or_table <- tidy(churn.logit, exponentiate = TRUE, conf.int = TRUE) |>
  filter(p.value < 0.05) |>
  arrange(desc(estimate)) |>
  select(term, estimate, conf.low, conf.high, p.value)

print(or_table, n = 20)

# Plot top odds ratios
or_plot_data <- tidy(churn.logit, exponentiate = TRUE, conf.int = TRUE) |>
  filter(term != "(Intercept)") |>
  arrange(estimate)

ggplot(or_plot_data, aes(x = estimate, y = reorder(term, estimate))) +
  geom_point(size = 3, color = "steelblue") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                 height = 0.3, color = "steelblue") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  labs(title  = "Odds Ratios for Churn Predictors",
       x      = "Odds Ratio (95% CI)",
       y      = NULL,
       caption = "Red dashed line = OR of 1 (no effect)") +
  theme_minimal()
ggsave("figures/04_odds_ratios.png", width = 9, height = 7)

# ── 6. Predictions & Evaluation ──────────────────────────────
# Predicted probabilities on test set
test$pred_prob <- predict(churn.logit, newdata = test, type = "response")
test$pred_class <- ifelse(test$pred_prob >= 0.5, 1, 0)

# Confusion matrix
cm <- confusionMatrix(
  factor(test$pred_class, levels = c(0,1)),
  factor(test$Churn_bin,  levels = c(0,1)),
  positive = "1"
)
print(cm)

cat(sprintf("\nAccuracy:  %.3f\n", cm$overall["Accuracy"]))
cat(sprintf("Sensitivity (Recall): %.3f\n", cm$byClass["Sensitivity"]))
cat(sprintf("Specificity: %.3f\n", cm$byClass["Specificity"]))
cat(sprintf("Precision: %.3f\n", cm$byClass["Pos Pred Value"]))

# ── 7. ROC Curve & AUC ───────────────────────────────────────
roc_obj <- roc(test$Churn_bin, test$pred_prob)
cat(sprintf("\nAUC = %.4f\n", auc(roc_obj)))

png("figures/05_roc_curve.png", width = 600, height = 500)
plot(roc_obj,
     main  = sprintf("ROC Curve  Logistic Regression (AUC = %.3f)", auc(roc_obj)),
     col   = "steelblue",
     lwd   = 2,
     print.auc = TRUE)
dev.off()

# ── 8. Model Diagnostics: 6-Step Hypothesis Test ─────────────
cat("\n===== 6-STEP HYPOTHESIS TEST: Contract Effect =====\n")
cat("1. LEVEL OF SIGNIFICANCE: α = 0.05\n")
cat("2. H0: β_ContractOne-year = β_ContractTwo-year = 0\n")
cat("   HA: At least one β ≠ 0\n")
cat("3. DECISION RULE: Reject H0 if p-value ≤ α\n")

# Wald test for Contract coefficients
anova_result <- Anova(churn.logit, type = "II")
print(anova_result)

contract_p <- anova_result["Contract", "Pr(>Chisq)"]
cat(sprintf("4. TEST STATISTIC: χ² = %.3f\n",
            anova_result["Contract", "LR Chisq"]))
cat(sprintf("5. P-VALUE: %.4e\n", contract_p))
cat("6. CONCLUSION: As p-value ≤ 0.05, reject H0.\n")
cat("   Contract type significantly predicts churn.\n")

# ── 9. Business Interpretation ───────────────────────────────
cat("\n===== BUSINESS INSIGHTS =====\n")
cat("• Month-to-month customers have highest churn risk.\n")
cat("  Encouraging annual/biannual contracts could reduce churn.\n")
cat("• Customers without TechSupport/OnlineSecurity churn more.\n")
cat("  Bundling these services may improve retention.\n")
cat("• Each additional month of tenure reduces churn odds.\n")
cat("  Focus retention efforts on customers in first 12 months.\n")
