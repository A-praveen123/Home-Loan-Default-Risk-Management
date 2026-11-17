# Home Loan Default: Risk Management (PRCP-1006)

Predicting which loan applicants are likely to default, and which customer segments are eligible for a loan, using the Home Credit data (7 relational tables, Banking domain).

> **Notebook:** `Home_Loan_Default_-_Risk_Management.ipynb`  (all tasks are in this single notebook)

---

## 1. Problem statement

| Task | Deliverable |
|---|---|
| **Task 1** | A complete data analysis report on the given data |
| **Task 2** | A predictive model to identify the factors and customer segments eligible for taking a loan |
| **Model Comparison Report** | Performance of multiple models, with the best model recommended for production |
| **Challenges Report** | Challenges faced on the data and the techniques used, with reasons |

**Target:** `TARGET` = 1 (defaulter) / 0 (non-defaulter).

---

## 2. Dataset

Source: the PRCP-1006 project zip (public Home Credit Default Risk data).

| File | Description | Granularity |
|---|---|---|
| `application_train.csv` | Main table with the target and static application data | 1 row per loan |
| `bureau.csv` | Client's previous credits from other institutions | 1 row per previous credit |
| `bureau_balance.csv` | Monthly balances of those bureau credits | 1 row per credit per month |
| `previous_application.csv` | Previous Home Credit applications | 1 row per previous application |
| `POS_CASH_balance.csv` | Monthly snapshots of POS and cash loans | 1 row per loan per month |
| `credit_card_balance.csv` | Monthly snapshots of credit cards | 1 row per card per month |
| `installments_payments.csv` | Repayment history (made and missed payments) | 1 row per instalment |

The CSV files are not included in this repository because of their size. Download the zip from the project link and extract it into a local folder.

---

## 3. Approach

1. **Setup and loading:** memory-efficient loading (`float32` / small-int downcasting).
2. **Exploratory data analysis (Task 1):**
   - Class imbalance, missing values and data-quality checks.
   - Numeric and categorical features compared against default rate.
   - Correlation analysis.
   - Overview of the auxiliary tables.
3. **Feature engineering:**
   - Every child table is aggregated to **one row per client** (`SK_ID_CURR`: mean, max, min, sum, counts) and left-joined.
   - Domain features: credit-to-income, annuity-to-income, credit term, employment-to-age, `EXT_SOURCE` mean / product / std, payment ratio, days past due, card utilisation, bureau debt-to-credit.
4. **Cleaning:**
   - `DAYS_EMPLOYED = 365243` placeholder is turned into `NaN` plus an anomaly flag.
   - `XNA` codes are treated as missing.
   - Extreme incomes are winsorised at the 99.9th percentile.
5. **Modelling (Task 2):** stratified **64 / 16 / 20** train / validation / test split, then:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - HistGradientBoosting (plain and tuned with `RandomizedSearchCV`)
   - LightGBM
6. **Model comparison:** ROC-AUC, PR-AUC, KS, precision, recall, F1, fit time, plus ROC and PR curves.
7. **Factors and segments:**
   - Permutation importance for the key default drivers.
   - Four risk bands (A to D) from validation-set score cut-offs.
   - Decile lift and an approval-rate vs. default-rate curve.
   - Segment tables (income type, education, age band, income band, contract type).
   - Shallow decision-tree rules for explainable eligibility.
8. **Challenges report:** each challenge with the technique used and why.
9. **Deployment helper:** the best model is saved, and `score_applicants()` returns PD, risk band and decision (Eligible / Review / Decline).

---

## 4. Risk bands and decision policy

| Band | Score position (validation) | Action |
|---|---|---|
| A: Low risk | lowest 40% | Approve (standard terms) |
| B: Moderate | next 30% | Approve (standard / risk-based pricing) |
| C: Elevated | next 20% | Manual review / lower limit |
| D: High risk | riskiest 10% | Decline or require collateral |

Cut-offs are set on the validation set so test-set results stay unbiased.

---

## 5. Results

> **Fill this section in from a run on the full, real dataset.** The numbers depend on the data and the random seed, so copy them from your final notebook outputs (`artifacts/model_comparison.csv`).

| Model | ROC-AUC | PR-AUC | KS | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| Random Forest | | | | | | |
| HistGradientBoosting | | | | | | |
| HistGradientBoosting (tuned) | | | | | | |
| LightGBM | | | | | | |

- **Recommended production model:** _name and one-line reason_
- **Top default drivers:** _list the top 5 from the permutation-importance table_
- **Default rate in the data:** _x%_  (imbalance ratio 1 : _y_)
- **Band results:** _default rate in band A vs. band D and the lift of band D_

---

## 6. How to run

**Requirements:** Python 3.9+ and about 8 GB RAM for the full data (`bureau_balance.csv` has ~27M rows and `installments_payments.csv` ~13M).

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib jupyter
```

1. Download and extract the dataset zip into a folder.
2. Open the notebook and set the data folder in the first code cell (on Windows keep the `r` before the quote):

   ```python
   DATA_DIR = r'C:\path\to\your\folder'
   ```

3. Run **Kernel → Restart & Run All**.

If a file is missing, the setup cell stops and lists which of the 7 CSVs it could not find.

**Score new applicants with the saved model:**

```python
import joblib
art = joblib.load('artifacts/home_loan_default_model.joblib')
scores = score_applicants(new_features_df)   # columns must match art['features']
```

---

## 7. Project structure

```
.
├── Home_Loan_Default_-_Risk_Management.ipynb   # full analysis, models and reports
├── README.md
├── artifacts/                                  # created by the notebook
│   ├── home_loan_default_model.joblib          # model, feature list, band cut-offs
│   └── model_comparison.csv                    # comparison table
└── data/                                       # CSV files (not tracked)
```

---

## 8. Key challenges and techniques (summary)

| Challenge | Technique |
|---|---|
| Very large tables | Dtype downcasting, chunked reading of `bureau_balance`, freeing memory with `gc` |
| One-to-many joins | Aggregate each child table to one row per client |
| Imbalanced target | Stratified splits, class weights, AUC / PR-AUC / KS instead of accuracy |
| Missing values | Median imputation for linear and bagged models, native NaN handling in boosting, counts filled with 0 |
| Placeholder values (`365243`, `XNA`) | Convert to `NaN` and add an anomaly flag |
| Weak raw signal | Domain ratio and behaviour features |
| Over-optimistic results | Separate validation set for early stopping and thresholds, test set used once |
| Explaining a black-box model | Permutation importance, risk bands, shallow-tree rules, logistic-regression challenger |

The full table with reasons is in the notebook (Challenges report section).

---

## 9. Possible improvements

- Probability calibration (Platt or isotonic) before setting cut-offs.
- SHAP reason codes for adverse-action explanations.
- Fairness checks across protected groups.
- Population-stability (PSI) and KS monitoring after deployment.
- A/B test of the risk-band cut-offs against the current credit policy.

---

## 10. Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · matplotlib · seaborn · Jupyter

---

## 11. Author

**A. Praveen**: [GitHub](https://github.com/A-praveen123) · [LinkedIn](https://linkedin.com/in/praveen-a-21b36531a)
