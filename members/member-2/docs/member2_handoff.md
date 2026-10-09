# Member 2 Handoff to Member 3

**Prepared by:** Member 2  
**Recipient:** Member 3 (ML/Data-Mining Strategy & Modelling)  
**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Dataset:** UCI Online Retail  

This document describes exactly what Member 3 receives from Member 2, how to use it,
what validation checks were performed, known limitations, and the temporal boundary
separating the historical modelling window from Member 4's validation window.

---

## 1. Primary deliverable: rfm_table.csv

**Location:** `data/processed/rfm_table.csv`

### 1.1 Verified schema

| Column | dtype | Description | Intended use |
| --- | --- | --- | --- |
| CustomerID | int64 | Unique customer identifier | Join key; **not** a model input |
| Recency | int64 | Days from last retained purchase to 2011-09-09 | Raw profiling; lower = more recent |
| Frequency | int64 | Count of distinct retained invoices | Raw profiling; higher = more orders |
| Monetary | float64 | Total retained positive line spend in GBP | Raw profiling; higher = more spend |
| Log_Frequency | float64 | `log1p(Frequency)` | Intermediate; Member 3 should not re-transform |
| Log_Monetary | float64 | `log1p(Monetary)` | Intermediate; Member 3 should not re-transform |
| Recency_scaled | float64 | StandardScaler applied to **raw Recency** | **Model input X** |
| Frequency_scaled | float64 | StandardScaler applied to **Log_Frequency** | **Model input X** |
| Monetary_scaled | float64 | StandardScaler applied to **Log_Monetary** | **Model input X** |

**Critical naming note:** `Frequency_scaled` and `Monetary_scaled` are standardized
versions of the *log-transformed* columns, not the raw columns. Despite the shorter
column names, no further transformation is needed before feeding these to clustering.

### 1.2 Row count and uniqueness

| Check | Result |
| --- | --- |
| Rows | 3,362 |
| Unique CustomerIDs | 3,362 |
| Duplicate IDs | 0 |
| Duplicate rows | 0 |
| Missing values (all columns) | 0 |
| Infinite values (numeric columns) | 0 |
| Monetary <= 0 | 0 |

### 1.3 Raw RFM ranges

| Feature | Minimum | 25th pct | Median | 75th pct | Maximum |
| --- | --- | --- | --- | --- | --- |
| Recency (days) | 0 | 25 | 73 | 150 | 281 |
| Frequency (invoices) | 1 | 1 | 2 | 4 | 131 |
| Monetary (GBP) | £2.90 | £260.66 | £555.015 | £1,363.26 | £177,729.62 |

### 1.4 Scaled feature statistics

| Feature | Mean (population) | Std (population) |
| --- | --- | --- |
| Recency_scaled | 5.9488e-12 ≈ 0 | 0.999999999989 ≈ 1 |
| Frequency_scaled | -9.5479e-11 ≈ 0 | 0.999999999994 ≈ 1 |
| Monetary_scaled | 6.2463e-12 ≈ 0 | 0.999999999995 ≈ 1 |

The sample standard deviation (pandas `.std()`) is approximately 1.000149 due to
Bessel's correction (n−1 denominator). StandardScaler uses the population convention (n).
This is not a scaling error.

### 1.5 One-order customers

1,366 customers (40.63%) placed exactly one retained invoice (Frequency = 1).
This large discrete mass at Frequency = 1 affects baseline scoring (Frequency score 1 is
unused in the midrank quintile approach).

---

## 2. Secondary deliverable: cleaned_transactions.csv

**Location:** `data/interim/cleaned_transactions.csv`

| Property | Value |
| --- | --- |
| Rows | 231,806 |
| Columns | 8 |
| Unique customers | 3,362 |
| Date range | 2010-12-01 08:26:00 through 2011-09-08 19:58:00 |
| InvoiceNo dtype | int64 (numeric; all C-prefix rows removed) |
| Missing values | 0 |

This file is available for any additional historical analysis. Member 3 does not need
to re-read it for clustering — the RFM table already aggregates it.

---

## 3. Cleaning pipeline summary

| Step | Action | Rows removed | Rows after |
| --- | --- | --- | --- |
| Raw | — | — | 541,909 |
| 1 | Exact duplicates removed | 5,268 | 536,641 |
| 2 | No CustomerID | 135,037 | 401,604 |
| 3 | Cancellations (C-prefix OR Qty < 0) | 8,872 | 392,732 |
| 4 | UnitPrice <= 0 | 40 | 392,692 |
| 5 | Non-product codes (POST, DOT, C2, S) | 1,248 | 391,444 |
| 6 | Date cutoff (keep rows before 2011-09-09) | 159,638 | 231,806 |

RFM aggregation: 231,806 rows → 3,362 customers.
Defensive Monetary > 0 filter: 0 customers removed.

---

## 4. Temporal boundary

### 4.1 Historical window (Member 2's scope)

| Boundary | Date/time | Action |
| --- | --- | --- |
| Start | 2010-12-01 08:26:00 | Earliest retained transaction |
| End (exclusive) | 2011-09-09 00:00:00 | Snapshot date; transactions on/after this date excluded |

**Recency reference point:** `SNAPSHOT_DATE = pd.Timestamp("2011-09-09")`.
Recency = `(SNAPSHOT_DATE − last_retained_invoice_date).days`.
A customer whose last purchase was 2011-09-08 has Recency = 1.

### 4.2 Future validation window (Member 4's scope)

| Boundary | Date/time | Note |
| --- | --- | --- |
| Start (inclusive) | 2011-09-09 00:00:00 | First instant not included in historical RFM |
| End | ~2011-12-09 | Raw data ends here; exact endpoint to be agreed with Member 4 |

**Critical rule for Member 3:** do **not** use any transaction from 2011-09-09 onward
when fitting, transforming, or selecting models. The historical cohort is 3,362 customers
with data strictly before 2011-09-09.

**Critical rule for Member 4:** the future period must be prepared independently from the
raw Excel file. Do **not** re-use the cleaned_transactions.csv (which contains history
only) as a source of future data. Aggregate future transactions by CustomerID, then
left-join to the cluster_assignments.csv using CustomerID.

---

## 5. What Member 3 should use

```python
FEATURES = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
X = rfm[FEATURES].to_numpy(dtype=float, copy=True)
```

- Use `rfm_table.csv` as the single source of truth.
- Do not re-run the cleaning pipeline.
- Do not re-scale or re-transform features.
- Do not use CustomerID as a distance dimension.
- Retain raw Recency, Frequency, Monetary columns for cluster profiling in interpretable units.

---

## 6. Input audit contract

Member 3's `modelling_helpers.py` includes an `audit_input` function that checks:

| Check | Expected result |
| --- | --- |
| Rows / unique CustomerIDs | 3,362 / 3,362 |
| Columns | 9 |
| Missing values / infinite values | 0 / 0 |
| Duplicate IDs / rows / model-feature vectors | 0 / 0 / 0 |
| Historical rows on/after cutoff | 0 |
| Date range in cleaned_transactions | 2010-12-01 through 2011-09-08 |
| Transformation identity: Frequency_scaled = StandardScaler(log1p(Frequency)) | Tolerance 1e-7 |
| Scaling stats: mean ≈ 0, std ≈ 1 | Verified |
| One-order customers | 1,366 (40.63%) |

Member 3's executed audit confirmed all checks passed.

---

## 7. Known limitations for Member 3

1. **Monetary is gross retained positive spend, not net revenue.** Returns not invoiced
   with a C-prefix are not deducted.

2. **F–M Spearman correlation ≈ 0.78** (post-cutoff). Scaling does not remove this
   correlation; it is a structural property of purchasing behaviour. K-Means distance
   is sensitive to correlated features.

3. **1,366 one-order customers** (40.63%) create a discrete mass at Frequency = 1.
   This limits how well a Gaussian distribution fits the logged Frequency.

4. **171 M-code rows** (miscellaneous manual transactions) are retained in the cleaned
   data. Their effect on Monetary for affected customers is minor.

5. **Cancellation logic is OR** (not AND as described informally in the log). The
   executed behaviour is correct; the log description is informal. Member 3's strategy
   document records this discrepancy.

6. **Absolute path in script:** the source code contains a hard-coded `PROJECT_ROOT`.
   Member 3 should not re-run Member 2's pipeline; use the existing outputs directly.

---

## 8. Figures available

| Figure | Location | What it shows |
| --- | --- | --- |
| `cleaning_funnel.png` | `members/member-2/figures/` | Rows remaining after each cleaning step |
| `rfm_distributions_raw.png` | `members/member-2/figures/` | Raw R, F, M histograms with median lines |
| `rfm_distributions_transformed.png` | `members/member-2/figures/` | Scaled R, F, M histograms after transformation |
| `rfm_correlation_heatmap.png` | `members/member-2/figures/` | Pearson correlation of the three scaled features |

---

## 9. Documentation available

| Document | Location | Contents |
| --- | --- | --- |
| Preprocessing log | `members/member-2/docs/preprocessing_log.md` | Step-by-step cleaning record, EDA cross-references |
| Preprocessing strategy | `members/member-2/docs/preprocessing_strategy.md` | Full plan rationale and output contract |
| Preprocessing decisions | `members/member-2/docs/preprocessing_decisions.md` | Structured decision records |
| Reproducibility guide | `members/member-2/docs/reproducibility.md` | How to re-run safely |
| Data dictionary | `docs/data-dictionary/member2_data_dictionary.md` | Full column definitions |

---

## 10. Summary for Member 3

> Member 2 delivers a 3,362-customer RFM table with three model-ready scaled columns,
> a 231,806-row cleaned transaction file, and full documentation. All quality checks passed.
> Fit all models on the existing `rfm_table.csv`. Do not re-run cleaning or re-scale features.
> The future period begins at 2011-09-09 00:00:00 and belongs entirely to Member 4.
