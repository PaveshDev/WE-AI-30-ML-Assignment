# Member 2 Preprocessing Strategy

**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Track/domain:** Guided Data Track; Retail & E-commerce  
**Dataset/lens:** UCI Online Retail; Customer Segmentation  
**Author role:** Member 2 — Data Quality, Cleaning, Feature Engineering, Transformations, Scaling

This document explains the complete preprocessing plan: why it is needed, how it is structured,
what each stage contributes, and what it delivers to Member 3.
It is a design explanation, not a reproduction of executed output logs.
Numerical evidence is cited from the executed code and output files listed in the references.

---

## 1. Objective and scope

### 1.1 What this stage is responsible for

Member 2 converts the raw UCI Online Retail Excel file into a model-ready customer-level
feature table. Specifically:

1. Audit and understand the raw dataset's quality issues using Member 1's EDA findings.
2. Apply a documented, ordered six-step cleaning pipeline to remove unusable transaction rows.
3. Aggregate clean transactions to one row per customer using Recency, Frequency and Monetary metrics.
4. Apply log transformations to skewed features and standardize all three model inputs.
5. Export the cleaned transaction file and the final RFM table with a validated schema.
6. Produce diagnostic charts for transparency.
7. Hand the prepared output to Member 3 with a documented contract.

### 1.2 What this stage is explicitly not responsible for

- Running any clustering algorithm or interpreting customer segments.
- Performing the EDA (Member 1's responsibility).
- Evaluating future customer behaviour (Member 4's responsibility).
- Applying the temporal cutoff *after* the historical modelling window is defined (order is fixed).

### 1.3 Inputs consumed

| Input | Source | Description |
| --- | --- | --- |
| `data/raw/Online Retail.xlsx` | UCI ML Repository | Raw 541,909-row transactional Excel file |
| `members/member-1/docs/eda_insight_log.csv` | Member 1 | Q1–Q12 EDA findings identifying quality issues |
| `members/member-1/docs/data_dictionary_raw.csv` | Member 1 | Column-level schema and initial type descriptions |
| `docs/reproducibility/dataset_fingerprint.json` | Member 1 | SHA-256 hash, row count, date bounds for provenance |

---

## 2. Why preprocessing is needed

### 2.1 The granularity mismatch

The UCI Online Retail dataset records individual product lines on individual invoices.
One customer who placed three orders of five items each produces 15 rows in the raw file.
Clustering algorithms require one row per customer. The preprocessing stage bridges that gap.

### 2.2 Data-quality issues requiring correction

Member 1's EDA (Q1–Q12) identified seven categories of problems:

| EDA Ref | Problem | Scale |
| --- | --- | --- |
| Q12 | Exact duplicate transaction rows | 5,268 rows |
| Q1 | Missing CustomerID (cannot link to a person) | 135,037 rows (~25%) |
| Q7, Q8 | Cancellation invoices and negative quantities | 8,872 rows |
| Q8 | Zero or negative UnitPrice | 40 rows |
| Q9 | Non-product stock codes (postage, overhead, samples) | 1,248 rows in scope |
| Q2 | Extreme right skew in Frequency and Monetary | Feature engineering required |
| Q11 | Strong Frequency–Monetary correlation | Known limitation; acknowledged, not removed |

### 2.3 Temporal leakage risk

The project design reserves the period from 2011-09-09 through approximately December 2011
for Member 4's behavioural validation. If transactions from that future window were included
in computing Recency or totalling Monetary, the historical model would be contaminated by
information that only exists after the intended snapshot. The date cutoff is an explicit
anti-leakage measure, not an arbitrary data reduction step.

### 2.4 Feature scale mismatch

Without transformation and scaling, Euclidean-distance-based algorithms (K-Means) would
treat a £177,000 Monetary difference as overwhelmingly more important than a 281-day
Recency difference, not because Monetary is more behaviourally meaningful but solely
because its numbers are larger. This bias must be corrected before handing off to Member 3.

| Feature | Raw range | Issue |
| --- | --- | --- |
| Recency | 0–281 days | Bounded; no severe skew |
| Frequency | 1–131 invoices | Moderate right skew |
| Monetary | £2.90–£177,729.62 | Extreme right skew; ratio max/median ≈ 320:1 |

---

## 3. The preprocessing plan

### 3.1 Pipeline order and rationale

The pipeline follows a strict sequence. Each step's output becomes the next step's input.
The order matters: changing the sequence would alter both the removed-row counts and,
more importantly, the correctness of downstream computations.

| Step | Operation | Key rationale for position |
| --- | --- | --- |
| 1 | Remove exact duplicate rows | Must be first; all subsequent totals must start from non-inflated data |
| 2 | Remove rows with missing CustomerID | Removes rows that cannot be attributed to any customer; must precede aggregation |
| 3 | Remove cancellations (C-prefix OR Qty < 0) | Removes all negative-spend lines so RFM Monetary is never negative |
| 4 | Remove zero/negative UnitPrice rows | Removes data-entry errors that produce zero or negative line spend |
| 5 | Remove non-product stock codes | Removes service/overhead charges from product spend totals |
| 6 | Apply date cutoff (keep rows before 2011-09-09) | Enforces temporal boundary before any customer-level aggregation |
| 7 | Compute RFM features (one row per customer) | Aggregation only possible after all row-level quality filters are applied |
| 8 | Log-transform F and M; StandardScaler all three | Scaling only possible after correct raw values are established |

### 3.2 Why cleaning order matters

**Duplicate removal before filtering:** Removing duplicates first guarantees that all true
duplicates are gone before the denominator changes from subsequent filtering.

**CustomerID removal before cancellations:** Applying CustomerID removal first reduces the
DataFrame size, making subsequent operations faster and ensuring cancelled-invoice counts
in Step 3 refer only to identified customers.

**Cancellations before price filtering:** Some cancellation rows carry negative prices.
Removing them in Step 3 means Step 4 only needs to handle the smaller remaining population.

**Date cutoff before RFM:** If RFM were computed first and the cutoff applied afterwards,
a customer's Recency might use a 2011-November purchase as their "most recent" event,
making them appear more recent than they truly are relative to the 2011-09-09 snapshot.

**Log transformation before scaling:** StandardScaler normalizes the distribution it
receives. If scaling were applied to the raw right-skewed distributions, the resulting
scaled values would still be dominated by extreme outliers pulling the mean and std.
Log first, then scale.

### 3.3 OR versus AND for cancellation detection

The executed code uses OR (row removed if C-prefix **or** negative quantity):

```python
mask_cancel = mask_c_prefix | mask_neg_qty
```

The original preprocessing log description said AND (catch rows that are both C-prefixed
and have negative quantity). Member 3's strategy document notes this discrepancy:

> "Member 2's log says cancellation tests use AND; its source/notebook use OR. Their
> executed evidence shows complete overlap in the retained identifiable cohort."

The OR implementation is the correct defensive form:
- C-prefix rows with negative quantity: caught by both criteria.
- C-prefix rows with non-negative quantity: caught by C-prefix only.
- Negative-quantity rows without C-prefix: caught by Qty < 0 only.

The code output confirms "Negative-only (no C): ~0" rows, meaning almost no additional
rows were caught by the Qty < 0 criterion alone after C-prefix removal. OR is still
the more defensively correct form and is the authoritative implementation.

---

## 4. Feature engineering strategy

### 4.1 Why RFM

RFM (Recency, Frequency, Monetary) is the standard framework for summarizing customer
purchasing behaviour from transactional data. Its advantages for this project:

1. **Directly computable** from the available transaction columns without external data.
2. **Interpretable** — all three dimensions have a clear human meaning in days, orders and pounds.
3. **Established** in retail analytics and directly actionable by the business.
4. **Compatible** with unsupervised clustering: no target label is required.
5. **Endorsed** by the project specification and Member 1's EDA framework.

Alternatives considered but not adopted:

| Alternative | Reason not adopted |
| --- | --- |
| Average order value (AOV) | Derivable from M/F; adds redundancy |
| Return rate | Requires matching cancellation records to original invoices; beyond this stage's scope |
| Country | EDA Q6: 91.4% UK with unreliable non-UK values; would add noise |
| Product diversity / category features | No product taxonomy available in this dataset |

### 4.2 RFM exact definitions

| Feature | Formula | Date reference | Unit | Note |
| --- | --- | --- | --- | --- |
| Recency | `(SNAPSHOT_DATE − max(InvoiceDate)).days` per CustomerID | 2011-09-09 | Integer days | Lower = more recent |
| Frequency | `nunique(InvoiceNo)` per CustomerID | — | Integer count | Distinct invoices, not line items |
| Monetary | `sum(Quantity × UnitPrice)` per CustomerID | — | GBP float | Positive retained lines only |

**Recency:** a customer whose last purchase was 2011-09-08 has Recency = 1 day.
A customer whose last purchase was 2010-12-01 has Recency = 281 days.
Higher Recency means less recent.

**Frequency:** counts distinct invoice numbers, not individual product lines.
A customer who placed one order containing 20 different products has Frequency = 1.

**Monetary:** sums `Quantity × UnitPrice` over all retained positive transaction lines.
After Steps 1–6, every remaining Quantity and UnitPrice is positive, so every line spend
is positive. The defensive `Monetary > 0` check after aggregation found 0 customers to remove.

### 4.3 Snapshot date

The snapshot date `2011-09-09` divides the dataset into:
- **Historical window (before 2011-09-09):** Member 2 builds RFM; Member 3 fits clusters.
- **Future validation window (2011-09-09 onward):** Member 4 evaluates behavioural outcomes.

The raw data spans December 2010 to December 2011, leaving approximately 90 days of
future data for Member 4's evaluation.

---

## 5. Transformation strategy

### 5.1 Why transform before scaling

The log transformation addresses a problem that scaling alone cannot solve.
StandardScaler divides by the standard deviation, but if that standard deviation is
inflated by extreme outliers, the scaled value of an extreme customer is still far
from typical customers.

Illustration for Monetary:

| Customer | Raw Monetary | log1p(Monetary) | Ratio to median |
| --- | --- | --- | --- |
| Typical (median) | £555.02 | ≈ 6.32 | 1.0× |
| Wholesale extreme | £177,729.62 | ≈ 12.09 | 1.91× (was 320×) |

The log compresses the ratio from 320:1 to approximately 1.9:1.

### 5.2 Which features are log-transformed

| Feature | Log-transformed? | Rationale |
| --- | --- | --- |
| Frequency | Yes — `log1p(Frequency)` | Right-skewed; 1,366 of 3,362 customers have Frequency = 1 |
| Monetary | Yes — `log1p(Monetary)` | Strongly right-skewed; max/median ≈ 320:1 |
| Recency | No | Bounded range (0–281 days); less severe skew |

**Why log1p and not log?**
`np.log(0)` returns -inf. Although Monetary is always > 0 and Frequency ≥ 1 after cleaning,
`log1p` — computing `log(1 + value)` — is the more defensive form. It preserves ordering:
`log1p(a) < log1p(b)` whenever `a < b`.

**Does log1p guarantee normality?** No. The transformation compresses the right tail but
does not produce a Gaussian distribution. The one-order customer mass (1,366 customers at
Frequency = 1, log1p ≈ 0.693) creates a discrete spike that survives the transformation.
Member 3's GMM diagnostics confirm this: very narrow covariance components formed around
these repeated discrete frequency values.

### 5.3 StandardScaler

```
scaled_value = (prepared_value − μ) / σ
```

Where `prepared_value` is raw Recency, Log_Frequency, or Log_Monetary; `μ` is the
population mean; `σ` is the population standard deviation.

**Why StandardScaler and not MinMaxScaler?**
MinMaxScaler uses `(x − min) / (max − min)`. The extreme £177,729.62 Monetary value
makes the denominator very large, compressing most customers into a tiny band near zero.
After log transformation, distributions are approximately symmetric, making mean-centring
more appropriate.

**Verified scaled statistics (from executed code):**

| Feature | Mean | Population std |
| --- | --- | --- |
| Recency_scaled | 5.9488e-12 | 0.999999999989 |
| Frequency_scaled | -9.5479e-11 | 0.999999999994 |
| Monetary_scaled | 6.2463e-12 | 0.999999999995 |

These are effectively zero means and unit standard deviations. The sample standard deviation
shown by `pandas.std()` is approximately 1.000149 because it uses Bessel's correction
(denominator n−1) while StandardScaler uses the population convention (denominator n).
This is not a scaling error.

### 5.4 Outlier treatment

No outlier capping or removal was applied. Rationale:

1. **Log already compresses extremes.** Ratio max:median falls from ~320:1 to ~1.9:1.
2. **Extreme customers are genuine** wholesale accounts (EDA Q2 and Q10).
3. **No principled threshold** — any cap at e.g. 99th percentile would be arbitrary without business context.
4. **Member 3 can revisit if needed.** The preprocessing log states this option explicitly.
   Member 3's leading K-Means result was not distorted by extreme customers.

**Trade-off acknowledged:** retaining extreme values means they will still influence cluster
centroids, even after log transformation.

---

## 6. Output contract

### 6.1 cleaned_transactions.csv

| Property | Value |
| --- | --- |
| Path | `data/interim/cleaned_transactions.csv` |
| Rows | 231,806 |
| Columns | 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) |
| InvoiceNo dtype | int64 (numeric after C-prefix removal and CSV round-trip) |
| Date range | 2010-12-01 08:26:00 through 2011-09-08 19:58:00 |
| Unique customers | 3,362 |
| Missing values | 0 |

### 6.2 rfm_table.csv

| Property | Value |
| --- | --- |
| Path | `data/processed/rfm_table.csv` |
| Rows | 3,362 |
| Columns | 9 |
| Missing values | 0 |
| Infinite values | 0 |
| Duplicate CustomerIDs | 0 |
| Monetary <= 0 | 0 |

Column schema:

| Column | dtype | Meaning | Intended use |
| --- | --- | --- | --- |
| CustomerID | int64 | Unique customer identifier | Join key; not a model input |
| Recency | int64 | Days from last retained purchase to 2011-09-09 | Raw reporting and profiling |
| Frequency | int64 | Count of distinct retained invoices | Raw reporting and profiling |
| Monetary | float64 | Total retained positive line spend (GBP) | Raw reporting and profiling |
| Log_Frequency | float64 | `log1p(Frequency)` | Intermediate transformation |
| Log_Monetary | float64 | `log1p(Monetary)` | Intermediate transformation |
| Recency_scaled | float64 | StandardScaler applied to raw Recency | **Model input X** |
| Frequency_scaled | float64 | StandardScaler applied to Log_Frequency | **Model input X** |
| Monetary_scaled | float64 | StandardScaler applied to Log_Monetary | **Model input X** |

**Important naming note:** `Frequency_scaled` and `Monetary_scaled` are standardized versions
of the *log-transformed* columns, not the raw columns. Member 3's study guide Section 3
verifies this to absolute tolerance 1e-7.

---

## 7. Dependencies

### 7.1 Upstream (what I need)

| Dependency | Provider | Risk if absent |
| --- | --- | --- |
| `data/raw/Online Retail.xlsx` | UCI / downloaded automatically by script | Pipeline cannot start |
| EDA insight log Q1–Q12 | Member 1 | Decision rationale would be undocumented |
| Raw data dictionary | Member 1 | Column types and meanings would be unclear |

### 7.2 Downstream (what I provide)

| Consumer | What they need | Where it is |
| --- | --- | --- |
| Member 3 | `rfm_table.csv` with exact 9-column schema | `data/processed/rfm_table.csv` |
| Member 3 | Cleaned transactions for audit date-range checking | `data/interim/cleaned_transactions.csv` |
| Member 4 | Understanding of temporal boundary | This file and `member2_handoff.md` |
| Group report | Cleaning funnel numbers and rationale | `preprocessing_log.md`, this file |

---

## 8. Known limitations and group decisions

1. **Monetary is gross retained spend, not net revenue.** Cancellation rows are removed,
   but returns not invoiced with a C-prefix are not separately identified.

2. **M-code rows (171 rows) are retained.** Manual transaction rows with StockCode M that
   passed all quality criteria remain in the cleaned data.

3. **CustomerID exclusion is deliberate.** The ~135,037 rows without a CustomerID are
   excluded entirely. Segments describe *identified* customers only.

4. **Scaler fitted on historical cohort only.** If new customers were scored later, the
   fitted scaler parameters would need to be re-applied consistently.

5. **Future validation window not prepared here.** Member 4 should prepare that window
   independently from the raw dataset.

---

## 9. Reproducibility note

The notebook uses an absolute `PROJECT_ROOT` path specific to the original development
environment. Update this path before re-running. See `docs/reproducibility.md` for
full instructions. Member 3 uses Member 1's upward project-root discovery convention;
future revisions of the Member 2 pipeline should adopt the same.

---

## 10. References

| Source | Location |
| --- | --- |
| Executed Python script | `members/member-2/notebooks/02_cleaning_and_rfm.py` |
| Executed Jupyter notebook | `members/member-2/notebooks/02_cleaning_and_rfm.ipynb` |
| Cleaning log | `members/member-2/docs/preprocessing_log.md` |
| Member 1 EDA insights | `members/member-1/docs/eda_insight_log.csv` |
| Dataset fingerprint | `docs/reproducibility/dataset_fingerprint.json` |
| Member 3 strategy (upstream qualification) | `members/member-3/docs/modelling_strategy.md` |
| RFM table (output) | `data/processed/rfm_table.csv` |
| Cleaned transactions (output) | `data/interim/cleaned_transactions.csv` |
