# Member 2 Study Guide – Data Preprocessing & Feature Engineering

**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Track/domain:** Guided Data Track; Retail & E-commerce  
**Dataset/lens:** UCI Online Retail; Customer Segmentation  
**My contribution:** Data quality analysis, cleaning pipeline, RFM construction, log-transformation, scaling, and a clean handoff to Member 3.

This is a personal study guide, not the final report. It explains the existing work; the pipeline was not rerun to write it. Follow the linked files when you need the underlying evidence.

**Reading route:** understand Sections 3–8 first; use Sections 9–14 beside the result files; revise Sections 20–26 before a viva.

**Number convention:** row counts are exact. Percentages use two decimal places. Monetary values use two decimal places unless stated otherwise. "N/A" means a field was not applicable, not zero.

## 1. My role in the group

My responsibility begins with Member 1's exploratory data analysis. I designed and executed the full cleaning pipeline, computed the customer-level RFM table, applied log transformations and StandardScaler, and handed the model-ready output to Member 3.

| Responsibility | Owner |
| --- | --- |
| Dataset understanding and original exploratory analysis | Member 1 |
| Cleaning, RFM construction, log transformations and scaling | Me: Member 2 |
| Baseline, clustering, parameter experiments, internal comparison and seed stability | Member 3 |
| Future three-month behavioural evaluation, final segment names and business recommendations | Member 4 |

I did not run any clustering model or interpret segments. I produced clean, consistently scaled purchasing features for Member 3 to use directly.

**How I can explain this in a viva**

> "I took the raw UCI Online Retail data and removed noise, bad records and future transactions. I then summarised each customer's history into three behavioural features — Recency, Frequency and Monetary — log-transformed the skewed ones, and standardised all three so Member 3's distance-based models could treat each feature equally."

## 2. Why my part is important

Raw transactional data cannot be fed into clustering algorithms. It is at the wrong level of granularity (individual line items, not customers), contains data-quality issues, and has vastly different feature scales.

My stage converts 541,909 raw transaction rows into a 3,362-row customer-level table with three prepared, scale-normalised behavioural dimensions. Without this:

- Distance-based models like K-Means would be dominated by monetary values measured in hundreds or thousands of pounds compared to recency measured in small integers.
- Duplicate and cancelled transactions would inflate Monetary, making some customers look falsely active.
- Future transactions would leak into the historical modelling window reserved for Member 4's validation.

Clean, well-scaled input is a prerequisite for defensible downstream modelling.

## 3. Input from Member 1

**Main input:** `data/raw/Online Retail.xlsx`  
**Preparation record:** Member 1's exploratory data analysis insights (`eda_insight_log.csv`).  
**Cross-references used from EDA:** Q1, Q2, Q7, Q8, Q9, Q11, Q12.

Member 1 identified the key data-quality risks: missing CustomerIDs, duplicate rows, cancellation invoices, negative quantities, non-product stock codes, spend skewness, and F-M correlation. My pipeline addresses each of these explicitly.

### Raw dataset facts

| Property | Value |
| --- | --- |
| Total rows | 541,909 |
| Columns | 8 (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) |
| Rows with missing CustomerID | ~135,037 |
| Rows with missing Description | 1,454 |
| Approximate date range | December 2010 – December 2011 |

The raw dataset is at transaction-line level. Each row represents one product line on one invoice, not one customer or one order.

## 4. The cleaning pipeline — overview

The pipeline follows a strict order. The sequence matters because some earlier steps change the denominator that later steps operate on.

| Step | Action | Rows Before | Rows Removed | Rows After |
|------|---------|-------------|--------------|------------|
| — | Raw | — | — | 541,909 |
| 1 | Duplicates removed | 541,909 | 5,268 | 536,641 |
| 2 | No CustomerID | 536,641 | 135,037 | 401,604 |
| 3 | Cancellations (C-prefix OR Qty < 0) | 401,604 | 8,872 | 392,732 |
| 4 | Price <= 0 | 392,732 | 40 | 392,692 |
| 5 | Non-product codes | 392,692 | 1,248 | 391,444 |
| 6 | Date cutoff (keep rows before 9 Sep 2011) | 391,444 | 159,638 | 231,806 |

**Final cleaned transactions:** 231,806 rows  
**Final RFM table:** 3,362 customers

## 5. Cleaning pipeline — step by step

### 5.1 Step 1: Remove exact duplicate rows

**What:** `df.drop_duplicates()` removes rows where every column value is identical to another row.

**Why (EDA Q12):** duplicate line items would count the same purchase twice and inflate each customer's Monetary value. Removing them first ensures all later totals are correct.

**Result:** 5,268 rows removed; 536,641 remain.

**How I can explain this in a viva**

> "I removed duplicates first so that all subsequent computations — particularly Monetary totals — are not inflated by repeated records."

### 5.2 Step 2: Remove rows with missing CustomerID

**What:** drop rows where `CustomerID` is null, then cast the remaining values to `int64`.

**Why (EDA Q1):** we cannot build a per-customer model without knowing which customer made each purchase. These 135,037 rows cannot be linked to a customer and must be excluded.

**Note on Description nulls:** the raw data contains 1,454 rows with a missing Description. No dedicated rule removes them. All such rows are eliminated as a side effect of Step 2 (and to a lesser extent Steps 3–5). The cleaned dataset has 0 null Description values.

**Result:** 135,037 rows removed; 401,604 remain.

**How I can explain this in a viva**

> "Without a CustomerID we cannot aggregate a customer's history or assign them to a segment. These rows are discarded entirely."

### 5.3 Step 3: Remove cancellations

**What:** remove rows where `InvoiceNo` starts with `'C'` **OR** `Quantity < 0`.

**Why (EDA Q7, Q8):** cancellation invoices represent reversed or returned orders. Including them would reduce Monetary (because they carry negative spend) and could create customers with zero or negative net spend that cannot be log-transformed.

**Criterion details:**

| Criterion | Count |
| --- | --- |
| C-prefix rows | ~8,872 (approximate; see actual run) |
| Negative-qty with no C-prefix | ~0 additional |
| Total removed (union) | 8,872 |

Using **OR** catches both C-prefix and any stray negative-quantity rows that may lack the prefix. The AND condition would miss those stray rows. After removing C-prefix rows, ~0 additional rows are removed by the Quantity < 0 criterion, but OR is the more defensively correct form.

**Result:** 8,872 rows removed; 392,732 remain.

**How I can explain this in a viva**

> "Cancellation invoices are identified by a C-prefix or negative quantity. I used OR to catch both simultaneously, which matches standard retail data practice."

### 5.4 Step 4: Remove rows where UnitPrice <= 0

**What:** drop any row where `UnitPrice` is zero or negative.

**Why (EDA Q8):** a zero or negative price produces zero or negative line spend. Such rows are likely data-entry errors or internal adjustments rather than genuine customer purchases.

**Result:** 40 rows removed; 392,692 remain.

**How I can explain this in a viva**

> "Zero or negative prices create meaningless spend values and are removed as data-quality errors."

### 5.5 Step 5: Remove non-product stock codes

**What:** remove rows where `StockCode` (uppercased) matches `POST`, `DOT`, `C2`, or `S`.

**Why (EDA Q9):** these codes represent service or overhead charges — postage (`POST`), dot (`DOT`), carriage charges (`C2`), and samples (`S`) — not actual retail products.

**What is kept:**

| Code | Treatment |
| --- | --- |
| POST | Removed — postage charge |
| DOT | Removed — overhead |
| C2 | Removed — carriage |
| S | Removed — sample |
| D | Not explicitly removed — all D rows removed by earlier steps |
| M | Not explicitly removed — 171 valid M rows survive to the cleaned dataset |

**Result:** 1,248 rows removed; 391,444 remain.

**How I can explain this in a viva**

> "Non-product codes inflate or distort customer spend. I removed the four specified codes and checked that D and M rows were handled correctly by earlier steps or were valid purchases."

### 5.6 Step 6: Date cutoff — keep rows before 9 September 2011

**What:** convert `InvoiceDate` to datetime and drop any row with `InvoiceDate >= 2011-09-09`.

**Why (project design):** the snapshot date `2011-09-09` is the RFM reference point. Transactions on or after this date belong to the **future validation window** reserved for Member 4's behavioural evaluation. Using them in RFM construction would contaminate that held-out period.

**Snapshot date meaning:** Recency is measured as "whole elapsed days from the customer's most recent retained purchase to the snapshot date". The snapshot must be fixed before Recency can be calculated.

**Result:** 159,638 rows removed; 231,806 remain.

**How I can explain this in a viva**

> "Cutting dates before computing RFM ensures the validation window — starting 9 September 2011 — is completely untouched by the historical model. Member 3 fits on history only; Member 4 evaluates on the future period, with Member 4 to confirm the exact final evaluation date."

## 6. Building the RFM table (Step 7)

After cleaning, 231,806 transaction rows covering 3,362 unique customers remain. We aggregate to one row per customer using three purchasing dimensions.

### 6.1 What RFM means

| Feature | Definition | Unit | Direction |
| --- | --- | --- | --- |
| Recency | Days from customer's last retained invoice to 2011-09-09 | Whole days (int) | Lower = more recent = behaviourally better |
| Frequency | Count of distinct retained invoice numbers | Count (int) | Higher = more orders = behaviourally better |
| Monetary | Sum of `Quantity x UnitPrice` across all retained positive lines | GBP (float) | Higher = more spend = behaviourally better |

### 6.2 Important definitions

**Recency** uses elapsed days — `(SNAPSHOT_DATE - last_invoice_date).days` — giving a whole integer. A customer who last purchased on 2011-09-08 has Recency 1; a customer whose last purchase was in December 2010 has Recency ~281.

**Frequency** counts distinct invoice numbers, not individual product lines or total item quantities. A customer who placed three orders (each containing five products) has Frequency 3, not 15.

**Monetary** sums `Quantity x UnitPrice` over all retained positive transaction lines. Cancellations and returns have already been removed in Step 3. After RFM aggregation, a defensive check confirms all customers have Monetary > 0; in this cleaned dataset, 0 customers required removal.

### 6.3 Actual raw RFM ranges

| Feature | Minimum | Median | Maximum |
| --- | --- | --- | --- |
| Recency (days) | 0 | 73 | 281 |
| Frequency (invoices) | 1 | 2 | 131 |
| Monetary (GBP) | £2.90 | £555.015 | £177,729.62 |

The Monetary maximum of £177,729.62 is orders of magnitude larger than the median of £555. This extreme right-skew is the primary motivation for the log transformation in Step 8.

**One-order customers:** 1,366 customers (40.63%) placed exactly one retained invoice. This single-purchase mass is a discrete feature of the data that affects how ties are handled in Member 3's baseline scoring.

### 6.4 Defensive Monetary > 0 check

Before log-transformation, the code checks `(rfm["Monetary"] <= 0).sum()`. The result is **0**; no customers required removal. Member 1's EDA (Q2) predicted around 50 such customers, but that analysis included cancellation rows in the monetary calculation. After Steps 1–6 remove cancellations and negative-quantity rows, all remaining customers already have positive net spend. The check is retained as a defensive data-quality guard.

**How I can explain this in a viva**

> "RFM gives each customer three numbers: how long since they last bought, how many times they bought, and how much they spent. I compute these from only the historical transactions so the future period is genuinely held out."

## 7. Log transformation and scaling (Step 8)

### 7.1 Why transform and scale at all

Raw Recency ranges from 0 to 281. Raw Monetary ranges from £2.90 to £177,729.62. Without scaling, K-Means and other distance-based models would give enormously more weight to Monetary simply because its numbers are larger. Scaling removes that unit-of-measurement bias.

However, Frequency and Monetary also have extreme right skews (long upper tails). Standardizing skewed data still leaves a few large values pulling distances. The log transformation first compresses that tail.

### 7.2 Log transformation — log1p

The code applies `np.log1p(value)`, which computes the natural logarithm of `(1 + value)`.

| Why log1p and not log? | Because Frequency starts at 1; `log(1) = 0` and `log(0)` is undefined. `log1p` maps 1 -> 0.693, 2 -> 1.099, etc., and is always safe for non-negative integer inputs. |

Only Frequency and Monetary receive the log transformation. Recency has a more bounded range and is scaled directly from its raw value.

| Column | Transformation | Notes |
| --- | --- | --- |
| Recency | None (raw) | Bounded range; no severe skew in this dataset |
| Frequency | `log1p(Frequency)` -> `Log_Frequency` | Compresses right tail from 1-order to 131-order customers |
| Monetary | `log1p(Monetary)` -> `Log_Monetary` | Compresses tail from £2.90 to £177,729.62 |

A concrete example: `log1p(177729.62) ≈ 12.09`; `log1p(555.0) ≈ 6.32`. The ratio shrinks from ~320x to ~2x, making the high spender far less dominant in Euclidean distance.

### 7.3 StandardScaler

After log-transformation, the three prepared features — raw Recency, Log_Frequency, Log_Monetary — are passed to `sklearn.preprocessing.StandardScaler`.

StandardScaler subtracts the population mean and divides by the population standard deviation:

```text
scaled value = (prepared value - mean) / std
```

The result has:
- Mean ≈ 0 (effectively zero, around 10^-11 due to floating-point precision)
- Population standard deviation ≈ 1 (recorded as 0.999999999989 to 0.999999999995)

The displayed sample standard deviation is approximately **1.000149** because `pandas.std()` uses an n-1 (Bessel-corrected) denominator, while StandardScaler uses the population (n) denominator. This is not a scaling error.

**Why StandardScaler and not MinMaxScaler?**

MinMaxScaler maps values to [0, 1] but is sensitive to outliers: the extreme £177,729.62 customer would compress all others toward the lower end. After log transformation, the distributions are approximately normal, making StandardScaler's mean-centering approach optimal for K-Means.

### 7.4 Scaled feature statistics

| Feature | Recorded mean | Population standard deviation |
| --- | --- | --- |
| Recency_scaled | 5.9488e-12 | 0.999999999989 |
| Frequency_scaled | -9.5479e-11 | 0.999999999994 |
| Monetary_scaled | 6.2463e-12 | 0.999999999995 |

These are effectively zero means and unit standard deviations. Equal population standard deviation does **not** mean every customer contributes equally, every feature is independent, or the data have become Gaussian. Historical Frequency/Monetary Spearman correlation is **0.7842**; scaling does not remove that.

### 7.5 Column naming note — important

The column names `Frequency_scaled` and `Monetary_scaled` are **shorter names** for convenience. Their actual input is:

| Scaled column | What was actually scaled |
| --- | --- |
| `Recency_scaled` | Raw `Recency` |
| `Frequency_scaled` | `Log_Frequency` (i.e., `log1p(Frequency)`) |
| `Monetary_scaled` | `Log_Monetary` (i.e., `log1p(Monetary)`) |

This was a deliberate design choice to keep column names readable. Member 3's study guide confirms this and verifies the transformation relationships to an absolute tolerance of 1e-7.

**How I can explain this in a viva**

> "Frequency and Monetary were both log-transformed first to compress their right tails, then all three prepared features were standardised so each contributes equally to Euclidean distance. I chose StandardScaler over MinMaxScaler because log-transformed values are roughly normally distributed, and MinMaxScaler is more sensitive to remaining outliers."

## 8. Output files

| File | Location | Rows / Content |
| --- | --- | --- |
| `cleaned_transactions.csv` | `data/interim/` | 231,806 transaction rows, post-cleaning pre-RFM |
| `rfm_table.csv` | `data/processed/` | 3,362 customer rows, 9 columns |
| `cleaning_funnel.png` | `members/member-2/figures/` | Horizontal bar chart showing rows retained at each step |
| `rfm_distributions_raw.png` | `members/member-2/figures/` | Three histograms of raw R, F, M with median lines |
| `rfm_distributions_transformed.png` | `members/member-2/figures/` | Three histograms of scaled features |
| `rfm_correlation_heatmap.png` | `members/member-2/figures/` | Pearson correlation of the three scaled features |

### RFM table columns (9 columns)

| Column | Type | Meaning |
| --- | --- | --- |
| CustomerID | int64 | Unique customer identifier; not a behavioural input |
| Recency | int64 | Whole elapsed days from last retained purchase to snapshot |
| Frequency | int64 | Count of distinct retained invoice numbers |
| Monetary | float64 | Total retained positive line spend in GBP |
| Log_Frequency | float64 | `log1p(Frequency)` |
| Log_Monetary | float64 | `log1p(Monetary)` |
| Recency_scaled | float64 | StandardScaler applied to raw Recency |
| Frequency_scaled | float64 | StandardScaler applied to Log_Frequency |
| Monetary_scaled | float64 | StandardScaler applied to Log_Monetary |

Member 3 uses only `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled` as model inputs (X). The raw and log columns are retained for profile interpretation in human-understandable units.

## 9. Design decisions

### 9.1 Why duplicate removal is first (Step 1)

Removing duplicates before computing totals ensures that no customer's Monetary value is inflated by repeated line items. The EDA (Q12) identified this risk explicitly.

### 9.2 Why OR for cancellation detection (Step 3)

The OR condition catches cancellations with a C-prefix invoice (the primary indicator) and also any stray negative-quantity rows that may lack the prefix. The AND condition would miss those stray rows. After removing C-prefix rows, ~0 additional rows are removed by the Quantity < 0 criterion, but OR is the more defensively correct form. The preprocessing log documents this explicitly.

### 9.3 Why date cutoff before RFM (Step 6 before Step 7)

If the date cutoff were applied after RFM aggregation, a customer's Recency calculation might use a transaction from September–December 2011 as their "most recent" purchase, making their historical Recency appear lower than it truly is before the cutoff. Applying the cutoff first guarantees:

1. Recency is measured correctly relative to the 2011-09-09 snapshot.
2. The three-month validation window is completely held out for Member 4.

### 9.4 Why StandardScaler (not MinMaxScaler)

StandardScaler is optimal when features are approximately normally distributed after log transformation. MinMaxScaler was considered and rejected because it maps to [0, 1] by dividing by the range `max - min`. The extreme value of £177,729.62 would make that range very large, compressing most customers into a narrow band near zero.

### 9.5 Why no outlier capping

Log transformation already compresses the extreme right tail. For example:
- £177,729.62 → `log1p` ≈ 12.09
- Median £555.015 → `log1p` ≈ 6.32

The ratio shrinks from ~320x to ~2x. Capping would require an arbitrary threshold and would destroy real information about genuine wholesale accounts. If Member 3 found problematic outlier-driven clusters during modelling, capping could be revisited. Member 3's results show the leading K-Means solution is not distorted in that way.

### 9.6 D and M stock codes

D rows are not explicitly excluded by Step 5 but are removed by earlier steps (cancellation/negative-quantity removal and/or missing CustomerID filtering). No D rows survive to the cleaned dataset. M rows (misc/manual) are not excluded — 171 valid M rows survive. These represent legitimate manual adjustment lines that passed all quality criteria.

## 10. EDA cross-references

| EDA Ref | Topic | How addressed |
|---------|-------|---------------|
| Q1 | Missing CustomerID | Removed in Step 2 |
| Q2 | Spend skew | Log-transformed Monetary in Step 8 |
| Q7 | Cancellations | Removed in Step 3 (dual criteria) |
| Q8 | Negative/zero values | Steps 3 + 4 combined |
| Q9 | Non-product codes | Removed POST/DOT/C2/S in Step 5; D removed by earlier steps, M survives |
| Q11 | F-M correlation | Log-transformed BOTH in Step 8 |
| Q12 | Duplicates | Removed first in Step 1 |

## 11. Important notebook

[02_cleaning_and_rfm.ipynb](../notebooks/02_cleaning_and_rfm.ipynb) is the executed Member 2 analysis.

The adjacent readable source file is [02_cleaning_and_rfm.py](../notebooks/02_cleaning_and_rfm.py). It uses `# %%` percent-cell markers to separate each logical section.

The pipeline sections in order:

1. Setup and imports.
2. Download dataset if not present (UCI Online Retail zip).
3. Load raw Excel.
4. Step 1 — duplicate removal.
5. Step 2 — missing CustomerID.
6. Step 3 — cancellations.
7. Step 4 — price <= 0.
8. Step 5 — non-product codes.
9. Step 6 — date cutoff.
10. Save `cleaned_transactions.csv`.
11. Step 7 — build RFM table; defensive Monetary > 0 check.
12. Step 8 — log transform and StandardScaler; save `rfm_table.csv`.
13. Plot: cleaning funnel, raw RFM distributions, transformed distributions, correlation heatmap.

In one flow:

**Download → Load → Deduplicate → Filter IDs → Remove Cancellations → Remove Bad Prices → Remove Non-Products → Date Cutoff → Aggregate RFM → Transform → Scale → Export → Visualise.**

## 12. Important Python libraries used

| Library | Purpose |
| --- | --- |
| `pandas` | Data loading, filtering, groupby aggregation, CSV export |
| `numpy` | `log1p` transformation, array operations |
| `sklearn.preprocessing.StandardScaler` | Zero-mean, unit-variance scaling |
| `matplotlib` | Histogram and bar chart plotting |
| `seaborn` | Correlation heatmap; consistent visual theme |
| `openpyxl` | Required by pandas to read the Excel (.xlsx) file |

## 13. Differences between EDA statistics and post-cutoff RFM

EDA statistics (Q1–Q12) were computed by Member 1 on the **full cleaned dataset without the temporal cutoff**. The RFM table uses only rows before 2011-09-09. This causes expected differences that are not errors:

| Metric | Full-data EDA | Post-cutoff RFM |
|--------|--------------|-----------------|
| Total customers | ~4,338 | 3,362 |
| One-time buyers | ~30% | ~40.6% |
| Spearman(F, M) | ~0.81 | ~0.78 |

The smaller customer count reflects that some customers placed all their orders in the September–December 2011 window and appear in the full EDA but not in the historical RFM. The higher one-time buyer rate reflects that some repeat customers made their second purchase after the cutoff.

## 14. Handoff to Member 3

Member 3 received:

| File | What it contains |
| --- | --- |
| `data/processed/rfm_table.csv` | 3,362 customers x 9 columns; model-ready |
| `data/interim/cleaned_transactions.csv` | 231,806 rows; history only |
| `members/member-2/docs/preprocessing_log.md` | Step-by-step cleaning record and design decisions |
| `members/member-2/figures/` | Four diagnostic charts |

Member 3's input audit confirmed:
- 3,362 rows / 3,362 unique CustomerIDs
- 9 columns
- 0 missing values / 0 infinite values
- 0 duplicate IDs / 0 duplicate rows / 0 duplicate model-feature vectors
- Historical date range: 2010-12-01 through 2011-09-08
- 0 rows on or after the cutoff date
- Transformation relationships verified to absolute tolerance 1e-7
- Scaled feature means effectively 0; population standard deviations effectively 1

## 15. Figures I created

### 15.1 cleaning_funnel.png

**What:** horizontal bar chart showing the row count retained after each of the six cleaning steps and the raw starting count.

**Axes:** step labels on the vertical axis; number of rows on the horizontal axis.

**Observation:** the two largest drops are Step 2 (no CustomerID, −135,037 rows) and Step 6 (date cutoff, −159,638 rows). Steps 3–5 remove considerably fewer rows but address important data-quality issues.

**How I can explain this figure in a viva**

> "The cleaning funnel shows that losing rows without CustomerIDs and applying the temporal cutoff account for the majority of the reduction. The other steps remove far fewer rows but address important data-quality issues like cancellations and service charges."

### 15.2 rfm_distributions_raw.png

**What:** three side-by-side histograms of raw Recency, Frequency and Monetary, each with a red dashed median line.

**Observation:** Frequency and Monetary have strong right skews. The median for Monetary (£555) is far below the mean due to the extreme upper tail. This plot motivates the log transformation.

**How I can explain this figure in a viva**

> "The raw distributions show that Frequency and Monetary are right-skewed — most customers have small values, but a few have very large ones. This skew would distort distance-based models if untreated."

### 15.3 rfm_distributions_transformed.png

**What:** three histograms of `Recency_scaled`, `Frequency_scaled` and `Monetary_scaled` after log transformation and StandardScaler. The red dashed line marks the mean (0).

**Observation:** after transformation, Frequency_scaled and Monetary_scaled are approximately symmetric around zero. The mass of one-order customers creates a visible spike on the left of Frequency_scaled.

**How I can explain this figure in a viva**

> "After log transformation and standardisation, the features are roughly centred at zero. The one-order customer spike is still visible in Frequency_scaled — the log cannot fully smooth a discrete mass — but it is far less extreme than the raw distribution."

### 15.4 rfm_correlation_heatmap.png

**What:** a Pearson correlation heatmap of the three scaled features, shown as the lower triangle.

**Observation:** Frequency_scaled and Monetary_scaled have a correlation of approximately **0.78**. Recency_scaled has a mild negative correlation with the other two.

**How I can explain this figure in a viva**

> "The heatmap confirms that Frequency and Monetary are correlated because customers who order often also tend to spend more. Scaling does not remove this overlap — it is a known limitation of RFM that Member 3's study acknowledges."

## 16. Additional notes

### 16.1 InvoiceNo data type change

Raw InvoiceNo is string/object type because cancellation invoices contain values such as `"C536379"`. After Step 3 removes all C-prefixed invoices, remaining values are purely numeric. When exported to CSV and re-read by pandas, InvoiceNo is inferred as `int64`. This is not a data-loss issue but downstream code that expects string InvoiceNo should be aware of it.

### 16.2 Description nulls

The raw dataset contains 1,454 rows with null Description values. No dedicated rule removes them. They are eliminated as a side effect of earlier cleaning steps (primarily Step 2). The cleaned dataset has 0 null Description values.

### 16.3 Monetary > 0 check

After Step 3 removes cancellations and negative-quantity rows, all remaining customers already have positive net spend. The defensive `Monetary > 0` check at the start of Step 7 found 0 customers to remove. Member 1's EDA predicted ~50 such customers, but that analysis included cancellation transactions in the Monetary calculation.

## 17. Limits and group decisions

- **Monetary is not profit:** it is the sum of `Quantity x UnitPrice` for retained positive lines. Cancellations are removed, but returns that do not generate a cancellation invoice are not separately identified. Monetary does not reflect net revenue after returns.
- **M-code rows are retained:** 171 manual/miscellaneous rows with StockCode M that passed all quality criteria are included.
- **Cancellation logic is OR:** the implementation uses OR rather than the AND described in the original documentation. This is the correct interpretation — it is documented explicitly and not silently changed.
- **Date cutoff is a strict boundary:** exactly 2011-09-09 00:00:00. Transactions on that date are excluded (`>= SNAPSHOT_DATE` is removed). Recency for a customer whose last purchase was 2011-09-08 is 1 day.
- **No capping:** extreme spend values (up to ~£177k) are retained and compressed by the log step. Member 3's results show the leading K-Means solution is not distorted by this.
- **Scaler fit on the full historical cohort:** StandardScaler was fitted on all 3,362 historical customers. If new customers were to be added later, the fitted scaler parameters would need to be re-applied consistently or re-fitted.

## 18. Key numbers I should remember

| Item | Value |
| --- | --- |
| Raw rows | 541,909 |
| Cleaned transaction rows | 231,806 |
| Historical customers (RFM table) | 3,362 |
| One-time buyers | 1,366 (40.63%) |
| Snapshot date | 2011-09-09 00:00:00 |
| Recency range | 0–281 days; median 73 |
| Frequency range | 1–131 invoices; median 2 |
| Monetary range | £2.90–£177,729.62; median £555.015 |
| Spearman F-M correlation (post-cutoff) | ~0.78 |
| Scaled mean / population std | ≈ 0 / ≈ 1 (for all three) |
| Rows removed — duplicates | 5,268 |
| Rows removed — no CustomerID | 135,037 |
| Rows removed — cancellations | 8,872 |
| Rows removed — price <= 0 | 40 |
| Rows removed — non-product codes | 1,248 |
| Rows removed — date cutoff | 159,638 |

## 19. Important preprocessing decisions (summary table)

| Decision | Alternatives considered | Reason | Evidence |
| --- | --- | --- | --- |
| Remove duplicates first (Step 1) | Remove after filtering | Prevents inflated Monetary from duplicated line items | EDA Q12 |
| OR for cancellation detection | AND only | Catches stray negative-quantity rows without C-prefix | ~0 additional rows but more defensive |
| Date cutoff before RFM | RFM then cut | Ensures correct Recency and preserves validation window | Project design; Member 4 handoff |
| log1p for Frequency and Monetary | log (fails on 0); no transform; square root | Safe for integer >= 1 inputs; compresses right tail sufficiently | Visible skew in raw distributions |
| No log for Recency | Log Recency | Bounded range; less severe skew in this dataset | Raw Recency distribution |
| StandardScaler | MinMaxScaler; RobustScaler; no scaling | Post-log distributions approximately normal; MinMaxScaler sensitive to outliers | Transformed distribution plots |
| No outlier capping | IQR cap; percentile cap | Log transformation already compresses extreme tails | log1p(177729) ≈ 12.09 vs log1p(555) ≈ 6.32 |
| Retain D/M codes (M only survives) | Exclude D and M | D removed by earlier steps; M rows are valid purchases | Actual row counts and EDA Q9 |

## 20. Boundary between Member 2 and Member 3

| My Member 2 work | Member 3's next work |
| --- | --- |
| Raw data ingestion and quality validation | Input audit confirming handoff contract |
| Six-step cleaning pipeline | No re-cleaning; use existing cleaned data |
| RFM aggregation (one row per customer) | Use 3,362-customer table directly |
| Log transformation and StandardScaler | Consume existing scaled columns; verify relationships |
| Temporal cutoff enforcement | Fit all models on history only |
| Four diagnostic charts | Model-specific figures (search curves, PCA, sizes) |

Member 3 does not re-run cleaning or re-scale features. The existing `rfm_table.csv` is the single source of truth for all subsequent modelling.

## 21. What Member 3 receives from me

| Handoff | How Member 3 uses it |
| --- | --- |
| `data/processed/rfm_table.csv` | Read directly; use `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled` as feature matrix X |
| `data/interim/cleaned_transactions.csv` | Available for any additional historical analysis if needed |
| `members/member-2/docs/preprocessing_log.md` | Understand cleaning decisions and check provenance |
| `members/member-2/figures/` | Reference the distributions and correlation for the modelling strategy |

Member 3 confirmed the handoff via an automated input audit that checks schema, missing values, duplicate IDs, date bounds, transformation algebra, scaling statistics, and the one-order count.

## 22. Common viva questions and answers

### 1. Why did you use RFM?

RFM gives three interpretable purchasing dimensions — how recently, how often, and how much — that are well-established in retail analytics and directly computable from transactional data without a target label.

### 2. Why remove duplicates before anything else?

If a transaction row is duplicated, any Monetary total computed from it would count the same purchase twice. Removing duplicates first ensures all downstream totals are correct.

### 3. Why is the cancellation criterion OR, not AND?

OR catches both C-prefix rows (the primary indicator) and any stray negative-quantity rows that may lack the C prefix. The AND condition would miss stray rows. The implementation uses OR as the more defensively correct form; this is documented explicitly.

### 4. Why apply the date cutoff before computing RFM?

Recency depends on the customer's most recent purchase relative to the snapshot date. If a post-cutoff transaction were included in the history, Recency would appear artificially low. Applying the cutoff first also ensures the September–December 2011 window is completely reserved for Member 4's validation.

### 5. What is the snapshot date and why 9 September 2011?

The snapshot date is the reference point for Recency. It was chosen to divide the dataset into a historical training window and a three-month held-out validation window for Member 4. The raw data ends in December 2011, so 9 September leaves a ~90-day validation period.

### 6. Why use log1p and not plain log?

`log(0)` is undefined. While Monetary is always positive after cleaning, Frequency starts at 1, and `log(0)` would fail if any zero-frequency customer existed. `log1p` applies `log(1 + value)`, which is safe and consistent for non-negative inputs.

### 7. Why not log-transform Recency?

Recency has a bounded range (0–281 days) with a less severe skew than Frequency or Monetary. Standardising the raw Recency was sufficient. We did not run an explicit experiment comparing logged and unlogged Recency — inheriting Member 1's findings was the appropriate boundary.

### 8. Why StandardScaler and not MinMaxScaler?

After log transformation, the distributions are approximately normal, making StandardScaler's mean-zero unit-variance approach appropriate. MinMaxScaler divides by `max - min`. The extremely high Monetary maximum (£177,729.62) would make that range very large, compressing most customers into a tiny portion of the [0, 1] space.

### 9. What does StandardScaler actually do?

It subtracts each feature's mean and divides by its population standard deviation: `(x - mean) / std`. After this, each feature has approximately mean 0 and standard deviation 1, so no single feature dominates Euclidean distance due to scale alone.

### 10. Why is the sample standard deviation not exactly 1.0?

StandardScaler uses the population standard deviation (denominator n). Pandas `.std()` uses Bessel's correction (denominator n-1). The result is approximately 1.000149 rather than 1.000000. This is expected and not a scaling error.

### 11. Why is there an F-M correlation of 0.78 after scaling?

Scaling removes differences in units and scale but not linear relationships. Customers who order frequently also tend to spend more. The correlation is a structural property of purchasing behaviour, not an artefact of the scale. Member 3 acknowledges this limitation.

### 12. What happened to the ~976 customers between EDA (~4,338) and RFM (3,362)?

They placed all their retained orders in the September–December 2011 validation window. After applying the date cutoff, their historical records are excluded.

### 13. How many one-time buyers are there and why does it matter?

1,366 customers (40.63%) placed exactly one retained invoice. This large discrete mass at Frequency = 1 affects Member 3's baseline scoring — all one-order customers share the same average rank, placing them in Frequency score 2. Frequency score 1 is unused in this dataset.

### 14. What does Monetary measure exactly?

Total positive line spend in GBP, computed as `SUM(Quantity x UnitPrice)` across all retained non-cancellation, non-negative-quantity transactions before the date cutoff. It is not net revenue after returns, not profit.

### 15. Why retain M-code rows?

M rows represent miscellaneous manual transactions that passed all quality criteria — positive price, valid CustomerID, not a cancellation, not a service charge code. Removing them would require additional business knowledge about what they represent. They account for 171 rows out of 391,444 post-Step-5 rows.

## 23. 1-minute explanation of my part

> "I was responsible for cleaning and preprocessing the UCI Online Retail dataset and preparing the customer-level feature table for Member 3's clustering work.
>
> The raw data had 541,909 transaction rows with various quality issues — duplicates, missing customer identifiers, cancellation records, service charge codes, and transactions from a future period we needed to hold out. I applied a six-step pipeline to remove these, ending with 231,806 clean historical transaction rows.
>
> I then aggregated those rows to one row per customer — 3,362 customers — computing Recency in days, Frequency as distinct invoice counts, and Monetary as total GBP spend. I log-transformed Frequency and Monetary to compress their right skews, then standardised all three prepared features with StandardScaler to give each equal weight in distance calculations.
>
> The output is a 3,362-row table with three scaled model-ready columns. Member 3's input audit confirmed all quality checks passed."

## 24. 30-second explanation of my part

> "I cleaned 541,909 raw transactions down to 231,806 historical rows and aggregated them into a 3,362-customer RFM table. I log-transformed the skewed Frequency and Monetary features and standardised all three so Member 3's clustering models treat each feature equally. The table passed Member 3's automated input audit with no issues."

## 25. Final memory summary

**Member 2 in one sentence**

I turned 541,909 raw retail transaction rows into a clean, log-transformed, standardised 3,362-customer RFM table for Member 3 to cluster.

**5 things I must remember**

1. Six cleaning steps in strict order: duplicates → no CustomerID → cancellations → price <= 0 → non-product codes → date cutoff.
2. Snapshot date is 2011-09-09; everything on or after is excluded to preserve the validation window.
3. Log-transform Frequency and Monetary (not Recency) with `log1p`, then StandardScaler all three.
4. `Frequency_scaled` and `Monetary_scaled` are StandardScaler outputs of the log-transformed columns, not the raw columns.
5. Final output: 231,806 cleaned transactions; 3,362-customer RFM table; passed Member 3's input audit with 0 issues.

---

## 26. Worked numerical examples

### 26.1 RFM calculation — three example customers

Assume three customers with the following retained (post-cleaning, pre-cutoff) transactions:

| CustomerID | Invoice date | Invoice | Qty | UnitPrice | Line spend |
| --- | --- | --- | --- | --- | --- |
| 1001 | 2011-09-08 | I-A | 10 | £2.55 | £25.50 |
| 1001 | 2011-09-08 | I-A | 5 | £3.30 | £16.50 |
| 1001 | 2011-07-01 | I-B | 2 | £12.00 | £24.00 |
| 1002 | 2011-01-15 | I-C | 100 | £1.25 | £125.00 |
| 1003 | 2011-03-10 | I-D | 3 | £5.00 | £15.00 |
| 1003 | 2011-06-20 | I-E | 1 | £8.99 | £8.99 |
| 1003 | 2011-08-05 | I-F | 7 | £4.50 | £31.50 |

Snapshot date = 2011-09-09.

**Customer 1001:**
- Last invoice: I-A on 2011-09-08 → Recency = (2011-09-09 − 2011-09-08).days = **1 day**
- Distinct invoices: {I-A, I-B} = **2**
- Total spend: £25.50 + £16.50 + £24.00 = **£66.00**
- RFM: (1, 2, 66.00)

**Customer 1002:**
- Last invoice: I-C on 2011-01-15 → Recency = (2011-09-09 − 2011-01-15).days = **237 days**
- Distinct invoices: {I-C} = **1**
- Total spend: £125.00
- RFM: (237, 1, 125.00)

**Customer 1003:**
- Last invoice: I-F on 2011-08-05 → Recency = (2011-09-09 − 2011-08-05).days = **35 days**
- Distinct invoices: {I-D, I-E, I-F} = **3**
- Total spend: £15.00 + £8.99 + £31.50 = **£55.49**
- RFM: (35, 3, 55.49)

### 26.2 Log transformation — worked example

Continuing with the three customers above:

| CustomerID | Recency | Frequency | Monetary | Log_Frequency | Log_Monetary |
| --- | --- | --- | --- | --- | --- |
| 1001 | 1 | 2 | 66.00 | log1p(2) = 1.0986 | log1p(66.00) = 4.2047 |
| 1002 | 237 | 1 | 125.00 | log1p(1) = 0.6931 | log1p(125.00) = 4.8283 |
| 1003 | 35 | 3 | 55.49 | log1p(3) = 1.3863 | log1p(55.49) = 4.0284 |

Notice: Customer 1001 (Frequency = 2) and Customer 1003 (Frequency = 3) have very similar
Log_Frequency values (1.10 vs 1.39), whereas in raw space they differ by 50%.
This is the compression effect of the log.

### 26.3 StandardScaler — worked example

Suppose these three customers are the entire population (in reality, n = 3,362).
The StandardScaler formula is `(x − μ) / σ` where μ is the population mean
and σ is the population standard deviation.

**For Recency (raw):**
- Values: [1, 237, 35]
- μ_R = (1 + 237 + 35) / 3 = 91.0
- σ_R = sqrt(((1−91)² + (237−91)² + (35−91)²) / 3) = sqrt((8100 + 21316 + 3136) / 3) = sqrt(10850.67) ≈ 104.16

Scaled Recency values:
- Customer 1001: (1 − 91) / 104.16 ≈ −0.864
- Customer 1002: (237 − 91) / 104.16 ≈ +1.402
- Customer 1003: (35 − 91) / 104.16 ≈ −0.538

**Interpretation:** Customer 1001 is 0.86 standard deviations below the mean Recency
(i.e., more recent than average). Customer 1002 is 1.40 standard deviations above (i.e.,
less recent than average — they have been quiet for longer).

In the actual dataset with 3,362 customers, the scaler uses the empirical mean and std
of the historical cohort. The principles are identical.

---

## 27. Visualisation — figures created

### 27.1 cleaning_funnel.png

**Location:** [`figures/cleaning_funnel.png`](../figures/cleaning_funnel.png)

**What:** a horizontal bar chart showing the number of rows remaining after each of the six
cleaning steps and the raw starting count. Each bar is labelled with its exact row count.

**Axes:** step labels on the vertical axis; number of rows on the horizontal axis.

**Key observations:**
- The two largest drops are Step 2 (−135,037 rows for missing CustomerID) and Step 6
  (−159,638 rows for the date cutoff). Together they account for 97.8% of all rows removed.
- Steps 3, 4 and 5 remove far fewer rows (8,872 + 40 + 1,248 = 10,160) but address important
  data-quality issues: cancellations produce negative spend, zero prices create meaningless
  line values, and service codes inflate non-product Monetary.

**What to say in a viva:**
> "The cleaning funnel shows that the majority of row reduction comes from two structural
> decisions: excluding unidentified customers and applying the temporal cutoff. The smaller
> steps address genuine data-quality issues. All six steps are necessary."

**Limitations:** the funnel shows row counts, not the proportion of *customers* affected.
A customer represented by many rows may lose some rows in Steps 3–5 but remain in the dataset.

### 27.2 rfm_distributions_raw.png

**Location:** [`figures/rfm_distributions_raw.png`](../figures/rfm_distributions_raw.png)

**What:** three side-by-side histograms of raw Recency, Frequency and Monetary (50 bins each),
each with a red dashed median line.

**Key observations:**
- **Recency:** distribution spread from 0 to 281 days. The histogram shows a reasonably
  distributed shape with a larger count near the lower end — many customers have bought
  relatively recently.
- **Frequency:** strong right skew. The vast majority of customers have 1–5 orders; a handful
  have 50+ (maximum 131). The median is 2.
- **Monetary:** very strong right skew. Most customers cluster near zero relative to the maximum
  of £177,729.62. The median (£555.015) is far below the mean (£1,594.71) — the mean is inflated
  by extreme wholesale customers.

**What this motivates:** the Frequency and Monetary skews directly justify the log transformation.
Recency's more bounded shape justifies leaving it untransformed.

**Limitations:** the y-axis (customer count) makes high-Monetary customers look rare. A log-scale
axis on y would further compress the visualization. The histograms cannot show the bimodality
of the one-order customer mass (all at Frequency = 1) versus repeat buyers.

### 27.3 rfm_distributions_transformed.png

**Location:** [`figures/rfm_distributions_transformed.png`](../figures/rfm_distributions_transformed.png)

**What:** three histograms of `Recency_scaled`, `Frequency_scaled`, and `Monetary_scaled` after
log transformation and StandardScaler. The red dashed line marks the mean (0).

**Key observations:**
- **Recency_scaled:** reasonably symmetric around zero; slightly right-skewed (more customers
  have above-average Recency = less recent). No log transformation was applied.
- **Frequency_scaled:** approximately symmetric around zero with a visible spike on the left.
  This spike represents the 1,366 one-order customers (Frequency = 1, log1p ≈ 0.693), all at
  nearly the same standardized value. The log compresses but cannot eliminate a discrete mass.
- **Monetary_scaled:** the most symmetric of the three after transformation. The log reduces
  the extreme right tail dramatically.

**What this shows:** the log transformation substantially improves distributional symmetry
for Frequency and Monetary, but does not produce perfect Gaussian distributions.
The one-order customer spike in Frequency_scaled is a structural property of the data.

**What to say in a viva:**
> "After log transformation and standardisation, the features are much more centred around zero.
> The Frequency spike on the left is the one-order customer mass — 40.63% of customers ordered
> exactly once. The log cannot smooth a discrete mass; the transformation is beneficial but not
> a normalization guarantee."

### 27.4 rfm_correlation_heatmap.png

**Location:** [`figures/rfm_correlation_heatmap.png`](../figures/rfm_correlation_heatmap.png)

**What:** a lower-triangle Pearson correlation heatmap of the three scaled features, annotated
with correlation coefficients. Uses a red-yellow-blue diverging colormap.

**Key observations:**
- **Frequency_scaled ↔ Monetary_scaled:** strong positive correlation ≈ **0.78** (displayed as
  the post-scaling Pearson r; the pre-scaling Spearman r was ≈ 0.78 as well). Customers who
  order frequently also tend to spend more.
- **Recency_scaled ↔ Frequency_scaled:** mild negative correlation. More recent customers
  tend to have placed more orders. (Note: Recency direction — lower Recency = more recent;
  negative correlation means recently-purchasing customers have higher Frequency.)
- **Recency_scaled ↔ Monetary_scaled:** mild negative correlation. More recent customers
  tend to have spent more.

**What to say in a viva:**
> "The heatmap shows that Frequency and Monetary are highly correlated — customers who order
> more also spend more. This is not an artefact of scaling; it is a property of purchasing
> behaviour. Scaling does not remove correlations, only scale differences. Member 3 acknowledges
> this as a known limitation: the two features partly measure the same underlying loyalty."

**Interpretation limitations:**
- Pearson correlation measures linear association. The actual relationship between (log-transformed)
  Frequency and Monetary may include non-linear components.
- Correlation does not imply causation. We cannot conclude that ordering more *causes* more spending;
  both are consequences of purchasing patterns.
- The heatmap shows the lower triangle only (upper triangle is masked for readability).

---

## 28. Ethics, limitations and bias considerations

### 28.1 Population bias: identified customers only

The cleaning pipeline removes approximately 135,037 rows (25%) because they lack a CustomerID.
These are likely guest checkouts, phone orders, or till-sales where no account was created.

**Consequence:** the RFM model and all downstream segments describe only identified customers.
If guests systematically differ from account holders — younger, one-time, or anonymous purchasers
— then the segments we produce do not represent the full customer population.

**Implication:** any business action based on the segments (e.g., a targeted email campaign)
can only reach identified customers. Recommendations should clearly state this scope.

### 28.2 Temporal bias: Christmas seasonality

EDA Q5 notes that revenue climbs steeply toward a peak in November 2011. The future validation
window (September–December 2011) contains the Christmas peak. This means:

- Repeat-purchase rates observed by Member 4 in the validation window may be higher than
  in a typical quarter.
- Seasonal behaviour may not represent the retailers' year-round customer dynamics.
- Conclusions about "loyal" versus "at-risk" customers drawn from Christmas-period behaviour
  should be qualified accordingly.

### 28.3 Gross spend, not net revenue

Monetary represents the sum of positive retained line spend. Cancellation rows are removed,
but returns that did not generate a C-prefix invoice are not separated from genuine sales.
Some customers' Monetary values include spend that was later returned — making their apparent
spend higher than their effective spend. This could cause customers to appear in a
"higher-spending" segment when their net spend is much lower.

### 28.4 EDA Q8 — sale/reversal pairs

Member 1's EDA (Q8) notes that the largest positive quantity (80,995 items) and the
corresponding cancellation (−80,995 items) are a matched sale/reversal pair. The pipeline
removes the cancellation but retains the large sale. This inflates Monetary for the affected
customer. The preprocessing log (D/M code section) notes this is acknowledged but not corrected.

### 28.5 Segmentation as description, not prescription

Customer segments are descriptive constructs based on historical purchasing patterns.
They do not establish:
- Why customers behave as they do (causal explanation).
- What actions will change customer behaviour (causal intervention).
- Whether a customer labelled "at risk" will actually churn (predictive outcome).

Using segment labels to justify differential treatment of customer groups (e.g., targeted
discounts for "high-value" customers, ignoring "low-value" ones) should be done with explicit
acknowledgement that the segments are based on historical transaction patterns only.

### 28.6 Wholesale customer sensitivity

The top-spending customers are likely wholesale accounts rather than individual retail
consumers. Their presence in the dataset affects cluster centroids. Member 3's K-Means k=2
analysis showed one cluster with median Monetary £1,561 (mean £3,300) — suggesting wholesale
accounts disproportionately influence that cluster's profile. Any business recommendation based
on "high-value customers" should clarify whether it targets retail consumers or wholesale buyers.

---

## 29. Additional viva questions and answers

### 16. Does log1p guarantee that the data becomes Gaussian (normally distributed)?

No. Log1p compresses the right tail and generally makes a right-skewed distribution more
symmetric, but it does not guarantee normality. In our dataset, the 1,366 customers with
Frequency = 1 all map to log1p(1) ≈ 0.693. This creates a discrete spike in Log_Frequency
that persists after log transformation and is clearly visible in `rfm_distributions_transformed.png`.
A formal normality test (e.g., Shapiro-Wilk) would likely reject normality for Frequency_scaled.

### 17. Why does Member 3's K-Means give Frequency score 1 as "unused"?

Member 3's rule-based baseline uses average-rank-based quintile scoring. With 1,366 customers
all having Frequency = 1, they all receive the same average rank. That rank falls in the range
corresponding to score 2 (not score 1). Score 1 would require customers with lower Frequency
than all one-order customers — but Frequency cannot be less than 1. This is an inherent
consequence of the discrete frequency distribution and the midrank scoring method.

### 18. What is the difference between Recency_scaled and Frequency_scaled in terms of what was actually scaled?

`Recency_scaled` is `StandardScaler(raw Recency)`. `Frequency_scaled` is
`StandardScaler(log1p(Frequency))`. Despite their similar-looking names, the inputs
to the scaler are different: Recency goes in raw (as elapsed days), Frequency goes in
as its log-transformed version. This is documented explicitly and verified by Member 3
to tolerance 1e-7.

### 19. Why does the correlation heatmap show Pearson r, not Spearman r?

The heatmap shows Pearson correlation of the *scaled* features (which are the inputs
to Member 3's clustering). Pearson correlation is computed as `rfm[scaled_cols].corr()`.
Member 1's EDA used Spearman correlation (≈ 0.81) on the raw unscaled columns because
Spearman is rank-based and more robust to the skewed raw distributions. The post-cutoff
Spearman r on scaled features is approximately 0.78 — close but slightly lower because
the temporal cutoff excludes some customers.

### 20. If you applied MinMaxScaler instead of StandardScaler, how would the output change?

MinMaxScaler would map each feature to [0, 1] using `(x − min) / (max − min)`. For Monetary:
min = £2.90, max = £177,729.62, range = £177,726.72. The median customer (£555.015) would
be mapped to (555.015 − 2.90) / 177726.72 ≈ 0.0031 — less than 0.3% of the [0, 1] range.
All customers except the extreme wholesale buyer would be compressed into a narrow band near zero.
StandardScaler avoids this by centering at the mean and dividing by standard deviation,
giving a more interpretable spread across all customers.

### 21. What would happen if you applied the date cutoff after computing RFM instead of before?

A customer who last purchased in November 2011 (after the cutoff) would have their
November date used as their "last invoice date." Their Recency would be computed as
(2011-09-09 − 2011-11-20).days = −72 days — a negative Recency, which is impossible.
More subtly, their Monetary would include November spend, contaminating the historical
measure with future transactions. Applying the cutoff first prevents both issues.

### 22. How would you verify that the cleaned dataset contains no future transactions?

Run: `assert (df["InvoiceDate"] < pd.Timestamp("2011-09-09")).all()` on the cleaned
transactions file. Member 3's input audit performs this check: the result was 0 rows
on or after the cutoff date, and the maximum transaction date was 2011-09-08 19:58:00.

### 23. A reviewer says "your Monetary feature is biased because it includes a £177,000 customer." How would you respond?

The £177,729.62 customer is a genuine wholesale account, not a data error. The log1p
transformation reduces their relative distance from the median from approximately 320x
to 1.9x. No outlier capping was applied because setting an arbitrary threshold (e.g.,
£50,000) would require business knowledge about what constitutes "too extreme" — knowledge
we do not have at the preprocessing stage. Member 3's leading clustering result (K-Means k=2,
silhouette 0.4114) was not distorted by this extreme value. If this customer were capped,
the change would be recorded as a deliberate preprocessing decision, not a silent correction.

### 24. Why did you not include a "return rate" feature?

A return rate (returned items ÷ total items) would require matching each cancellation invoice
to its original sale invoice. The UCI Online Retail data does not provide explicit links between
cancellation and original invoices; the match would have to be approximated by customer,
product, and date — a complex operation beyond the scope of this preprocessing stage.
It is noted as a potential future improvement.

### 25. The preprocessing log says "AND" for cancellations but the code uses "OR." Which is correct and why?

The code is correct. Using OR removes a row if it has a C-prefix invoice *or* negative quantity.
This catches:
1. C-prefix invoices with non-negative quantities (standard cancellation format).
2. Negative-quantity rows without a C-prefix (defensive catch for edge cases).
Using AND would miss case (2). The log description was informal; the implemented OR logic is
the authoritative definition. Member 3's strategy document records this discrepancy explicitly.

---

## 30. Final revision summary

### In one sentence

Member 2 converted 541,909 raw retail transaction rows into a clean, log-transformed,
standardised 3,362-customer RFM table for Member 3's clustering work.

### Pipeline: six steps in strict order

```
541,909 raw rows
   ↓ Step 1: drop_duplicates()          → −5,268   → 536,641
   ↓ Step 2: dropna(CustomerID)         → −135,037 → 401,604
   ↓ Step 3: C-prefix OR Qty < 0        → −8,872   → 392,732
   ↓ Step 4: UnitPrice <= 0             → −40      → 392,692
   ↓ Step 5: POST, DOT, C2, S codes     → −1,248   → 391,444
   ↓ Step 6: InvoiceDate >= 2011-09-09  → −159,638 → 231,806 cleaned rows
   ↓ Step 7: groupby(CustomerID) RFM    → 3,362 customers
   ↓ Step 8: log1p(F,M) + StandardScaler → 9-column rfm_table.csv
```

### Key numbers

| Item | Value |
| --- | --- |
| Snapshot date | 2011-09-09 |
| Cleaned transaction rows | 231,806 |
| RFM customers | 3,362 |
| One-time buyers | 1,366 (40.63%) |
| Recency range | 0–281 days; median 73 |
| Frequency range | 1–131; median 2 |
| Monetary range | £2.90–£177,729.62; median £555.015 |
| F–M Spearman (post-cutoff) | ≈ 0.78 |
| Scaled mean / population std | ≈ 0 / ≈ 1 (all three) |

### Three things that make RFM unique to this dataset

1. One-time buyer mass (40.63%) creates a discrete spike in Log_Frequency.
2. Wholesale extreme (£177,729.62) justifies log1p but not capping.
3. Temporal cutoff reduces customer count from ~4,338 (full EDA) to 3,362 (historical only).

### Three common misstatements to avoid in a viva

1. "Log transformation makes the data Gaussian." → Wrong. It compresses the tail; normality is not guaranteed.
2. "Frequency_scaled and Monetary_scaled are the raw features standardised." → Wrong. They are the *log-transformed* features standardised.
3. "The cancellation criterion is AND." → Wrong. The code uses OR. AND is what the informal log description said; OR is implemented.
