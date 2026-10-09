# Preprocessing & Feature Engineering Log (Member 2)

## Cleaning Pipeline

| Step | Action | Rows Before | Rows Removed | Rows After |
|------|--------|-------------|--------------|------------|
| -- | Raw | -- | -- | 541,909 |
| 1 | 1. Duplicates removed | 541,909 | 5,268 | 536,641 |
| 2 | 2. No CustomerID | 536,641 | 135,037 | 401,604 |
| 3 | 3. Cancellations | 401,604 | 8,872 | 392,732 |
| 4 | 4. Price <= 0 | 392,732 | 40 | 392,692 |
| 5 | 5. Non-product codes | 392,692 | 1,248 | 391,444 |
| 6 | 6. Date cutoff | 391,444 | 159,638 | 231,806 |

**Final cleaned transactions**: 231,806 rows
**Final RFM table**: 3,362 customers

## Design Decisions

### 1. Duplicate removal first (Step 1)
Duplicates are removed before any totals are calculated, as recommended by EDA Q12.
This prevents inflated Monetary values from duplicated line items.

### 2. Cancellation detection (Step 3)
The implemented criterion is a **union (OR)**: a row is removed if InvoiceNo starts with 'C' **or** Quantity < 0 (`mask_cancel = mask_c_prefix | mask_neg_qty`). This catches C-prefix invoices and any stray negative-quantity rows that lack the C-prefix. The diagnostic output showed ~0 negative-only rows (no C-prefix) after earlier filtering, but OR is the more defensively correct form. An earlier informal description in this log used AND; the code is the authoritative implementation.

### 3. Non-product codes (Step 5)
Removed: POST, DOT, C2, S (delivery charges, carriage, samples).
D and M are not explicitly excluded by the non-product-code filter. In the final cleaned
dataset, no D rows remain because they were removed by earlier transaction-quality rules
(cancellation/negative-quantity removal and/or missing CustomerID filtering), while valid
M rows (171 rows) remain.

### 4. Date cutoff before RFM (Step 6 before Step 7)
The cutoff date is 2011-09-09, which serves as the snapshot for Recency calculation.
Cutting dates before computing RFM ensures the future validation window (starting 2011-09-09) is reserved for Member 4. The end of that window is not set by Member 2; Member 4 must confirm the exact end date based on the raw dataset and the agreed evaluation design.

### 5. Zero-spend customers — VALIDATED
Validated that all customers have Monetary > 0 before log transformation.
Result: 0 customers required removal. After Steps 1–6 remove cancellations and
negative-quantity rows, all remaining customers already have positive net spend.
The EDA (Q2) predicted ~50 customers with zero/negative net spend, but that analysis
included cancellation transactions in the monetary calculation. The Monetary > 0 filter
is retained as a defensive data-quality check.

### 6. Scaler -- StandardScaler
After log-transformation, Frequency and Monetary are approximately normally distributed.
StandardScaler centers features at mean=0 with unit variance, which is optimal for K-Means (equal feature contribution to distance).
MinMaxScaler was considered but rejected because it is more sensitive to remaining outliers.

### 7. Outlier capping -- NOT APPLIED
Log-transformation already compresses the extreme right tail (e.g., GBP 280k -> ~12.5 on log scale vs median GBP 674 -> ~6.5).
Capping would introduce an arbitrary threshold and destroy real information about wholesale accounts.
If Member 3 discovers problematic outlier clusters during modelling, capping can be revisited.

## EDA Cross-references

| EDA Ref | Topic | How addressed |
|---------|-------|---------------|
| Q1 | Missing CustomerID | Removed in Step 2 |
| Q2 | Spend skew | Log-transformed Monetary in Step 8 |
| Q7 | Cancellations | Removed in Step 3 (dual criteria) |
| Q8 | Negative/zero values | Steps 3 + 4 combined |
| Q9 | Non-product codes | Removed POST/DOT/C2/S in Step 5; D/M not excluded (D removed by earlier steps, M survives) |
| Q11 | F-M correlation | Log-transformed BOTH in Step 8 |
| Q12 | Duplicates | Removed first in Step 1 |

## Additional Notes

### Description nulls
The raw dataset contains 1,454 rows with missing Description values. These are not removed
through a dedicated cleaning rule. All such rows were eliminated as a consequence of earlier
cleaning steps, primarily CustomerID/data-quality filtering (Step 2). The cleaned dataset
contains 0 null Description values.

### InvoiceNo data type
Raw InvoiceNo is string/object dtype because cancellation invoices contain values such as
"C536379". After Step 3 removes all C-prefixed invoices, the remaining values are purely
numeric. When exported to CSV and re-read, pandas infers InvoiceNo as int64. This is not
a data-loss issue but should be noted for downstream code that may expect string InvoiceNo.

### EDA statistics vs post-cutoff RFM values
EDA statistics (Q1–Q12 in `eda_insight_log.csv`) were computed on the full cleaned dataset
without the temporal cutoff. The RFM modelling table uses only rows before 2011-09-09,
reserving the period from 2011-09-09 onward as a validation window for Member 4 (Member 4
must confirm the final evaluation end date based on raw data and evaluation design). This causes expected
differences:

| Metric | Full-data EDA | Post-cutoff RFM |
|--------|--------------|------------------|
| Total customers | ~4,338 | 3,362 |
| One-time buyers | ~30% | ~40.6% |
| Spearman(F, M) | ~0.81 | ~0.78 |

These differences are expected and do not indicate an error.

## Output Files

- `data/interim/cleaned_transactions.csv` -- 231,806 rows
- `data/processed/rfm_table.csv` -- 3,362 customers
- `members/member-2/figures/cleaning_funnel.png`
- `members/member-2/figures/rfm_distributions_raw.png`
- `members/member-2/figures/rfm_distributions_transformed.png`
- `members/member-2/figures/rfm_correlation_heatmap.png`

## Related Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Preprocessing strategy | `members/member-2/docs/preprocessing_strategy.md` | Full pipeline plan and rationale |
| Preprocessing decisions | `members/member-2/docs/preprocessing_decisions.md` | 11 structured decision records with evidence |
| Handoff to Member 3 | `members/member-2/docs/member2_handoff.md` | Schema, temporal boundary, safe usage |
| Reproducibility guide | `members/member-2/docs/reproducibility.md` | How to re-run and validate safely |
| Study guide | `members/member-2/docs/study.md` | Complete viva preparation (30 sections, 25 Q&As) |
| Data dictionary | `docs/data-dictionary/member2_data_dictionary.md` | Full column definitions for both output files |

## Cancellation criterion — note on earlier wording

The Design Decisions section (Step 3) above now correctly states OR as the implemented criterion.
An earlier version of this log used AND informally. The OR implementation is confirmed by code
line 173 (`mask_cancel = mask_c_prefix | mask_neg_qty`) and is documented in
`members/member-3/docs/modelling_strategy.md` and in
`members/member-2/docs/preprocessing_decisions.md` (Decision M2-03).