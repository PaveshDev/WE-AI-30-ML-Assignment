# Member 2 — Data Quality, Cleaning & Feature Engineering

**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Dataset:** UCI Online Retail  
**Lens:** Customer Segmentation

## Responsibility

My role is to convert the raw 541,909-row UCI Online Retail transaction dataset into a
clean, log-transformed, standardised 3,362-customer RFM table for Member 3's clustering work.

| Task | Status |
| --- | --- |
| Six-step cleaning pipeline | ✓ Complete |
| RFM feature engineering | ✓ Complete |
| Log transformation (Frequency, Monetary) | ✓ Complete |
| StandardScaler (all three features) | ✓ Complete |
| Diagnostic charts | ✓ Complete |
| Preprocessing log | ✓ Complete |
| Study guide | ✓ Complete |
| Preprocessing strategy | ✓ Complete |
| Preprocessing decisions | ✓ Complete |
| Handoff to Member 3 | ✓ Complete — Member 3 audit passed |
| Reproducibility guide | ✓ Complete |
| Data dictionary | ✓ Complete |
| Personal learning journey | ✓ Complete (one-page report + working notes) |

---

## File tree

```
members/member-2/
├── README.md                          ← this file
├── docs/
│   ├── preprocessing_log.md           ← cleaning funnel table and design decisions
│   ├── preprocessing_strategy.md      ← full pipeline plan and rationale
│   ├── preprocessing_decisions.md     ← structured decision records (11 decisions)
│   ├── member2_handoff.md             ← technical handoff to Member 3
│   ├── reproducibility.md             ← how to re-run the pipeline safely
│   └── study.md                       ← complete personal study guide and viva prep
├── figures/
│   ├── cleaning_funnel.png            ← rows remaining at each cleaning step
│   ├── rfm_distributions_raw.png      ← raw R, F, M histograms with median lines
│   ├── rfm_distributions_transformed.png ← scaled feature histograms
│   └── rfm_correlation_heatmap.png    ← Pearson correlation of scaled features
└── notebooks/
    ├── 02_cleaning_and_rfm.py         ← readable percent-cell source
    └── 02_cleaning_and_rfm.ipynb      ← executed Jupyter notebook

data/interim/
└── cleaned_transactions.csv          ← 231,806 rows, 8 columns (Member 2 output)

data/processed/
└── rfm_table.csv                     ← 3,362 customers, 9 columns (Member 2 output)

docs/data-dictionary/
└── member2_data_dictionary.md        ← full column definitions for both output files

reports/personal-learning-journeys/member-2/
└── Member2_Personal_Learning_Journey.md  ← personal reflection draft
```

---

## Cleaning pipeline summary

| Step | Action | Rows before | Rows removed | Rows after |
| --- | --- | --- | --- | --- |
| — | Raw | — | — | 541,909 |
| 1 | Exact duplicates removed | 541,909 | 5,268 | 536,641 |
| 2 | No CustomerID | 536,641 | 135,037 | 401,604 |
| 3 | Cancellations (C-prefix OR Qty < 0) | 401,604 | 8,872 | 392,732 |
| 4 | Price <= 0 | 392,732 | 40 | 392,692 |
| 5 | Non-product codes (POST, DOT, C2, S) | 392,692 | 1,248 | 391,444 |
| 6 | Date cutoff (keep rows before 2011-09-09) | 391,444 | 159,638 | 231,806 |

**Final cleaned transactions:** 231,806 rows  
**Final RFM table:** 3,362 customers (one per unique identified CustomerID)

---

## RFM table schema

| Column | Type | Meaning | Use |
| --- | --- | --- | --- |
| CustomerID | int64 | Unique customer identifier | Join key; not a model input |
| Recency | int64 | Days from last retained purchase to 2011-09-09 | Raw profiling |
| Frequency | int64 | Count of distinct retained invoices | Raw profiling |
| Monetary | float64 | Total retained positive line spend in GBP | Raw profiling |
| Log_Frequency | float64 | `log1p(Frequency)` | Intermediate |
| Log_Monetary | float64 | `log1p(Monetary)` | Intermediate |
| Recency_scaled | float64 | StandardScaler applied to raw Recency | **Model input X** |
| Frequency_scaled | float64 | StandardScaler applied to Log_Frequency | **Model input X** |
| Monetary_scaled | float64 | StandardScaler applied to Log_Monetary | **Model input X** |

> **Important:** `Frequency_scaled` and `Monetary_scaled` are standardized versions of
> the **log-transformed** columns, not the raw columns. No further transformation is needed.

---

## Reproducibility

> [!WARNING]
> The script contains a hard-coded absolute `PROJECT_ROOT` path. Update this before running.

From the repository root with a Python 3.11+ virtual environment:

```powershell
# Install dependencies
.venv\Scripts\python.exe -m pip install pandas numpy matplotlib seaborn openpyxl scikit-learn

# Run the pipeline (updates output files and figures)
.venv\Scripts\python.exe members\member-2\notebooks\02_cleaning_and_rfm.py
```

**Do not re-run if Member 3's results are already committed** — re-running changes the SHA-256
hashes of the output files recorded in Member 3's `run_manifest.json`.

Validation commands (read-only, no output changes):

```python
import pandas as pd, numpy as np
rfm = pd.read_csv("data/processed/rfm_table.csv")
assert rfm.shape == (3362, 9)
assert rfm["CustomerID"].nunique() == 3362
assert rfm.isnull().sum().sum() == 0
print("RFM table OK:", rfm.shape)
```

Full instructions: [`docs/reproducibility.md`](docs/reproducibility.md)

---

## Documentation links

| Document | Link | Purpose |
| --- | --- | --- |
| Study guide | [`docs/study.md`](docs/study.md) | Complete viva preparation guide (24 sections, 22+ Q&As) |
| Preprocessing log | [`docs/preprocessing_log.md`](docs/preprocessing_log.md) | Cleaning funnel, EDA cross-refs, design decisions |
| Preprocessing strategy | [`docs/preprocessing_strategy.md`](docs/preprocessing_strategy.md) | Full pipeline plan and rationale |
| Preprocessing decisions | [`docs/preprocessing_decisions.md`](docs/preprocessing_decisions.md) | 11 structured decision records |
| Handoff to Member 3 | [`docs/member2_handoff.md`](docs/member2_handoff.md) | Schema, temporal boundary, limitations |
| Reproducibility guide | [`docs/reproducibility.md`](docs/reproducibility.md) | Environment, paths, validation commands |
| Data dictionary | [`../../docs/data-dictionary/member2_data_dictionary.md`](../../docs/data-dictionary/member2_data_dictionary.md) | Full column definitions |
| Personal learning journey | [`../../reports/personal-learning-journeys/member-2/Member2_Personal_Learning_Journey.md`](../../reports/personal-learning-journeys/member-2/Member2_Personal_Learning_Journey.md) | Reflection draft |

---

## Figures

| Figure | Purpose |
| --- | --- |
| [`figures/cleaning_funnel.png`](figures/cleaning_funnel.png) | Horizontal bar chart showing rows retained after each cleaning step |
| [`figures/rfm_distributions_raw.png`](figures/rfm_distributions_raw.png) | Raw R, F, M histograms showing right skew |
| [`figures/rfm_distributions_transformed.png`](figures/rfm_distributions_transformed.png) | Scaled R, F, M histograms after log transform + StandardScaler |
| [`figures/rfm_correlation_heatmap.png`](figures/rfm_correlation_heatmap.png) | Pearson correlation heatmap (F–M ≈ 0.78) |

---

## Handoff to Member 3 — summary

| What | Where | Status |
| --- | --- | --- |
| 3,362-customer RFM table | `data/processed/rfm_table.csv` | ✓ Audit passed |
| 231,806-row cleaned transactions | `data/interim/cleaned_transactions.csv` | ✓ Available |
| Temporal boundary: 2011-09-09 00:00:00 | Enforced by date cutoff in Step 6 | ✓ 0 rows on/after cutoff |
| Future period (2011-09-09 onward) | Not included in any output | ✓ Reserved for Member 4 |

Member 3's automated input audit confirmed: 3,362/3,362 unique customers, 0 missing values,
0 infinite values, 0 duplicate IDs, date range 2010-12-01 to 2011-09-08, transformation
relationships verified to tolerance 1e-7, and one-order count = 1,366 (40.63%).

---

## Key numbers

| Item | Value |
| --- | --- |
| Snapshot date | 2011-09-09 |
| Cleaned transactions | 231,806 |
| RFM customers | 3,362 |
| One-time buyers | 1,366 (40.63%) |
| Recency range | 0–281 days (median 73) |
| Frequency range | 1–131 invoices (median 2) |
| Monetary range | £2.90–£177,729.62 (median £555.015) |
| Post-cutoff Spearman F–M | ≈ 0.78 |
