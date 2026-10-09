# Member 2 Reproducibility Guide

**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Notebook:** `members/member-2/notebooks/02_cleaning_and_rfm.py` / `.ipynb`

This guide explains how to safely re-run the Member 2 preprocessing pipeline, what
to expect from each output, how to validate results, and what risks to avoid.

---

## 1. Environment requirements

| Package | Purpose | Minimum tested version |
| --- | --- | --- |
| Python | Runtime | 3.11 |
| pandas | Data loading, filtering, groupby, export | Latest compatible with Python 3.11 |
| numpy | `log1p` transformation, array ops | Latest |
| scikit-learn | `StandardScaler` | Latest |
| matplotlib | Histogram and bar chart plotting | Latest |
| seaborn | Correlation heatmap, visual theme | Latest |
| openpyxl | Required by pandas to read `.xlsx` | Latest |

The project-level `requirements.txt` lists all these packages (except openpyxl, which is
implicitly required for Excel loading).

**Install inside a virtual environment from the repository root:**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install pandas numpy matplotlib seaborn openpyxl scikit-learn
```

On Linux/macOS:

```bash
python -m venv .venv
.venv/bin/python -m pip install pandas numpy matplotlib seaborn openpyxl scikit-learn
```

---

## 2. Known path issue

The script `02_cleaning_and_rfm.py` contains a hard-coded absolute path:

```python
PROJECT_ROOT = r"c:\Users\ASUS\Music\WE-AI-30-ML-Assignment"
```

**This path will not work on other machines.** Before running the script, update this
line to your own repository root. For example:

```python
PROJECT_ROOT = r"C:\path\to\WE-AI-30-ML-Assignment"
```

> [!CAUTION]
> Do not commit the changed path back to the repository. The path change is local only.
> A future improvement would be to use upward directory discovery (as Member 3 does)
> or an environment variable.

---

## 3. Input files required

| File | Location | How to obtain |
| --- | --- | --- |
| `Online Retail.xlsx` | `data/raw/` | Downloaded automatically by the script if absent |

The script downloads the raw data from the UCI ML Repository if the Excel file does not exist:

```python
ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
```

If automatic download fails (network unavailable), manually place `Online Retail.xlsx`
in `data/raw/` before running.

**Dataset fingerprint (for verification):**

```json
{
  "sha256": "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d",
  "rows": 541909,
  "columns": 8
}
```

Source: `docs/reproducibility/dataset_fingerprint.json` (recorded by Member 1).

---

## 4. How to run

### 4.1 Run as a Python script (recommended for clean execution)

```powershell
# From the repository root:
.venv\Scripts\python.exe members\member-2\notebooks\02_cleaning_and_rfm.py
```

Expected console output includes:
- `LOADING RAW DATA`: 541,909 rows loaded
- `STEP 1 — Remove exact duplicate rows`: 5,268 removed; 536,641 remain
- `STEP 2 — Remove rows with missing CustomerID`: 135,037 removed; 401,604 remain
- `STEP 3 — Remove cancellations`: 8,872 removed; 392,732 remain
- `STEP 4 — Remove rows where UnitPrice <= 0`: 40 removed; 392,692 remain
- `STEP 5 — Remove non-product codes`: 1,248 removed; 391,444 remain
- `STEP 6 — Date cutoff`: 159,638 removed; 231,806 remain
- `STEP 7 — Build RFM table`: 3,362 customers; 0 with Monetary <= 0
- `STEP 8 — Log-transform & Scale`: scaled means ≈ 0; population std ≈ 1
- Charts saved to `members/member-2/figures/`

### 4.2 Run in Jupyter

Open `members/member-2/notebooks/02_cleaning_and_rfm.ipynb` in Jupyter and run all cells
in order. Ensure the virtual environment kernel is selected.

> [!WARNING]
> Running the notebook or script will **regenerate all output files and figures**:
> - `data/interim/cleaned_transactions.csv`
> - `data/processed/rfm_table.csv`
> - All four figure PNGs in `members/member-2/figures/`
>
> If Member 3's results depend on the current RFM table, re-running changes the
> SHA-256 hash recorded by Member 3's `run_manifest.json`. Only re-run if you
> intend to regenerate the full pipeline.

---

## 5. Validation checks (safe to run without modifying outputs)

The following Python commands can be run to verify existing outputs without re-running the pipeline.

### 5.1 Verify RFM table schema and key properties

```python
import pandas as pd
import numpy as np

rfm = pd.read_csv("data/processed/rfm_table.csv")

assert rfm.shape == (3362, 9), f"Expected (3362, 9), got {rfm.shape}"
assert rfm["CustomerID"].nunique() == 3362, "Duplicate CustomerIDs found"
assert rfm.isnull().sum().sum() == 0, "Missing values found"
assert np.isinf(rfm.select_dtypes(include=float).values).sum() == 0, "Infinite values found"
assert (rfm["Monetary"] > 0).all(), "Non-positive Monetary found"
assert (rfm["Frequency"] >= 1).all(), "Zero-frequency customer found"

print("RFM table: all checks passed")
print(f"  Shape: {rfm.shape}")
print(f"  One-order customers: {(rfm['Frequency'] == 1).sum()} ({(rfm['Frequency'] == 1).mean()*100:.2f}%)")
print(f"  Recency range: {rfm['Recency'].min()} – {rfm['Recency'].max()} days")
print(f"  Monetary range: £{rfm['Monetary'].min():.2f} – £{rfm['Monetary'].max():,.2f}")
```

**Expected output:**

```
RFM table: all checks passed
  Shape: (3362, 9)
  One-order customers: 1366 (40.63%)
  Recency range: 0 – 281 days
  Monetary range: £2.90 – £177,729.62
```

### 5.2 Verify transformation algebra

```python
# Verify that Frequency_scaled and Monetary_scaled are derived from log-transformed values
from sklearn.preprocessing import StandardScaler

rfm = pd.read_csv("data/processed/rfm_table.csv")
scaler = StandardScaler()
recomputed = scaler.fit_transform(
    rfm[["Recency", "Log_Frequency", "Log_Monetary"]]
)

tol = 1e-7
for i, col in enumerate(["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]):
    max_err = abs(recomputed[:, i] - rfm[col].values).max()
    assert max_err < tol, f"{col}: max error {max_err:.2e} exceeds tolerance {tol}"
    print(f"  {col}: max discrepancy = {max_err:.2e} (within tolerance {tol})")

print("Transformation algebra check: passed")
```

### 5.3 Verify cleaned transactions

```python
df = pd.read_csv("data/interim/cleaned_transactions.csv")

assert len(df) == 231806, f"Expected 231,806 rows, got {len(df)}"
assert df["CustomerID"].nunique() == 3362, "Customer count mismatch"
assert df.isnull().sum().sum() == 0, "Missing values in cleaned transactions"

# Date range
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
snapshot = pd.Timestamp("2011-09-09")
assert (df["InvoiceDate"] < snapshot).all(), "Rows on or after 2011-09-09 found"

print("Cleaned transactions: all checks passed")
print(f"  Rows: {len(df):,}")
print(f"  Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
```

**Expected output:**

```
Cleaned transactions: all checks passed
  Rows: 231,806
  Date range: 2010-12-01 08:26:00 to 2011-09-08 19:58:00
```

---

## 6. Outputs and their locations

| Output | Location | Rows / Size | When generated |
| --- | --- | --- | --- |
| `cleaned_transactions.csv` | `data/interim/` | 231,806 rows | Step 6 (after date cutoff) |
| `rfm_table.csv` | `data/processed/` | 3,362 rows, 9 columns | Step 8 (after scaling) |
| `cleaning_funnel.png` | `members/member-2/figures/` | — | End of script |
| `rfm_distributions_raw.png` | `members/member-2/figures/` | — | End of script |
| `rfm_distributions_transformed.png` | `members/member-2/figures/` | — | End of script |
| `rfm_correlation_heatmap.png` | `members/member-2/figures/` | — | End of script |

---

## 7. Important warnings

> [!WARNING]
> **Do not regenerate outputs if Member 3's results are already committed.** Member 3's
> `run_manifest.json` records SHA-256 hashes of `rfm_table.csv` and
> `cleaned_transactions.csv`. Regenerating these outputs changes the hashes and breaks
> Member 3's audit trail. Consult Member 3 before re-running.

> [!WARNING]
> **InvoiceNo dtype change.** The raw data has string InvoiceNo (e.g., `"C536379"`). After
> removing C-prefix rows and exporting to CSV, pandas infers `int64` on re-read. Code that
> expects string InvoiceNo in the cleaned file will need updating.

> [!NOTE]
> **Figure backend.** The script uses `matplotlib.use("Agg")` for non-interactive rendering.
> This is correct for script execution. When running in Jupyter, this line can cause
> warnings but does not prevent figure generation.

---

## 8. Source code inspection (no execution required)

The following logic can be verified by reading the source code directly without running it:

| Claim | Location in source | Verification method |
| --- | --- | --- |
| Duplicate removal uses `drop_duplicates()` | Lines 100–108 | Read |
| CustomerID missing check uses `isna().sum()` | Lines 113–125 | Read |
| Cancellation criterion is OR | Lines 133–164 | Read: `mask_cancel = mask_c_prefix | mask_neg_qty` |
| Non-product codes: POST, DOT, C2, S | Lines 185–203 | Read: `NON_PRODUCT_CODES = {"POST", "DOT", "C2", "S"}` |
| Snapshot date is 2011-09-09 | Line 209 | Read: `SNAPSHOT_DATE = pd.Timestamp("2011-09-09")` |
| Recency is `(SNAPSHOT - last_date).days` | Lines 244–248 | Read |
| Frequency is `nunique(InvoiceNo)` | Lines 251–256 | Read |
| Monetary is `sum(Quantity * UnitPrice)` | Lines 241, 259–263 | Read |
| Log1p applied to Frequency and Monetary | Lines 293–294 | Read |
| StandardScaler fitted on Recency, Log_F, Log_M | Lines 296–299 | Read |

---

## 9. Differences from Member 3's pipeline

Member 3 uses `run_notebook.py` and `verify_outputs.py` for fresh-kernel execution and
independent verification. Member 2's pipeline does not use these helpers because it was
developed before the shared tooling was established.

Practical difference: re-running Member 2's notebook or script in Jupyter does not use
a fresh kernel; stale variable state from earlier cells can persist. Running the `.py`
script directly avoids this.

---

## 10. Contact point for issues

If the dataset download fails, the file is not present, or row counts differ from those
documented here, check:

1. That the correct Python environment is active.
2. That the `PROJECT_ROOT` path is updated to match your local repository.
3. That `Online Retail.xlsx` is in `data/raw/` (either downloaded or manually placed).
4. That the `sha256` of your Excel file matches the fingerprint recorded by Member 1:
   `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`.
