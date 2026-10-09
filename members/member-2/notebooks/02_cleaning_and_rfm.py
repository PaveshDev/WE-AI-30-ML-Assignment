# %% [markdown]
"""
# Member 2 — Data Cleaning & RFM Feature Engineering
===================================================
UCI Online Retail → cleaned transactions + one-row-per-customer RFM table.

Pipeline order (strict):
  1. Remove exact duplicate rows
  2. Remove rows with no CustomerID
  3. Remove cancellations (InvoiceNo starts with 'C' OR Quantity < 0)
  4. Remove rows where UnitPrice <= 0
  5. Remove non-product stock codes: POST, DOT, C2, S
  6. Date cutoff: keep rows before 9 Sep 2011
  7. Build RFM table (Recency, Frequency, Monetary)
  8. Log-transform Frequency & Monetary, then StandardScaler all three

Design decisions:
  - Validate Monetary > 0 before log transform (defensive check; after
    cancellation and negative-quantity removal, 0 customers required removal)
  - D and M codes are not excluded by the non-product filter. In practice,
    no D rows survive earlier cleaning steps; only valid M rows remain.
  - Use StandardScaler (post-log distributions are roughly normal)
  - No capping (log already compresses extreme tails sufficiently)

Notes:
  - EDA statistics (Q1-Q12) were computed on the full dataset without the
    date cutoff. Post-cutoff RFM values differ (3,362 vs 4,338 customers,
    40.6% vs 30% one-time buyers, Spearman F-M ≈ 0.78 vs 0.81).
  - Description nulls (1,454 in raw) are not removed by a dedicated rule;
    they are eliminated as a side effect of CustomerID/data-quality filtering.
  - After removing C-prefix invoices, all remaining InvoiceNo values are
    numeric. When exported to CSV and re-read, pandas infers int64 dtype.
"""

# %% Setup & Imports
import os
import sys
import urllib.request
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"c:\Users\ASUS\Music\WE-AI-30-ML-Assignment"
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
INTERIM_DIR  = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DIR= os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR      = os.path.join(PROJECT_ROOT, "members", "member-2", "figures")
DOCS_DIR     = os.path.join(PROJECT_ROOT, "members", "member-2", "docs")

XLSX_PATH    = os.path.join(RAW_DIR, "Online Retail.xlsx")
ZIP_URL      = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ZIP_PATH     = os.path.join(RAW_DIR, "online_retail.zip")

for d in [INTERIM_DIR, PROCESSED_DIR, FIG_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

# %% Step 0: Download dataset if needed
if not os.path.exists(XLSX_PATH):
    print("=" * 60)
    print("STEP 0  — Downloading UCI Online Retail dataset")
    print("=" * 60)
    print(f"  URL : {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
    os.remove(ZIP_PATH)
    print(f"  Saved to: {XLSX_PATH}")
    print()

# %% Load raw data
print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)
df = pd.read_excel(XLSX_PATH, engine="openpyxl")
n_raw = len(df)
print(f"  Rows loaded : {n_raw:,}")
print(f"  Columns     : {list(df.columns)}")
print(f"  Dtypes      :")
for col in df.columns:
    print(f"    {col:20s} {str(df[col].dtype):10s}  nulls={df[col].isna().sum():,}")
print()

# Track row counts for the cleaning funnel
funnel = [("Raw", n_raw)]

# %% Step 1: Remove exact duplicates
print("=" * 60)
print("STEP 1  — Remove exact duplicate rows")
print("=" * 60)
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
removed = n_before - n_after
print(f"  Before : {n_before:,}")
print(f"  Removed: {removed:,}")
print(f"  After  : {n_after:,}")
print()
funnel.append(("1. Duplicates removed", n_after))

# %% Step 2: Remove rows with missing CustomerID
print("=" * 60)
print("STEP 2  — Remove rows with missing CustomerID")
print("=" * 60)
n_before = len(df)
n_missing = df["CustomerID"].isna().sum()
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
n_after = len(df)
removed = n_before - n_after
print(f"  Before          : {n_before:,}")
print(f"  Missing CustID  : {n_missing:,}")
print(f"  Removed         : {removed:,}")
print(f"  After           : {n_after:,}")
print()
funnel.append(("2. No CustomerID", n_after))

# %% Step 3: Remove cancellations
print("=" * 60)
print("STEP 3  — Remove cancellations")
print("=" * 60)
n_before = len(df)

# Criterion A: InvoiceNo starts with 'C'
mask_c_prefix = df["InvoiceNo"].astype(str).str.startswith("C")
n_c_prefix = mask_c_prefix.sum()

# Criterion B: Quantity < 0
mask_neg_qty = df["Quantity"] < 0
n_neg_qty = mask_neg_qty.sum()

# Overlap
mask_both = mask_c_prefix & mask_neg_qty
n_both = mask_both.sum()

# Only negative qty, no C prefix
mask_neg_only = mask_neg_qty & ~mask_c_prefix
n_neg_only = mask_neg_only.sum()

# Combined: either condition
mask_cancel = mask_c_prefix | mask_neg_qty
n_cancel = mask_cancel.sum()

df = df[~mask_cancel]
n_after = len(df)

print(f"  Before              : {n_before:,}")
print(f"  C-prefix rows       : {n_c_prefix:,}")
print(f"  Negative-qty rows   : {n_neg_qty:,}")
print(f"  Both (overlap)      : {n_both:,}")
print(f"  Negative-only (no C): {n_neg_only:,}")
print(f"  Total removed       : {n_cancel:,}")
print(f"  After               : {n_after:,}")
print()
funnel.append(("3. Cancellations", n_after))

# %% Step 4: Remove rows where UnitPrice <= 0
print("=" * 60)
print("STEP 4  — Remove rows where UnitPrice <= 0")
print("=" * 60)
n_before = len(df)
mask_zero_price = df["UnitPrice"] <= 0
n_zero = mask_zero_price.sum()
df = df[~mask_zero_price]
n_after = len(df)
print(f"  Before  : {n_before:,}")
print(f"  Price<=0: {n_zero:,}")
print(f"  After   : {n_after:,}")
print()
funnel.append(("4. Price <= 0", n_after))

# %% Step 5: Remove non-product stock codes
print("=" * 60)
print("STEP 5  — Remove non-product codes (POST, DOT, C2, S)")
print("=" * 60)
NON_PRODUCT_CODES = {"POST", "DOT", "C2", "S"}
KEEP_CODES = {"D", "M"}  # not excluded here; D rows removed by earlier steps, only M survives

n_before = len(df)
mask_non_product = df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
n_non_product = mask_non_product.sum()

# Report what we're keeping too
mask_keep = df["StockCode"].astype(str).str.upper().isin(KEEP_CODES)
n_keep = mask_keep.sum()

df = df[~mask_non_product]
n_after = len(df)
print(f"  Before              : {n_before:,}")
print(f"  Non-product removed : {n_non_product:,}  (codes: {NON_PRODUCT_CODES})")
print(f"  D/M rows kept       : {n_keep:,}  (discounts/adjustments)")
print(f"  After               : {n_after:,}")
print()
funnel.append(("5. Non-product codes", n_after))

# %% Step 6: Date cutoff — keep only before 9 Sep 2011
print("=" * 60)
print("STEP 6  — Date cutoff: keep rows before 9 Sep 2011")
print("=" * 60)
SNAPSHOT_DATE = pd.Timestamp("2011-09-09")

n_before = len(df)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
mask_after = df["InvoiceDate"] >= SNAPSHOT_DATE
n_after_cut = mask_after.sum()
df = df[~mask_after]
n_after = len(df)

print(f"  Snapshot date     : {SNAPSHOT_DATE.date()}")
print(f"  Before            : {n_before:,}")
print(f"  Rows on/after cut : {n_after_cut:,}")
print(f"  After             : {n_after:,}")
print(f"  Date range kept   : {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print()
funnel.append(("6. Date cutoff", n_after))

# %% Save cleaned transactions
cleaned_path = os.path.join(INTERIM_DIR, "cleaned_transactions.csv")
df.to_csv(cleaned_path, index=False)
print(f"  Saved cleaned transactions: {cleaned_path}")
print(f"  Final transaction rows    : {len(df):,}")
print(f"  Unique customers          : {df['CustomerID'].nunique():,}")
print(f"  Unique invoices           : {df['InvoiceNo'].nunique():,}")
print()

# %% Step 7: Build RFM table (one row per customer)
print("=" * 60)
print("STEP 7  — Build RFM table (one row per customer)")
print("=" * 60)

# Compute line-level spend
df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

# Recency: days since last purchase relative to snapshot date
recency = (
    df.groupby("CustomerID")["InvoiceDate"]
    .max()
    .apply(lambda x: (SNAPSHOT_DATE - x).days)
    .rename("Recency")
)

# Frequency: number of unique invoices per customer
frequency = (
    df.groupby("CustomerID")["InvoiceNo"]
    .nunique()
    .rename("Frequency")
)

# Monetary: total spend per customer
monetary = (
    df.groupby("CustomerID")["TotalSpend"]
    .sum()
    .rename("Monetary")
)

rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
n_customers_before = len(rfm)

print(f"  Customers before filtering : {n_customers_before:,}")
print(f"  Recency  range : {rfm['Recency'].min()} -- {rfm['Recency'].max()} days")
print(f"  Frequency range: {rfm['Frequency'].min()} -- {rfm['Frequency'].max()}")
print(f"  Monetary range : GBP {rfm['Monetary'].min():,.2f} -- GBP {rfm['Monetary'].max():,.2f}")
print()

# Defensive check: ensure Monetary > 0 for log transform (cancellations/returns removed in Step 3)
n_zero_spend = (rfm["Monetary"] <= 0).sum()
print(f"  Customers with Monetary <= 0: {n_zero_spend}")
rfm = rfm[rfm["Monetary"] > 0].copy()
n_customers_after = len(rfm)
print(f"  Removed: {n_zero_spend}")
print(f"  Customers after filtering  : {n_customers_after:,}")
print()

# Summary stats
print("  RFM summary (raw):")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().to_string())
print()

# %% Step 8: Log-transform & Scale
print("=" * 60)
print("STEP 8  — Log-transform Frequency & Monetary, then StandardScaler")
print("=" * 60)

rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])
rfm["Log_Monetary"]  = np.log1p(rfm["Monetary"])

scaler = StandardScaler()
rfm[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]] = scaler.fit_transform(
    rfm[["Recency", "Log_Frequency", "Log_Monetary"]]
)

print("  Scaled feature stats:")
for col in ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]:
    print(f"    {col:20s}  mean={rfm[col].mean():.6f}  std={rfm[col].std():.6f}"
          f"  min={rfm[col].min():.3f}  max={rfm[col].max():.3f}")
print()

# Save
rfm_path = os.path.join(PROCESSED_DIR, "rfm_table.csv")
rfm.to_csv(rfm_path, index=False)
print(f"  Saved RFM table: {rfm_path}")
print(f"  Rows: {len(rfm):,}")
print(f"  Columns: {list(rfm.columns)}")
print()

# %% Plot Charts
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("viridis", 8)

# ── Chart 1: Cleaning funnel ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
labels = [f[0] for f in funnel]
values = [f[1] for f in funnel]
bar_colors = [COLORS[0]] + [COLORS[i+1] for i in range(len(funnel)-1)]
bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1], edgecolor="white")
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of rows")
ax.set_title("Data Cleaning Funnel - Rows Remaining After Each Step", fontsize=14, fontweight="bold")
ax.set_xlim(0, max(values) * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cleaning_funnel.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: cleaning_funnel.png")

# ── Chart 2: Raw RFM distributions ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, color, xlabel in zip(
    axes,
    ["Recency", "Frequency", "Monetary"],
    [COLORS[0], COLORS[3], COLORS[5]],
    ["Days since last purchase", "Number of orders", "Total spend (GBP)"]
):
    ax.hist(rfm[col], bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(f"{col} (raw)", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Customers")
    # Add median line
    med = rfm[col].median()
    ax.axvline(med, color="red", linestyle="--", linewidth=1.5, label=f"Median: {med:,.0f}")
    ax.legend(fontsize=9)
plt.suptitle("RFM Distributions - Before Transformation", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rfm_distributions_raw.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: rfm_distributions_raw.png")

# ── Chart 3: Transformed RFM distributions ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, color, label in zip(
    axes,
    ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"],
    [COLORS[0], COLORS[3], COLORS[5]],
    ["Recency (scaled)", "log(Frequency) (scaled)", "log(Monetary) (scaled)"]
):
    ax.hist(rfm[col], bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Standard deviations from mean")
    ax.set_ylabel("Customers")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Mean (0)")
    ax.legend(fontsize=9)
plt.suptitle("RFM Distributions - After Log Transform + StandardScaler", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rfm_distributions_transformed.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: rfm_distributions_transformed.png")

# ── Chart 4: Correlation heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
corr = rfm[["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
    vmin=-1, vmax=1, center=0, square=True, linewidths=1,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax,
    xticklabels=["Recency", "Frequency", "Monetary"],
    yticklabels=["Recency", "Frequency", "Monetary"]
)
ax.set_title("Feature Correlation - Scaled RFM", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "rfm_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Chart saved: rfm_correlation_heatmap.png")
