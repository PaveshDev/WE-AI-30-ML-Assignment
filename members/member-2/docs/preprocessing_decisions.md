# Member 2 Preprocessing Decisions

**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Author role:** Member 2 — Data Quality, Cleaning, Feature Engineering, Transformations, Scaling

This document gives a structured record of each important preprocessing decision. Each entry
states the problem or observation, options considered, the selected implementation, the actual
evidence, why the choice is appropriate for this project, and known trade-offs.

Decisions marked **[Code-verified]** are directly confirmed by the executed source code.  
Decisions marked **[Log-documented]** are explicitly recorded in `preprocessing_log.md`.  
Decisions marked **[Retrospective]** are discussed here but were not written up beforehand.

---

## Decision M2-01: Remove exact duplicate rows first

**Problem:** The raw dataset contains 5,268 rows that are exact copies of another row (EDA Q12).
If these rows are included in Monetary totals, some customers' spend will be falsely inflated.

**Options considered:**

| Option | Risk |
| --- | --- |
| Remove duplicates first (before any filtering) | Correct: prevents inflated totals from the start |
| Remove duplicates after CustomerID filtering | Some duplicated rows might already be filtered; remaining copy might not be caught |
| Remove duplicates after all filters | Totals computed before deduplication would be wrong |

**Selected implementation:** `df.drop_duplicates()` as the first pipeline operation.  
**Code reference:** `02_cleaning_and_rfm.py` lines 100–108 [Code-verified].  
**Evidence:** 5,268 rows removed; 536,641 remain.  
**Why appropriate:** All downstream totals — Monetary in particular — must be computed on
non-inflated data. Removing duplicates first is a prerequisite for correctness.  
**Trade-off:** None identified. Exact-row deduplication is universally appropriate here.  
**Downstream effect:** Monetary values for affected customers are slightly lower than if
duplicates had been retained. The effect is correct.

---

## Decision M2-02: Remove rows with missing CustomerID

**Problem:** 135,037 rows (approximately 25% of raw data) have no CustomerID (EDA Q1).
Without a CustomerID, a transaction cannot be linked to a specific customer.

**Options considered:**

| Option | Risk |
| --- | --- |
| Impute CustomerID | No basis for imputation; unknown identity cannot be inferred |
| Treat as a separate anonymous segment | Cannot aggregate; no consistent identifier per customer |
| Remove entirely | Loses potential signal from unidentified customers |

**Selected implementation:** `df.dropna(subset=["CustomerID"])` then `astype(int)`.  
**Code reference:** `02_cleaning_and_rfm.py` lines 113–125 [Code-verified].  
**Evidence:** 135,037 rows removed; 401,604 remain.  
**Why appropriate:** Customer-level RFM aggregation requires a stable unique key per customer.
Without CustomerID, this is impossible.  
**Trade-off:** Segments describe only identified customers. Guest/till-sale behaviour is excluded.
This is an acknowledged population-scope limitation, documented in `preprocessing_log.md`.  
**Downstream effect:** All subsequent work (including Member 3's clusters and Member 4's
validation) applies only to identified customers.

---

## Decision M2-03: Cancellation removal using OR, not AND

**Problem:** Cancellation transactions carry negative spend and should be excluded so that
RFM Monetary is not reduced by returns (EDA Q7, Q8).

**Options considered:**

| Option | Rows caught | Risk of missing cancellations |
| --- | --- | --- |
| C-prefix only (`mask_c_prefix`) | Catches standard cancellation invoices | Misses negative-quantity rows without C-prefix |
| Qty < 0 only (`mask_neg_qty`) | Catches negative quantities | Misses C-prefix rows with non-negative quantity |
| C-prefix AND Qty < 0 (`mask_c_prefix & mask_neg_qty`) | Intersection | Misses rows that are C-prefix but non-negative, or negative-qty but no C-prefix |
| C-prefix OR Qty < 0 (`mask_c_prefix | mask_neg_qty`) | Union | Most defensive; catches all three cases |

**Selected implementation:** `mask_cancel = mask_c_prefix | mask_neg_qty` [Code-verified].  
**Evidence:** 8,872 rows removed; 392,732 remain. The "Negative-only (no C)" count confirmed ≈0
additional rows were caught by the Qty < 0 criterion alone after C-prefix removal.  
**Why appropriate:** OR is the more defensively correct form and matches standard retail
data-cleaning practice. The preprocessing log description used "AND" informally; the code
is the authoritative implementation.  
**Noted discrepancy:** Member 3's strategy document records: "Member 2's log says AND; its
source/notebook use OR. Their executed evidence shows complete overlap in the retained
identifiable cohort." This is documented, not silently hidden.  
**Trade-off:** None identified. OR is strictly more inclusive and correct.  
**Downstream effect:** All negative-spend rows are excluded from RFM Monetary.
The post-cleaning Monetary > 0 check confirmed 0 customers had non-positive Monetary.

---

## Decision M2-04: Remove rows where UnitPrice <= 0

**Problem:** A small number of rows have a zero or negative UnitPrice (EDA Q8). These produce
zero or negative line spend values that represent data-entry errors or internal adjustments.

**Options considered:**

| Option | Risk |
| --- | --- |
| Remove UnitPrice <= 0 | Loses 40 rows; correct for product sales |
| Keep them | Introduces zero/negative spend into Monetary calculation |

**Selected implementation:** `df[df["UnitPrice"] > 0]` [Code-verified].  
**Evidence:** 40 rows removed; 392,692 remain.  
**Why appropriate:** Genuine customer purchases always have a positive price. Zero or negative
prices are accounting artefacts, not real sales.  
**Trade-off:** 40 rows lost. This is negligible relative to the 401,604 pre-Step-4 count.  
**Downstream effect:** Monetary is free from zero or negative price contributions.

---

## Decision M2-05: Remove selected non-product stock codes (POST, DOT, C2, S)

**Problem:** Certain rows represent service or overhead charges rather than retail products
(EDA Q9). These inflate Monetary with non-product charges.

**Options considered:**

| Option | Risk |
| --- | --- |
| Remove POST, DOT, C2, S only | Minimal exclusion; D and M require separate assessment |
| Remove all non-numeric stock codes | Over-inclusive; removes valid M rows |
| Remove nothing | Service charges inflate product spend |

**Selected implementation:** `NON_PRODUCT_CODES = {"POST", "DOT", "C2", "S"}` [Code-verified].

**Code treatment of D and M:**

| Code | Meaning | Treatment | Outcome |
| --- | --- | --- | --- |
| POST | Postage charge | Removed in Step 5 | Gone |
| DOT | Dot/overhead | Removed in Step 5 | Gone |
| C2 | Carriage charge | Removed in Step 5 | Gone |
| S | Sample | Removed in Step 5 | Gone |
| D | Discount | Not explicitly removed — removed by earlier steps | 0 rows survive |
| M | Manual/misc adjustment | Not explicitly removed | 171 rows survive |

**Evidence:** 1,248 rows removed; 391,444 remain.  
**Why appropriate:** The four removed codes are unambiguously non-product service charges.
D rows were removed by earlier steps (cancellation/negative-quantity or missing CustomerID).
M rows represent miscellaneous manual transactions that passed all quality criteria.  
**Trade-off:** 171 M-code rows are retained. Their effect on Monetary for affected customers
is minor but unquantified at this level.  
**Downstream effect:** Product spend is cleaner; postage and overhead charges are excluded.

---

## Decision M2-06: Apply date cutoff before RFM computation

**Problem:** The project design requires a temporal split:
- Historical window (before 2011-09-09): used for RFM construction and clustering.
- Future validation window (2011-09-09 onward): reserved for Member 4's evaluation.

**Options considered:**

| Option | Risk |
| --- | --- |
| Apply date cutoff before RFM (Step 6 before Step 7) | Correct: Recency and totals use history only |
| Apply date cutoff after RFM | Recency would be computed using future purchases; Monetary would include future spend |
| No date cutoff | Future data contaminates historical RFM; Member 4 has no clean validation window |

**Selected implementation:** `SNAPSHOT_DATE = pd.Timestamp("2011-09-09")`;
keep rows where `InvoiceDate < SNAPSHOT_DATE` [Code-verified].  
**Evidence:** 159,638 rows removed; 231,806 remain. Historical date range verified:
2010-12-01 08:26:00 through 2011-09-08 19:58:00.  
**Why appropriate:** Applying the cutoff first guarantees:
1. Recency is correctly measured relative to the 2011-09-09 snapshot.
2. The future validation window is genuinely held out for Member 4.
3. No temporal leakage from future transactions into historical modelling.

**Trade-off:** 976 customers who purchased only in the future window are excluded from
the historical RFM table. This reduces total customers from ~4,338 (EDA full-data) to
3,362. The difference is expected and documented.  
**Downstream effect:** Member 3 fits on 3,362 historical customers only.
Member 4 must prepare the future window separately without using any RFM values derived here.

---

## Decision M2-07: Log-transform Frequency and Monetary (not Recency)

**Problem:** Frequency and Monetary have extreme right skews (EDA Q2, Q11). Without
transformation, a few wholesale customers would dominate Euclidean distance in clustering.

**Options considered:**

| Option | Risk |
| --- | --- |
| No transformation | Extreme values dominate distances |
| Square root | Compresses tail less effectively than log; still bounded for positive inputs |
| log(x) — plain logarithm | Undefined for x = 0; Frequency ≥ 1 so technically safe, but less defensive |
| log1p(x) = log(1 + x) | Safe for x ≥ 0; compresses tail strongly; preserves ordering |
| log-transform Recency too | Recency has a bounded range (0–281); less severe skew; not warranted |

**Selected implementation:** `np.log1p(Frequency)` and `np.log1p(Monetary)` [Code-verified].  
**Evidence:**
- log1p(177729.62) ≈ 12.09 vs log1p(555.015) ≈ 6.32 → ratio ~1.9:1 (was ~320:1).
- Frequency_scaled and Monetary_scaled distributions visible in `rfm_distributions_transformed.png`.

**Why appropriate:** log1p is safe for non-negative integer and float inputs and compresses
the right tail substantially.  
**Important caveat:** log1p does not guarantee normality. The one-order customer mass
(1,366 at Frequency = 1) creates a discrete spike in the log-transformed distribution.  
**Trade-off:** Some information about the exact magnitude of extreme values is lost.
Large wholesale customers still appear as outliers in log space but are far less extreme.  
**Downstream effect:** Frequency_scaled and Monetary_scaled present a much more tractable
input for K-Means distance calculations. Member 3 verified transformation relationships to 1e-7.

---

## Decision M2-08: Use StandardScaler for all three features

**Problem:** After log transformation, the three prepared features still have different
absolute scales. StandardScaler brings all three to mean ≈ 0, population std ≈ 1.

**Options considered:**

| Scaler | Why not chosen |
| --- | --- |
| No scaling | Recency (days) vs Monetary (log pounds) still on different scales |
| MinMaxScaler | Maps to [0,1] using max−min; extreme Monetary value makes denominator very large, compressing most customers near zero |
| RobustScaler | Uses median and IQR; not evaluated explicitly but would be a valid alternative |
| StandardScaler | Appropriate when distributions are approximately symmetric (post-log); recommended by EDA Q2 and Q11 context |

**Selected implementation:** `StandardScaler().fit_transform(rfm[["Recency", "Log_Frequency", "Log_Monetary"]])` [Code-verified].  
**Evidence:** Scaled means ≈ 0; population std ≈ 1 (verified above).  
**Why appropriate:** Post-log Frequency and Monetary distributions are approximately
symmetric. StandardScaler's mean-centering removes unit bias for Euclidean distance.  
**Trade-off:** StandardScaler is sensitive to remaining outliers in Recency (which was not
log-transformed). The outlier influence is accepted because the log transformation has
already addressed Frequency and Monetary extremes.  
**Downstream effect:** Member 3 uses Recency_scaled, Frequency_scaled, Monetary_scaled
as the three-dimensional feature matrix X for all clustering.

---

## Decision M2-09: Do not cap outliers

**Problem:** Extreme values remain in Monetary even after log transformation (the maximum
customer in log space is approximately 12.09, while the median is ≈ 6.32).

**Options considered:**

| Option | Risk |
| --- | --- |
| IQR-based capping (e.g., clip at Q3 + 1.5 × IQR) | Arbitrary threshold; destroys real information about wholesale accounts |
| Percentile capping (e.g., 99th percentile) | Equally arbitrary; no business justification |
| Winsorization | Same concern as percentile capping |
| No capping | Wholesale accounts remain in data; log compression is sufficient |

**Selected implementation:** No outlier capping [Log-documented].  
**Evidence:** Member 3's leading K-Means result (k=2, silhouette 0.4114) was not distorted
by extreme customers. See `members/member-3/docs/model_comparison.md`.  
**Why appropriate:** The highest-spending customers are likely genuine wholesale accounts
(EDA Q2, Q10). Log transformation reduces their relative distance from typical customers
by approximately 160-fold. No business threshold for "too extreme" was identified.  
**Trade-off:** Wholesale customers will influence centroids. This is acknowledged.  
**Downstream effect:** Member 3 should inspect cluster profiles for outlier-driven patterns.
If needed, capping can be revisited without breaking the existing pipeline outputs.

---

## Decision M2-10: Retain raw and log columns alongside scaled columns

**Problem:** If only scaled columns were exported, Member 3 and Member 4 would not be
able to interpret cluster centroids or profiles in interpretable human units.

**Options considered:**

| Option | Risk |
| --- | --- |
| Export only scaled columns | Cluster profiles would only show standard-deviation values |
| Export raw + log + scaled columns | 9 columns total; all interpretable contexts available |

**Selected implementation:** Export CustomerID, Recency, Frequency, Monetary, Log_Frequency,
Log_Monetary, Recency_scaled, Frequency_scaled, Monetary_scaled [Code-verified].  
**Evidence:** `rfm_table.csv` has 9 columns; Member 3 confirms raw RFM is available for
profiling (cluster_profiles.csv reports raw medians and means).  
**Why appropriate:** Member 3 explicitly uses raw Recency, Frequency and Monetary to
describe clusters in interpretable units for Member 4.  
**Trade-off:** Slightly wider CSV, but the added columns are essential for profiling.  
**Downstream effect:** Member 3 can describe segments as "median 125 days since last
purchase, 1 order, £304 spend" rather than only in standard-deviation units.

---

## Decision M2-11: Defensive Monetary > 0 check before log transformation

**Problem:** `log1p` requires non-negative inputs. Member 1's EDA predicted approximately
50 customers might have zero or negative net spend after cleaning.

**Options considered:**

| Option | Risk |
| --- | --- |
| Trust earlier cleaning steps; no check | Risk of undefined log values if edge case exists |
| Check and filter Monetary <= 0 | Removes problematic customers defensively |

**Selected implementation:** `rfm = rfm[rfm["Monetary"] > 0].copy()` [Code-verified].  
**Evidence:** 0 customers had Monetary <= 0. The EDA prediction of ~50 such customers was
based on an analysis that included cancellation rows; after Step 3 removes all cancellations,
every remaining customer has positive net spend.  
**Why appropriate:** Defensive data-quality guards are good practice even when the primary
cleaning steps should already prevent the edge case.  
**Trade-off:** None — 0 rows were removed.  
**Downstream effect:** log1p is applied to values that are guaranteed > 0.

---

## Open group decisions

The following issues were identified but not resolved within Member 2's scope:

| Issue | Status |
| --- | --- |
| Does postage / M-code rows count towards Monetary? | Noted in EDA Q9; not resolved; M rows retained |
| AND vs OR wording in the preprocessing log | Acknowledged; code uses OR (correct); log uses AND (informal description) |
| Validate exact future validation endpoint | 90 days from 2011-09-09 ≈ 2011-12-08; Member 4 to confirm |
| Qualification of Monetary as gross spend, not net revenue | Acknowledged in strategy and handoff; Member 4 to apply in final interpretation |

---

## References

| Source | Location |
| --- | --- |
| Executed Python script | `members/member-2/notebooks/02_cleaning_and_rfm.py` |
| Preprocessing log | `members/member-2/docs/preprocessing_log.md` |
| EDA insight log | `members/member-1/docs/eda_insight_log.csv` |
| Member 3 strategy (AND/OR qualification) | `members/member-3/docs/modelling_strategy.md` |
| Member 3 model comparison | `members/member-3/docs/model_comparison.md` |
