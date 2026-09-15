# Member 3 handoff to Member 4

## Inputs and scope

Input: `data/processed/rfm_table.csv`, 3,362 unique customers.
SHA-256: `6bf62e981a1e91770908de3b104b67b8e1c0b726926967b19be7a76d0eaf45e2`.
Exact fitting columns: `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled`;
the latter two are standardized log1p Frequency/Monetary. CustomerID is excluded.
No missing/infinite features, duplicate IDs or downstream input adjustments.
The historical transaction audit ends at **2011-09-08 19:58:00**.

Implemented the agreed rule-based RFM baseline, K-Means, Ward agglomerative and full
covariance GMM. ML group counts 2..8; primary seed 42. K-Means has 20 starts, GMM 10;
both have a 500-iteration cap. See modelling_strategy.md for exact parameters.

## Frozen outputs

**`data/processed/cluster_assignments.csv`** contains 3,362 rows:

| Column | Meaning |
| --- | --- |
| CustomerID | Unique join key; preserve original IDs |
| Recency, Frequency, Monetary | Historical raw RFM: days, distinct invoices, GBP retained positive spend |
| RFM_Baseline_Group | Neutral groups 0..3 from fixed score-total bands 3–6 / 7–9 / 10–12 / 13–15 |
| KMeans_k2 | Leading technical candidate, primary seed 42 |
| KMeans_k3 | Finer secondary candidate, primary seed 42 |
| Hierarchical_k2 | Ward coarse comparator |
| GMM_k2 | Full-covariance coarse comparator, primary seed 42 |
| GMM_k8_Diagnostic | AIC/BIC-selected within tested range; density diagnostic, not preferred segmentation |
| GMM_k2_MaxPosterior | Maximum membership probability for GMM k=2 |
| GMM_k8_Diagnostic_MaxPosterior | Maximum membership probability for diagnostic GMM k=8 |

Labels are neutral and local to a configuration: Cluster 0 is not necessarily the same
customers across columns. Posterior values are not repurchase probabilities.

## Technical comparison and stability

| method | k | silhouette | davies_bouldin | calinski_harabasz | smallest_cluster | largest_cluster | ari_mean | ari_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFM_Baseline | 4 | 0.2220 | 1.4174 | 2097.5180 | 719 | 935 | N/A | N/A |
| KMeans | 2 | 0.4114 | 0.9045 | 3136.7877 | 1395 | 1967 | 1.0000 | 1.0000 |
| KMeans | 3 | 0.3728 | 0.9158 | 3135.7482 | 866 | 1520 | 0.9989 | 0.9982 |
| Hierarchical | 2 | 0.3928 | 0.9468 | 2896.4257 | 1645 | 1717 | N/A | N/A |
| GMM | 2 | 0.3802 | 0.9724 | 2723.7079 | 1650 | 1712 | 1.0000 | 1.0000 |
| GMM | 8 | 0.1597 | 1.9549 | 1265.3859 | 95 | 772 | 0.7864 | 0.6517 |

K-Means k=2 leads the internal geometric metrics; its five-seed minimum ARI is
1.0000. k=3 K-Means has minimum ARI 0.9982, offering a stable
finer candidate. Ward k=2 is deterministic and balanced. GMM k=2 is seed-stable but
18.95% have posterior <0.70. GMM k=8 has better AIC/BIC but
minimum ARI 0.6517, poor geometric separation, and covariance-floor
components associated with discrete Frequency values. Do not pick it from BIC alone.

Stability used all ten pairwise ARIs across seeds 42, 7, 21, 99, 123 for every searched
stochastic configuration. It is initialization repeatability, not temporal validation.

## Your validation work

1. Keep assignments frozen; **2011-09-09 00:00:00 onward is the reserved future period**.
   Do not recompute historical RFM, rescale or refit using future transactions.
2. The interim cleaned file contains history only. Prepare future evaluation data from
   the appropriate source under documented group-agreed transaction rules; Member 3
   has not prepared or used the future period. Agree whether the endpoint is 90 days,
   three calendar months, or through observed 9 December coverage before evaluation.
3. Aggregate future outcomes by CustomerID and left-join them to these assignments with
   one-to-one validation. Keep all historical customers, including non-returners; fill
   absent purchases with zero only after confirming outcome coverage and cleaning rules.
4. Compare future repeat-purchase proportions, order counts and spend across groups,
   with group denominators and uncertainty. Repeat purchase is an evaluation measure,
   not a supervised model fitted by Member 3. Treat future-only customers separately.
5. Assess whether k=3 adds useful differentiation over k=2 and whether baseline rules
   remain competitive in practical interpretation. Examine ambiguous GMM members.
6. Apply final business names and recommendations after validation. Note Christmas
   seasonality, missing-ID exclusions, unequal customer exposure, correlated F/M and
   positive-spend versus net-revenue accounting. Agreement across seeds is not evidence
   that a label such as 'lost' is true.

## Evidence files and reproducibility

- `members/member-3/notebooks/03_modelling_and_comparison.ipynb`: executed narrative.
- `members/member-3/results/experiments.csv`: all 22 primary method/configuration rows.
- `candidate_comparison.csv`, `cluster_profiles.csv`: candidate evidence and profiles
  for every searched configuration; filenames here are under `members/member-3/results/`.
- `stability_runs.csv`, `stability_pairs.csv`, `stability_summary.csv`: full seed evidence.
- `gmm_component_diagnostics.csv`, `baseline_score_ranges.csv`: method diagnostics.
- `input_audit.json`, `input_summary.csv`, `run_manifest.json`: data/parameter/version fingerprints.
- `model_comparison.md`: full discussion; `modelling_strategy.md`: rubric justification.
- `modelling_decisions.md`: draft for the group lead, since the shared log is empty.

Run instructions are in `members/member-3/README.md`. No Member 1/2 input was overwritten;
no final business winner was selected. The Assignment Descriptor and Initial Submission
must still be cross-checked against the originals by the group, as they are absent here.
