"""Write Member 3 evidence documents from the executed notebook's real results."""
import json

import pandas as pd

from modelling_helpers import markdown_table


def write_reports(docs, audit, input_summary, experiments, comparison, profiles,
                  stability, seed_runs, gmm_diagnostics, score_ranges, variance,
                  hashes, assignments):
    def table(frame, columns=None, decimals=4):
        return markdown_table(frame if columns is None else frame[columns], decimals)

    def result(method, k):
        return experiments[(experiments.method == method) & (experiments.k == k)].iloc[0]

    km2, km3 = result("KMeans", 2), result("KMeans", 3)
    ward2, ward3 = result("Hierarchical", 2), result("Hierarchical", 3)
    gm2, gm8 = result("GMM", 2), result("GMM", 8)
    base = result("RFM_Baseline", 4)
    metrics = ["method", "k", "silhouette", "davies_bouldin", "calinski_harabasz",
               "smallest_cluster", "largest_cluster", "ari_mean", "ari_min"]
    gmm = experiments[experiments.method == "GMM"]
    km = experiments[experiments.method == "KMeans"]
    ward = experiments[experiments.method == "Hierarchical"]
    spread = seed_runs.groupby(["method", "k"], as_index=False).agg(
        silhouette_min=("silhouette", "min"), silhouette_max=("silhouette", "max"),
        inertia_min=("inertia", "min"), inertia_max=("inertia", "max"),
        bic_min=("bic", "min"), bic_max=("bic", "max"))
    baseline_profile = profiles[profiles.solution == "RFM_Baseline_Group"]
    summary = f"""# Member 3 model / method comparison

Generated from executed code on the unchanged historical RFM table. Values in this
document come from the notebook's in-memory results; full precision is retained in
`../results/`. The strategy document explains parameter choices and the rubric mapping.

## Technical conclusion

**K-Means k=2 is the leading technical candidate; K-Means k=3 is a finer secondary
candidate for Member 4's behavioural validation.** Ward k=2 and GMM k=2 remain
cross-method comparators. The rule-based four-group baseline is retained. GMM k=8 is
exported explicitly as a density-criterion diagnostic, not the recommended segmentation.
No final business winner or final segment names have been assigned.

K-Means k=2 leads the primary experiments on silhouette ({km2.silhouette:.4f}),
Davies-Bouldin ({km2.davies_bouldin:.4f}) and Calinski-Harabasz
({km2.calinski_harabasz:.2f}), with smallest/largest groups
{km2.smallest_cluster}/{km2.largest_cluster} and minimum seed-pair ARI {km2.ari_min:.4f}.
This agreement is strong internal evidence for a coarse segmentation, but it does not
establish whether two groups provide enough distinct decisions for the retailer.

## Handoff audit

- Rows and unique IDs: **{audit['rows']:,}**; duplicate IDs: {audit['duplicate_ids']};
  duplicate full rows: {audit['duplicate_rows']}; duplicate feature vectors:
  {audit['duplicate_feature_vectors']} (equal behaviours would have been retained).
- Missing / infinite values: {audit['missing_values']} / {audit['infinite_values']}.
- Historical transaction metadata: {audit['transaction_rows']:,} rows, dates
  {audit['date_min']} through {audit['date_max']}; reserved-period rows: {audit['reserved_period_rows']}.
- Existing transformations agree with Member 2's log1p/scaling formula within maximum
  absolute error {max(audit['transformation_max_absolute_errors'].values()):.3g}.
- One-order customers: {audit['one_order_customers']:,}
  ({100*audit['one_order_customers']/audit['rows']:.2f}%); raw Spearman(F,M):
  {audit['frequency_monetary_spearman']:.4f}. Standardization does not remove that overlap.
- No downstream correction, filtering or preprocessing replacement was necessary.

{table(input_summary, ['feature', 'dtype', 'min', '25%', '50%', '75%', 'max'])}

Population standard deviations are one to numerical/CSV precision. The sample standard
deviation is slightly greater because of its n-1 denominator. Exact means, deviations,
dtypes, errors and dates are retained in `input_audit.json` and `input_summary.csv`.

## Candidate comparison

Higher silhouette / Calinski-Harabasz and lower Davies-Bouldin indicate stronger geometric
separation. ARI applies only to stochastic seed repeats, not deterministic methods.
The baseline's geometric metrics evaluate fixed rules, not learned clustering.

{table(comparison, metrics)}

{table(comparison, ['method', 'k', 'configuration', 'stability_note', 'interpretability', 'strengths', 'limitations', 'role'])}

## Baseline evidence

Midrank quintile scores preserve all ties. Sum equally weighted scores and use fixed
total-score bands: Group 0 = 3–6; Group 1 = 7–9; Group 2 = 10–12; Group 3 = 13–15.
There is no score 1 for Frequency in this cohort: the tied one-order population has
its midrank in the second quintile. This is disclosed, not broken apart using IDs.
Scores can range from 1 to 5; the procedure does not guarantee every level is populated.

{table(score_ranges)}

The baseline has silhouette {base.silhouette:.4f}, Davies-Bouldin {base.davies_bouldin:.4f}
and Calinski-Harabasz {base.calinski_harabasz:.2f}. Its balanced groups and transparent
rules remain useful despite lower geometric separation. Correlated F and M influence
the total twice, while compensation can put different R/F/M patterns in one band.

{table(baseline_profile, decimals=2)}

## Complete primary experiments

All ML primary runs use seed 42 where applicable, k=2..8, and the same three scaled
features. All full-data geometric metrics are retained; no configurations are omitted.
The sizes dictionaries map local numeric cluster IDs to customer counts.

### K-Means

{table(km, ['k', 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'smallest_cluster', 'largest_cluster', 'sizes'])}

k=2 has the best separation metrics within this family. k=3 reduces inertia from
{km2.inertia:.2f} to {km3.inertia:.2f} and retains substantial groups
({km3.sizes}), while silhouette decreases to {km3.silhouette:.4f}. Its
Calinski-Harabasz ({km3.calinski_harabasz:.2f}) is close to k=2. The profiles below
show the additional differentiation; Member 4 must assess whether it matters in practice.
Increasing k beyond three reduces inertia mechanically but does not improve the reported
separation metrics. k=8 is also more initialization-sensitive (minimum ARI
{result('KMeans', 8).ari_min:.4f}). No elbow-only choice was made.

### Ward agglomerative

{table(ward, ['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'smallest_cluster', 'largest_cluster', 'sizes'])}

Ward k=2 has the highest silhouette and Calinski-Harabasz in its family, and nearly
equal groups ({ward2.sizes}). Ward k=3 has slightly better Davies-Bouldin
({ward3.davies_bouldin:.4f} versus {ward2.davies_bouldin:.4f}) but lower silhouette
({ward3.silhouette:.4f}) and more unequal groups ({ward3.sizes}). k=2 is retained
as the stronger coarse comparator; the DB disagreement is not hidden. At k=7 and k=8
the smallest group has {result('Hierarchical', 7).smallest_cluster} customers; these
fine splits require more explanation without stronger overall separation. Ward is
deterministic for fixed input/order/linkage; no fabricated random-seed ARI is reported.

### Gaussian Mixture

{table(gmm, ['k', 'aic', 'bic', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'smallest_cluster', 'largest_cluster', 'sizes'])}

{table(gmm, ['k', 'converged', 'iterations', 'mean_confidence', 'low_confidence_count', 'low_confidence_pct', 'min_covariance_eigenvalue'], decimals=6)}

GMM k=2 has the best geometric metrics within the family and balanced groups
({gm2.sizes}); {int(gm2.low_confidence_count)} customers ({gm2.low_confidence_pct:.2f}%)
have maximum posterior below 0.70. Its probabilities expose overlap, not calibrated
probabilities of buying again or of belonging to a proven true customer type.

Both AIC and BIC favour k=8 within the tested range (AIC {gm8.aic:.2f}, BIC
{gm8.bic:.2f}), but silhouette is only {gm8.silhouette:.4f} and minimum seed-pair ARI
is {gm8.ari_min:.4f}. All k>=3 primary fits have at least one eigenvalue at the
1e-6 regularization floor. The table below verifies components restricted to discrete
Frequency values. This can strongly improve continuous density fit without producing
useful behavioural groups. Higher posterior confidence alone does not resolve it.
Covariance/regularization sensitivity was not exhaustively searched; the result must not
be described as globally optimal GMM selection or a failure of all mixture models.

{table(gmm_diagnostics[gmm_diagnostics.k.isin([2, 8])], decimals=6)}

## Stability: all searched stochastic configurations

Five seeds (42, 7, 21, 99, 123), fixed parameters, ten pairwise ARIs per configuration.
The {len(seed_runs)} stochastic fits include the primary seed-42 fits once.
All completed without convergence warnings. Hierarchical/baseline seed tests are not
applicable. ARI is label-permutation invariant; the exported IDs remain primary seed 42.

{table(stability)}

{table(spread)}

ARI summarizes repeatability under initialization, not robustness to resampling,
outliers, different scalings or future periods. A stable algorithm can reproduce an
unhelpful segmentation. Per-run results and all pairs remain in the accompanying CSVs.

## Raw profiles for candidates and density diagnostic

All figures are in original units (Recency days, Frequency invoices, Monetary GBP).
Medians are emphasized for skew; means retain evidence of concentration. Cluster IDs
are neutral and local to each solution, not matched across methods. The machine-readable
profile file also covers every other searched configuration.

{table(profiles, decimals=2)}

## Visual evidence

- [K-Means search](../figures/kmeans_search.png)
- [Truncated Ward tree](../figures/ward_dendrogram.png)
- [GMM AIC/BIC](../figures/gmm_information_criteria.png)
- [Candidate PCA display](../figures/candidate_pca.png)
- [Candidate group sizes](../figures/candidate_sizes.png)

PC1 explains {100*variance[0]:.2f}% and PC2 {100*variance[1]:.2f}%
(combined {100*sum(variance):.2f}%). **PCA was not used for model fitting or metrics.**
The truncated tree uses all customers in the actual fitted Ward hierarchy.

## Seven decision dimensions

1. **Quantitative separation:** K-Means k=2 leads the primary geometric evidence;
   GMM density criteria disagree and Ward DB favours k=3 within its family.
2. **Balance:** inspect actual counts above; the baseline and coarse ML solutions
   all retain substantial groups. Small high-k groups are disclosed, not auto-rejected.
3. **Stability:** k=2 stochastic partitions are invariant over these seeds; k=3
   K-Means is highly consistent. GMM k=8 is materially less consistent.
4. **Interpretability:** baseline thresholds are easiest to audit; centroid/median
   profiles are compact; Ward adds a hierarchy; GMM adds uncertainty and assumptions.
5. **RFM suitability:** scaled log F/M limits extreme-spend domination but keeps
   correlated features and discrete Frequency. GMM covariance-floor evidence matters.
6. **Computational constraints:** {audit['rows']:,} rows and three features permit
   full-cohort fits and silhouette. A dense n-by-n float64 distance matrix would occupy
   approximately {audit['rows']**2*8/1024**2:.1f} MiB; hierarchical/pairwise work has
   quadratic scaling, whereas K-Means is more straightforward for larger cohorts.
   The bounded search uses seven k values, explicit restarts and one numerical thread;
   no timing benchmark or unsupported runtime comparison is claimed.
7. **Practical usefulness:** two groups are economical to explain, while three may
   support differentiated actions. Only Member 4's future evidence can establish whether
   the extra detail is useful; geometric quality alone cannot demonstrate commercial value.

## Questions for Member 4 and the group

- Validate frozen assignments using outcomes from 2011-09-09 onward, without refitting
  on those outcomes. Compare coarse/finer candidates and the rule baseline fairly.
- Agree an exact three-month endpoint (90 days differs from three calendar months),
  and account for partial final-day coverage and Christmas seasonality.
- Keep all historical customers in denominators, including those with no future orders;
  distinguish new future-only customers from this cohort and show uncertainty/sample sizes.
- Determine whether the extra k=3 profile differences correspond to useful future
  behaviour, and whether low GMM membership confidence affects interpretation.
- Qualify Monetary as retained positive spend, not net revenue/profit. Review Member 2's
  manual-line and cancellation policy before business claims; no such inputs were changed.
- Integrate the Member 3 decision draft and check the actual submitted documents, which
  are not present in this checkout. Record final segment names and method choice only
  after validation, acknowledging model selection on this validation period if applicable.
"""
    (docs / "model_comparison.md").write_text(summary, encoding="utf-8")

    handoff = f"""# Member 3 handoff to Member 4

## Inputs and scope

Input: `data/processed/rfm_table.csv`, {audit['rows']:,} unique customers.
SHA-256: `{hashes['data/processed/rfm_table.csv']}`.
Exact fitting columns: `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled`;
the latter two are standardized log1p Frequency/Monetary. CustomerID is excluded.
No missing/infinite features, duplicate IDs or downstream input adjustments.
The historical transaction audit ends at **{audit['date_max']}**.

Implemented the agreed rule-based RFM baseline, K-Means, Ward agglomerative and full
covariance GMM. ML group counts 2..8; primary seed 42. K-Means has 20 starts, GMM 10;
both have a 500-iteration cap. See modelling_strategy.md for exact parameters.

## Frozen outputs

**`data/processed/cluster_assignments.csv`** contains {len(assignments):,} rows:

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

{table(comparison, metrics)}

K-Means k=2 leads the internal geometric metrics; its five-seed minimum ARI is
{km2.ari_min:.4f}. k=3 K-Means has minimum ARI {km3.ari_min:.4f}, offering a stable
finer candidate. Ward k=2 is deterministic and balanced. GMM k=2 is seed-stable but
{gm2.low_confidence_pct:.2f}% have posterior <0.70. GMM k=8 has better AIC/BIC but
minimum ARI {gm8.ari_min:.4f}, poor geometric separation, and covariance-floor
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
"""
    (docs / "member3_handoff.md").write_text(handoff, encoding="utf-8")

    decisions = [
        ["M3-01: Feature matrix", "Raw RFM; three scaled RFM; extra features/PCA",
         "Use existing three scaled RFM; exclude ID; retain raw profiles",
         "Respect handoff; comparable numeric distance dimensions; no future data",
         f"Audit: {audit['rows']} complete customers; transformation checks passed"],
        ["M3-02: Baseline", "Quantile edges; row-order tie breaking; midrank scores",
         "Tie-preserving midrank 1–5 scores; sum; four fixed bands",
         "Same values get same scores; small practical baseline without tuning to ML",
         f"{audit['one_order_customers']} one-order ties; groups {base.sizes}"],
        ["M3-03: Search range", "Single k; 2..8; much larger search",
         "All ML methods use 2..8", "Compare coarse and finer groups within bounded assignment scope",
         "Complete experiments.csv; no hidden configurations or future-outcome selection"],
        ["M3-04: K-Means candidates", "2..8 clusters",
         "k=2 leading; k=3 finer secondary",
         "k=2 leads geometry; k=3 adds profile detail with strong repeatability",
         f"k2 silhouette {km2.silhouette:.4f}; k3 {km3.silhouette:.4f}; ARI minima {km2.ari_min:.4f}/{km3.ari_min:.4f}"],
        ["M3-05: Hierarchy", "Ward; alternative linkages; counts 2..8",
         "Ward Euclidean, retain k=2 comparator",
         "Numeric scaled data suits variance minimization; k2 best silhouette/CH and balanced; acknowledge k3 DB",
         f"k2 sizes {ward2.sizes}; DB k2 {ward2.davies_bouldin:.4f}, k3 {ward3.davies_bouldin:.4f}"],
        ["M3-06: GMM", "Full covariance 2..8; select by geometry or AIC/BIC",
         "k=2 comparator; k=8 separately labelled density diagnostic",
         "Retain disagreement; do not equate best continuous density fit with best customer groups",
         f"k8 BIC {gm8.bic:.2f}, silhouette {gm8.silhouette:.4f}, min ARI {gm8.ari_min:.4f}; covariance floor diagnostics"],
        ["M3-07: Stability", "Compare numeric IDs; pairwise ARI; resampling/temporal checks",
         "Five seeds; all ten label-invariant ARI pairs for every KMeans/GMM k",
         "Measure initialization sensitivity of fixed multistart procedure; temporal work belongs to M4",
         f"{len(seed_runs)} stochastic runs; deterministic seed ARI marked N/A"],
        ["M3-08: PCA and display", "Fit models on PCA; full 3D features; display only PCA",
         "Full 3D models; two-PC display; truncated actual Ward tree",
         "Avoid dropping model information; keep hierarchy legible",
         f"PC1/PC2 explain {100*variance[0]:.2f}%/{100*variance[1]:.2f}%"],
        ["M3-09: Business decision boundary", "Choose final winner now; technical shortlist then future validation",
         "Technical shortlist and frozen assignments to M4",
         "Internal metrics cannot establish business usefulness or future behaviour",
         "Candidate profiles, assignments and member3_handoff.md; no future rows fitted"],
        ["M3-10: Upstream issues", "Silently repair/cap; replace preprocessing; preserve and document",
         "No downstream data adjustment; record AND/OR wording, gross-spend and manual-line qualifications",
         "Handoff is numerically usable; business accounting choices need explicit group ownership",
         "M2 code uses OR; log says AND; M1 Q8 sale/reversal evidence; original hashes unchanged"],
    ]
    frame = pd.DataFrame(decisions, columns=["Decision", "Options considered", "Selected option", "Reason", "Supporting evidence"])
    (docs / "modelling_decisions.md").write_text(
        "# Member 3 modelling decisions — integration draft\n\n"
        "The shared `docs/decision-log/` contains only `.gitkeep`. This draft preserves\n"
        "existing work and is intended for the group lead to integrate after review.\n"
        "Numerical evidence is generated from the executed modelling notebook.\n\n"
        + table(frame) + "\n\n"
        "Group discussion: agree the validation endpoint, qualify retained positive spend\n"
        "and manual transactions, reconcile the cancellation-log wording, and check the\n"
        "original submission/descriptor before final reporting. These do not prevent the\n"
        "completed historical modelling handoff. Final segment naming belongs to Member 4.\n",
        encoding="utf-8")
