# Member 3 modelling strategy

## Objective and scope

Identify defensible groups of historical customers for the retailer to examine and
serve differently. There is no labelled outcome, supervised classifier, regression
target or purchase prediction. This implements the four methods and stability work
specified in the user's supplied Initial Submission commitments and rubric.
The actual Assignment Descriptor and submitted document are absent locally:
`docs/initial-submission/` and `reports/initial/` contain placeholders. The group lead
should cross-check wording against those originals before final submission.

Member 1's EDA notebook, insight log Q1–Q12 and dictionary were reviewed, including
the published spend-skew evidence. Member 2's Python source, executed notebook,
preprocessing log, figures and processed tables were reviewed. EDA used the full
observation period; its figures are background evidence, not data for choosing fits.
Member 3 consumes Member 2's historical RFM directly.

## Input contract and temporal boundary

- One row per identified customer in `data/processed/rfm_table.csv`.
- Model columns, in order: `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled`.
- Member 2 standardized Recency, log1p(Frequency), log1p(Monetary). These existing
  values enter the models without recomputing features or scaling.
- CustomerID is a join identifier, never a distance dimension. Raw Recency (days),
  Frequency (distinct invoices), Monetary (GBP positive retained line spend) are
  used only for rule scoring, auditing and profiling.
- Fit period: transactions strictly before **2011-09-09 00:00:00**. The period from
  that instant onward is reserved for Member 4. No random train/test split is needed
  for descriptive clustering of the historical cohort; temporal evaluation is separate.
- Validation checks schema, duplicates, numerical domains, finite values, transformation
  identities, population standard deviations and historical transaction dates/customer IDs.
  The transaction audit uses only the existing interim customer/date columns; it does
  not recreate the cleaning pipeline or RFM aggregation.
- SHA-256 hashes and package versions are recorded per run. Invalid inputs fail visibly
  rather than being repaired or silently filtered. Duplicate feature vectors with distinct
  IDs are retained because they represent distinct customers with equal behaviour.

### Handoff qualifications (no downstream input adjustment)

1. Member 2's log says cancellation tests use AND; its source/notebook use OR. Their
   executed evidence shows complete overlap in the retained identifiable cohort. This
   is a wording inconsistency, not a reason to alter the RFM file.
2. `Frequency_scaled` is scaled **log1p** Frequency, despite its shortened name.
   Its large one-order mass remains after transformation. StandardScaler equalizes
   variance; it does not make a discrete variable Gaussian or remove F/M correlation.
3. Member 2's code contains an author-specific absolute path. We do not rerun or edit
   it; Member 3 uses Member 1's upward project-root discovery convention.
4. Monetary excludes cancellation rows rather than netting them against sales. Thus it
   is not net revenue or profit. Member 1 Q8 records a large sale/reversal pair; removal
   of the reversal alone can leave large gross spend. Member 2 also retains valid M
   manual lines. Preserve these choices and ask the group/Member 4 to qualify business
   interpretation; do not silently cap or delete such customers downstream.

## Method justification against the rubric

| Method | Task | Data | Interpretability | Constraints |
| --- | --- | --- | --- | --- |
| Rule-based RFM | Sensible customer-grouping reference with no learned geometry | Scores original RFM by empirical midrank quintiles | Explicit 1–5 scores and four fixed total-score bands | Fast and deterministic; ties produce unequal score bins; trade-offs across R/F/M collapse into one total |
| K-Means | Discover compact behavioural groups | Three scaled numeric dimensions suit Euclidean distances | Centroids and raw median profiles are straightforward | Efficient multistart fitting; spherical compactness assumption and residual outlier sensitivity |
| Ward agglomerative | Compare nested group structures | Scaled Euclidean RFM supports Ward variance minimization | Merge hierarchy plus raw profiles exposes coarse/fine structure | Quadratic-scale memory/time is feasible at this cohort size; greedy merges cannot be undone; no native prediction for unseen rows |
| Gaussian Mixture | Allow overlapping behavioural groups | Full covariance accommodates correlated/elliptical features | Posterior membership describes ambiguity alongside profiles | More parameters and local optima; discrete Frequency can create very narrow Gaussian components and misleading density confidence |

## Baseline fixed before comparing models

Order each feature worst-to-best: descending Recency and ascending Frequency/Monetary.
Use average ranks for ties, `p=(rank-0.5)/n`, and `floor(5*p)+1` clipped to 1–5.
Equal values receive equal scores regardless of row order. This empirical quintile
approach needs no unique quantile edges and never arbitrarily splits one-order buyers.
Not every score must occur for discrete features. Exact observed score ranges and
counts are generated in `results/baseline_score_ranges.csv`.

Use equal weights, `total=R+F+M`, then neutral groups 0: 3–6, 1: 7–9, 2: 10–12,
3: 13–15. These correspond to mean-score bands <=2, >2–3, >3–4, >4. They are
simple, fixed ordinal bands rather than tuned clustering thresholds. Group numbers
are not final business names. Compensation and F/M redundancy are acknowledged.

## Parameter experiments and reproducibility

All three ML families search **2 through 8** groups. This compares a coarse split with
moderately detailed, potentially manageable groups without searching dozens of segments.
The finite range is a practical study boundary, not proof of a global optimum.

- K-Means: `init='k-means++'`, `n_init=20`, `algorithm='lloyd'`, `max_iter=500`,
  `tol=1e-4`. Multiple starts explicitly protect against a poor local initialization.
- Hierarchical: `linkage='ward'`, `metric='euclidean'`, full merge tree with distances.
- GMM: `covariance_type='full'`, `n_init=10`, `init_params='kmeans'`, `max_iter=500`,
  `tol=1e-3`, `reg_covar=1e-6`. Covariance family is held fixed to bound scope.
- Primary seed 42; stochastic repeats at 42, 7, 21, 99, 123, with all other settings fixed.
  Numerical threads are fixed at one. Versions are recorded; cross-version numerical
  changes and normal label permutation can still occur.

The notebook uses explicit major-section Markdown, small helper functions, and a
readable percent-cell `.py` source following Member 2's source/notebook convention.
`run_notebook.py` regenerates the notebook and uses a fresh kernel with the invoking
Python interpreter, so stale execution state cannot contribute results.

## Evidence and selection policy

Full-cohort silhouette (higher), Davies-Bouldin (lower), Calinski-Harabasz (higher),
group counts and size ranges are calculated on the same three-dimensional matrix,
including for the baseline. They measure geometric separation, not business value.
Compact Euclidean groupings can be favoured over overlapping density components.
K-Means inertia is within-family evidence; the elbow is not the sole selection rule.
GMM AIC/BIC are within-family density criteria, not scores to compare to K-Means.
Convergence, covariance eigenvalues and posterior ambiguity are retained as diagnostics.

All 10 seed-pair ARIs per configuration and per-seed metrics are retained. ARI compares
partitions independently of cluster numbering. Deterministic methods have seed stability
marked not applicable, not assigned a fabricated ARI of one. No seed is chosen because
it happens to give a nicer silhouette: the exports use the original seed-42 fits.

Shortlisting considers separation, balance, repeatability, original-unit profiles,
explanation burden, computational feasibility and potential practical usefulness.
Metric disagreement is reported, and multiple candidates can be retained. No future
outcomes, minimum-size exclusion filter or synthetic composite ranking determine selection.

## Visualisation and profiles

PCA is fitted on the historical scaled features **only for a two-dimensional display**.
Report PC1 and PC2 explained variance, and retain three dimensions for every model and
metric. PCA does not establish separation in hidden dimensions. Ward's dendrogram
shows the final 20 branches of the actual fitted hierarchy using SciPy; no thousands
of unreadable customer labels. Search curves and candidate size charts support decisions.

Profiles retain every cluster's count, percentage, raw median and mean R/F/M. Medians
represent skewed populations more reliably; means expose residual concentration.
Neutral labels are used throughout. Numeric labels are local to each configuration.

## Member 3 / Member 4 boundary and limitations

This stage provides internal technical evidence and frozen historical assignments.
Member 4 must evaluate future behaviour, interpret and name segments, decide the final
method and produce recommendations. Initialization stability is not evidence of temporal
or sampling robustness. Future Christmas-season activity is not a typical-quarter result.
The historical cohort excludes unidentified customers and later entrants; feature overlap,
unequal customer exposure time, retained accounting effects, residual extremes, sensitivity
to scaling and untested covariance/linkage choices limit generalization. Clusters are
descriptive constructions, not proven natural customer types or causal treatment groups.

## API references

Parameters and metric behaviour were checked against the installed sklearn 1.4 series
and SciPy 1.11 documentation:

- [K-Means parameters](https://scikit-learn.org/1.4/modules/generated/sklearn.cluster.KMeans.html)
- [GaussianMixture parameters](https://scikit-learn.org/1.4/modules/generated/sklearn.mixture.GaussianMixture.html)
- [Clustering methods and evaluation](https://scikit-learn.org/1.4/modules/clustering.html)
- [SciPy truncated dendrogram](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.cluster.hierarchy.dendrogram.html)
