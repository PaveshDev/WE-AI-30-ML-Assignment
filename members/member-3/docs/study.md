# Member 3 Study Guide – ML/Data-Mining Strategy & Modelling

**Project:** IT3091 Machine Learning, Group WE-AI-30  
**Track/domain:** Guided Data Track; Retail & E-commerce  
**Dataset/lens:** UCI Online Retail; Customer Segmentation  
**My contribution:** Unsupervised grouping, method comparison and a technical handoff to Member 4.

This is a personal study guide, not the final report. It explains the existing work; no models were rerun to write it. Follow the linked files when you need the underlying evidence.

**Reading route:** understand Sections 3–9 first; use Sections 10–15 beside the results and figures; revise Sections 22–28 before a viva.

**Number convention:** metric tables use four decimal places, matching the project's comparison tables. Percentages normally use two; the raw-median table explicitly uses three. Stored CSV values retain greater precision. Counts, configuration settings and dates are exact. “N/A” means a metric was not applicable, not zero.

## 1. My role in the group

My responsibility begins with Member 2's prepared customer table. I designed the modelling strategy and rule-based baseline, ran the three agreed clustering alternatives, experimented with group counts, compared evidence, checked initialization stability, exported customer assignments and identified technical candidates.

| Responsibility | Owner |
| --- | --- |
| Dataset understanding and original exploratory analysis | Member 1 |
| Cleaning, RFM construction, log transformations and scaling | Member 2 |
| Baseline, clustering, parameter experiments, internal comparison and seed stability | Me: Member 3 |
| Future three-month behavioural evaluation, final segment names and business recommendations | Member 4 |

I did not redo the EDA or preprocessing. I did not train a purchase classifier or predict whether a customer will return. I provided neutral groups and evidence; Member 4 determines their business meaning.

**How I can explain this in a viva**

> “I took the model-ready RFM table from Member 2 and compared four ways of grouping customers. My output is a defensible technical shortlist and customer-level assignments for Member 4 to validate.”

## 2. Why my part is important

Preprocessing makes data usable, but a clean table does not itself answer which customers behave similarly. My stage converts historical purchasing features into candidate groups.

Segmentation can help a retailer consider different approaches for customers with different purchasing patterns. That is a possible use, not a benefit demonstrated by our internal metrics.

Comparing methods tests whether conclusions depend on one algorithm's assumptions. Our baseline supplies a simple reference; K-Means favours compact groups, Ward supplies a hierarchy, and GMM allows overlapping distributions. The assignment requires a sensible baseline, three alternatives and justified comparison, so choosing K-Means without evidence would leave both a scientific and an assessment gap.

## 3. Input from Member 2

**Main input:** [rfm_table.csv](../../../data/processed/rfm_table.csv).  
**Preparation record:** [Member 2 preprocessing log](../../member-2/docs/preprocessing_log.md).  
**Audit evidence:** [input_audit.json](../results/input_audit.json) and [input_summary.csv](../results/input_summary.csv).

Member 2 removed duplicates, missing customer IDs, cancellation/negative-quantity rows, non-positive prices and selected non-product codes. The historical cutoff was applied before customer aggregation. The output represents retained transactions before **2011-09-09 00:00:00**.

### Actual handoff checks

| Check | Observed result |
| --- | --- |
| Rows / unique CustomerIDs | 3362 / 3362 |
| Columns | 9 |
| Missing values / infinite values | 0 / 0 |
| Duplicate IDs / duplicate rows / duplicate model-feature vectors | 0 / 0 / 0 |
| Existing historical transaction rows | 231806 |
| Historical date range recorded by audit | 2010-12-01 08:26:00 through 2011-09-08 19:58:00 |
| Historical rows on/after cutoff | 0 |
| One-order customers | 1366 (40.63%) |

The audit passed without downstream correction. Repeated feature vectors, had they existed, would not justify deleting different customers.

### What each column means

| Column | Category / actual dtype | Meaning |
| --- | --- | --- |
| CustomerID | Identifier; int64 | Customer join key, not behavioural magnitude |
| Recency | Raw; int64 | Whole elapsed days from the last retained purchase to the snapshot; lower means more recent |
| Frequency | Raw; int64 | Number of distinct retained invoices, not product lines or item quantity |
| Monetary | Raw; float64 | Total retained positive line spend in GBP; not profit or net revenue after returns |
| Log_Frequency | Transformed; float64 | Natural logarithm of 1 + Frequency |
| Log_Monetary | Transformed; float64 | Natural logarithm of 1 + Monetary |
| Recency_scaled | Model-ready; float64 | Standardized raw Recency |
| Frequency_scaled | Model-ready; float64 | Standardized **Log_Frequency**, despite the shorter name |
| Monetary_scaled | Model-ready; float64 | Standardized **Log_Monetary**, despite the shorter name |

Actual raw ranges are Recency **0–281 days**, Frequency **1–131 invoices**, and Monetary **£2.90–£177,729.62**. Their medians are **73 days**, **2 invoices** and **£555.015** respectively.

CustomerID is excluded because differences between identifier numbers have no purchasing meaning. Raw RFM is retained so we can explain groups in days, orders and pounds rather than only standard-deviation units.

### Scaled-feature status

| Feature | Recorded mean | Population standard deviation |
| --- | --- | --- |
| Recency_scaled | 5.9488e-12 | 0.999999999989 |
| Frequency_scaled | -9.5479e-11 | 0.999999999994 |
| Monetary_scaled | 6.2463e-12 | 0.999999999995 |

These are effectively zero means and unit population standard deviations. The displayed sample standard deviation is about **1.000149** because it uses an n−1 denominator; StandardScaler uses the population convention. This is not a scaling failure.

## 4. Why transformation and scaling matter

**Member 2 performed these operations; I consumed their results.**

Frequency and Monetary have long upper tails. A few large purchasers can otherwise dominate distance calculations. Member 2 used `log1p(value)`, meaning the natural logarithm of one plus the value, to compress that tail while preserving order.

Recency was standardized without a log transform. It represents elapsed time and had a much more bounded range than spending. We inherited that choice; my work does not claim that an experiment proved logging Recency would be worse.

StandardScaler subtracts each feature's mean and divides by its population standard deviation:

```text
scaled value = (prepared value - its mean) / its standard deviation
```

Here “prepared value” is raw Recency, Log_Frequency or Log_Monetary. A monetary difference measured in pounds should not overwhelm differences measured in days simply because pounds have larger numbers.

Equal variance does **not** mean that every customer contributes equally, every feature is independent, or the data have become Gaussian. Historical Frequency/Monetary Spearman correlation is **0.7842**. The large one-order population remains a discrete mass after logging, which helps explain later GMM diagnostics.

Member 3 verified the existing transformation relationships with an absolute tolerance of **1e-7**. The largest recorded discrepancy was about **4.59e-09**, consistent with CSV precision. The algebra used for this check never replaced the input matrix or refitted a scaler.

**How I can explain this in a viva**

> “Member 2 logged the skewed order counts and spend, then standardized the three prepared features. I used those existing columns so distance would not be dominated by the units of money, while still acknowledging that Frequency and Monetary overlap.”

## 5. Modelling feature matrix

`X` is the numeric table presented to the clustering methods: **3,362 rows by 3 columns**. Each row is one customer and each column is one prepared behavioural dimension.

The important code is:

```python
FEATURES = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
X = rfm[FEATURES].to_numpy(dtype=float, copy=True)
X.setflags(write=False)
```

The NumPy array contains values for computation; it has no target column. Marking it read-only helps protect the shared input. CustomerID remains in the original dataframe for joining results back to people.

All ML methods and geometric metrics use the same `X`. Otherwise a difference in scores might reflect different input features rather than different grouping methods. The baseline creates labels from raw RFM rules, but its separation metrics are calculated on this same `X`.

## 6. Why this is unsupervised learning

We have no known correct customer segments. There is no `y_train`, `y_test`, purchase target or classification accuracy calculation.

| Supervised prediction | Our unsupervised segmentation |
| --- | --- |
| Learns from known outcome labels | Uses behavioural features without known segment labels |
| Could learn whether a customer buys again | Groups customers by historical RFM similarity |
| Evaluates predictions against actual target outcomes | Examines geometry, sizes, profiles and stability |
| Produces a predicted outcome | Produces cluster memberships |

The models are fitted on the historical cohort. Future transactions are reserved for later behavioural evaluation, not used as training labels or clustering inputs. No random supervised train/test split is part of this workflow.

A code call such as `fit_predict(X)` means “fit and return cluster assignments” here. Likewise, GMM `predict_proba(X)` returns component-membership probabilities, not future-purchase probabilities.

## 7. The four methods used

### 7.1 Rule-based RFM scoring — baseline

**What/why:** an explicit grouping rule that gives us a practical reference before judging learned clusters. It uses familiar RFM behaviour and is easy to explain.

**How we scored:**

1. Rank each variable from worst to best: descending Recency; ascending Frequency and Monetary.
2. Give ties their **average rank**.
3. Calculate `p = (average_rank - 0.5) / n`.
4. Calculate `floor(5*p) + 1`, clipped to scores 1–5.
5. Add the three equally weighted scores and assign a fixed total-score band.

Lower Recency receives a better score; higher Frequency/Monetary receive better scores. “Better” is the scoring convention, not proof of future customer value.

This midrank method avoids duplicate quantile-edge errors and never splits ties using CustomerID or row order. With **1,366** one-order customers, their shared midrank falls in the second quintile. Consequently **Frequency score 1 is unused**: one order scores 2, two orders score 3, three or four orders score 4, and five or more score 5 in this dataset. We did not distort ties to force equal bins.

[baseline_score_ranges.csv](../results/baseline_score_ranges.csv) records observed minima/maxima, not universal thresholds for every possible new dataset.

**Result: four neutral groups.**

| Group | R+F+M total | Customers | Percentage |
| --- | --- | --- | --- |
| 0 | 3–6 | 935 | 27.81% |
| 1 | 7–9 | 905 | 26.92% |
| 2 | 10–12 | 803 | 23.88% |
| 3 | 13–15 | 719 | 21.39% |

The combined score can theoretically range from 3 to 15; not every theoretical total must occur. The rules were fixed before inspecting ML results. We did not create all possible three-score combinations as separate segments.

Baseline silhouette is **0.2220**, Davies–Bouldin **1.4174**, and Calinski–Harabasz **2097.5180**. These measure its geometric separation on `X`; they do not make the rule an ML clustering model.

**Meaning:** the baseline supplies sizeable, explainable groups but weaker geometric separation than the leading candidate. Summing can hide contrasting R/F/M patterns, and correlated F/M influence the total twice. Transparency remains a strength even when a separation score is lower.

**How I can explain this in a viva**

> “My baseline gives tie-preserving RFM scores and combines them into four fixed groups. It is a reference based on explicit rules, so I can test whether the more complex clustering methods add useful structure.”

### 7.2 K-Means

**What:** divide customers into k groups by repeatedly assigning them to a nearby centre and updating each centre.

| Term | Meaning in this work |
| --- | --- |
| Cluster | Customers assigned to the same group |
| Centroid | Mean position of a group's customers in the three scaled dimensions |
| Distance | Euclidean separation between a customer and a centroid in `X` |
| Iteration | An assignment/update cycle during fitting |
| Inertia | Total squared distance from customers to their assigned centroids |

**Why:** numeric scaled RFM suits Euclidean distances. With this cohort, a bounded multistart search is feasible, and centroid/raw-profile explanations are relatively simple. K-Means is computationally efficient in general; we did not produce a measured speed benchmark against other methods. It tends to favour compact, roughly spherical groups and can still respond to extremes.

**Actual configuration:** k=2 through 8; `random_state=42`; `init="k-means++"`; `n_init=20`; `algorithm="lloyd"`; `max_iter=500`; `tol=1e-4`.

- k-means++ chooses spread-out starting centres.
- Twenty starts reduce dependence on one poor initialization; the best inertia run is retained.
- The iteration cap bounds work and tolerance controls stopping.
- A seed makes the random initialization repeatable; it does not guarantee the best possible partition.

**All primary K-Means results:**

| k | Inertia | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Smallest / largest group |
| --- | --- | --- | --- | --- | --- |
| 2 | 5216.2813 | 0.4114 | 0.9045 | 3136.7877 | 1395 / 1967 |
| 3 | 3517.8854 | 0.3728 | 0.9158 | 3135.7482 | 866 / 1520 |
| 4 | 2714.5012 | 0.3460 | 0.9498 | 3039.6614 | 470 / 1112 |
| 5 | 2346.0046 | 0.3239 | 0.9941 | 2768.8901 | 184 / 965 |
| 6 | 2030.8302 | 0.3148 | 0.9696 | 2662.2846 | 190 / 758 |
| 7 | 1826.6904 | 0.3010 | 0.9809 | 2528.2542 | 184 / 725 |
| 8 | 1672.9705 | 0.2812 | 1.0494 | 2409.5458 | 134 / 617 |

**k=2:** strongest primary geometric scores, with Cluster 0 containing **1,967** customers and Cluster 1 **1,395**. Minimum seed-pair ARI is **1.0000**. Its raw profiles distinguish higher-recency/lower-frequency/lower-spend customers from more recent/higher-frequency/higher-spend customers.

**k=3:** retains sizeable groups **866, 976, 1,520**, minimum ARI **0.9982**, and Calinski–Harabasz close to k=2. Its profiles separate a more recent/higher-frequency group, a high-recency/low-frequency group and an intermediate group. Section 15 shows the actual raw values.

Inertia falls as more centres provide more flexibility. We therefore did not choose the minimum inertia or an elbow alone. k=3 trades some separation for potentially useful detail. Whether that detail helps the retailer is still a Member 4 question.

### 7.3 Hierarchical / Ward clustering

**What:** agglomerative means “build up by merging.” Start with individual customers and repeatedly merge groups, producing a nested tree.

**Ward linkage** chooses the merge that causes the smallest increase in within-group squared dispersion. It is more specific than simply joining the two nearest individual customers. Scaled numeric Euclidean RFM fits this variance-based approach.

**How:** sklearn `AgglomerativeClustering`, Ward linkage, Euclidean metric, k=2 through 8, with the complete merge tree and distances retained.

**Result:** retained **Ward k=2** has silhouette **0.3928**, Davies–Bouldin **0.9468**, Calinski–Harabasz **2896.4257**, and groups **1,717 / 1,645**.

There is a genuine metric disagreement: Ward k=3 improves Davies–Bouldin to **0.9317**, but silhouette drops to **0.3389** and groups are less balanced (**1,645 / 1,337 / 380**). k=2 remains the coarse comparator.

The dendrogram shows the actual fitted merging structure, simplified for readability. Ward has no random initialization, so seed repetitions do not test it in the same way as K-Means/GMM. Fixed-data determinism is not proof of robustness to new samples or changed customer behaviour.

**How I can explain this in a viva**

> “Ward gives a nested grouping of customers, merging groups that add the least internal variation. I retained its two-group solution as a comparator, while reporting that three groups had a slightly better Davies–Bouldin score.”

### 7.4 Gaussian Mixture Model

**What:** represent the distribution as a mixture of Gaussian components. Think of a component as a three-dimensional cloud with a centre, spread and orientation. It is a statistical approximation, not automatically a real business segment.

**Why:** customers may lie between behavioural groups; a full-covariance mixture can describe elliptical, correlated clouds and overlapping membership.

**How:** 2–8 components, full covariance, seed 42, ten initializations, `init_params="kmeans"`, a 500-iteration cap, `tol=1e-3`, and `reg_covar=1e-6`. Fitting iterates between estimating memberships and updating component parameters. Exported hard labels use the highest membership probability.

| K-Means | GMM |
| --- | --- |
| Assigns each customer to a nearest centroid | Estimates membership probabilities, then can choose the largest |
| Optimizes squared distance | Fits a probability density |
| Compact-centre view of a group | Centre, covariance shape and mixture weight |
| No built-in posterior membership probabilities | Can show customers with ambiguous component membership |

**AIC/BIC:** both balance density fit against model complexity. Better fit lowers the criterion; more parameters increase its penalty. BIC's penalty also depends on sample size. Lower values are preferred among these comparable GMM fits, not across unrelated K-Means inertia scores. Negative values are possible and do not indicate a calculation error.

**Actual primary results:**

| GMM components | AIC | BIC | Silhouette | Minimum seed ARI |
| --- | --- | --- | --- | --- |
| 2 | 21957.1243 | 22073.4098 | 0.3802 | 1.0000 |
| 8 | -7830.1966 | -7346.6936 | 0.1597 | 0.6517 |

**GMM k=2** is the retained comparator. It has groups **1,650 / 1,712**, Davies–Bouldin **0.9724** and Calinski–Harabasz **2723.7079**. **637 customers (18.95%)** have maximum posterior below **0.70**, our descriptive ambiguity threshold.

**GMM k=8** has the lowest primary AIC and BIC within the tested range, but weaker separation and materially lower seed agreement. We kept it in the export as `GMM_k8_Diagnostic`, not as the strongest segmentation.

**What “covariance floor” means:** covariance describes the spread of a component. The added diagonal regularization prevents a direction from collapsing to zero variance. A minimum covariance eigenvalue around **0.000001** means the cloud has become extremely thin in some direction.

The real diagnostics show k=8 components containing only Frequency 1, 2, 3 or 4 values, with eigenvalues at that floor. All primary k≥3 fits have at least one such narrow component. A continuous Gaussian density can fit these repeated discrete values very closely and improve AIC/BIC without yielding better-separated or more useful customer groups. This is a modelling-assumption concern, not evidence that Member 2's data should be silently changed.

**Meaning:** honest judgement reports the information-criterion winner and its weaknesses. It does not hide k=8, declare all GMMs invalid, or equate high membership confidence with true segments.

## 8. Metrics used to compare methods

| Metric | What it measures | Better direction | How we used it |
| --- | --- | --- | --- |
| Silhouette | How close a customer is to its own group compared with the nearest alternative group, averaged across customers | Higher; range −1 to 1 | Full-data Euclidean separation on the same three scaled features, including baseline labels |
| Davies–Bouldin | Within-group spread relative to distances between group centres, emphasizing each group's most similar other group | Lower | Check compactness and separation alongside silhouette |
| Calinski–Harabasz | Between-group dispersion relative to within-group dispersion, adjusted for group and sample counts | Higher | Complementary separation evidence on the same cohort |
| Inertia | Sum of squared distances to assigned K-Means centroids | Lower within comparable fits, but generally falls with more centres | Inspect diminishing returns; never select k from it alone |
| AIC | Density-fit criterion with a parameter-count penalty | Lower within the comparable GMM search | Retain the density-fit/complexity trade-off |
| BIC | Density-fit criterion with a sample-size-dependent complexity penalty | Lower within the comparable GMM search | Compare component counts and expose disagreement with geometric scores |
| Adjusted Rand Index (ARI) | Agreement of two partitions, adjusted for chance and unaffected by numeric label names | Higher; 1 is identical grouping, near 0 is chance-level agreement under its adjustment; can be negative | Compare stochastic runs with different seeds, not predicted labels against a true target |

A silhouette near zero suggests overlapping boundaries; a negative value suggests a customer may be closer on average to another group. **0.4114 is not 41.14% classification accuracy.**

The metrics answer different questions. AIC/BIC measure density fit; geometric metrics favour separated compact groups; ARI measures repeatability. Group sizes, profiles, explanation burden and future usefulness supply evidence none of those scores captures alone. There is no universal score threshold proving a good retail segmentation.

## 9. Stability analysis

**What:** whether the same customers tend to be grouped together when initialization changes.

**Why:** a solution that changes substantially with a seed is harder to defend as a reliable summary.

**How:** keep X, k and every other method parameter fixed; use seeds **42, 7, 21, 99, 123**. Each K-Means run still uses 20 starts; each GMM run still uses ten. Seeds and internal restarts are different levels of repetition.

- Two stochastic methods × seven k values × five seeds = **70 fitted stochastic runs**.
- There are **10 distinct pairs** among five seeds.
- Two methods × seven k values × ten pairs = **140 ARI comparisons**.
- Seed 42 is reused from the primary experiments; it is not counted twice.
- The full workflow additionally has seven deterministic Ward fits and the baseline.

| Configuration | Mean ARI | Minimum ARI | Maximum ARI |
| --- | --- | --- | --- |
| GMM k=2 | 1.0000 | 1.0000 | 1.0000 |
| GMM k=8 | 0.7864 | 0.6517 | 0.9978 |
| KMeans k=2 | 1.0000 | 1.0000 | 1.0000 |
| KMeans k=3 | 0.9989 | 0.9982 | 1.0000 |
| KMeans k=8 | 0.8688 | 0.7250 | 0.9795 |

K-Means k=2 has ARI **1.0000 for every seed pair**: the returned partitions agree. This does not require identical centroid numbers or exactly identical inertia; the saved runs contain slight inertia differences even when partitions agree.

GMM k=8's minimum ARI **0.6517** and mean **0.7864** show less consistent partitions. GMM k=2 remains fully consistent across these seeds; the instability claim is configuration-specific.

Cluster IDs are arbitrary. If the same two groups exchange names 0 and 1, direct numeric-label comparison would falsely report disagreement. ARI compares membership relationships instead.

This is **initialization stability**, not bootstrap/sample stability, temporal validation or proof of correct business segments. Baseline/Ward seed ARI is N/A; we did not invent a perfect score for them.

## 10. Actual model comparison results

Source: [candidate_comparison.csv](../results/candidate_comparison.csv). All rows use the same historical cohort and scaled-feature metric space.

| Method | Groups | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Mean ARI | Minimum ARI |
| --- | --- | --- | --- | --- | --- | --- |
| RFM_Baseline | 4 | 0.2220 | 1.4174 | 2097.5180 | N/A | N/A |
| KMeans | 2 | 0.4114 | 0.9045 | 3136.7877 | 1.0000 | 1.0000 |
| KMeans | 3 | 0.3728 | 0.9158 | 3135.7482 | 0.9989 | 0.9982 |
| Ward | 2 | 0.3928 | 0.9468 | 2896.4257 | N/A | N/A |
| GMM | 2 | 0.3802 | 0.9724 | 2723.7079 | 1.0000 | 1.0000 |
| GMM diagnostic | 8 | 0.1597 | 1.9549 | 1265.3859 | 0.7864 | 0.6517 |

Read it in three steps:

1. K-Means k=2 leads all three primary geometric metrics and is seed-stable.
2. K-Means k=3 sacrifices some silhouette/DB separation but keeps nearly the same CH score, stable groups and finer raw profiles.
3. Ward/GMM k=2 are credible comparisons; the rule baseline is transparent; GMM k=8 illustrates why best density fit is not enough.

Balance also matters: a method could isolate a tiny group and still improve one score. We recorded every group size rather than hiding inconvenient clusters. The comparison considers separation, balance, stability, interpretability, suitability for RFM, computation and potential practical usefulness.

## 11. Leading technical candidates

**Leading: K-Means k=2.** Its combined evidence is strongest: best primary geometric scores, substantial groups and invariant partitions across the tested seeds. It provides a simple coarse explanation of purchasing patterns.

**Secondary: K-Means k=3.** Its stable, substantial groups separate more recent/higher-frequency behaviour, high-recency/low-frequency behaviour and an intermediate profile. This creates an option for more detailed interpretation without claiming that more groups are automatically better.

**Technical candidate ≠ final business winner.** Neither the internal metrics nor seed agreement establishes that different groups will behave differently in the future or benefit from different actions. Member 4 must assess that using the reserved period, then choose final names and recommendations.

## 12. Figures I created

### 12.1 kmeans_search.png

[Open the K-Means search figure](../figures/kmeans_search.png).

**What/axes:** three panels show k on the horizontal axis and inertia, silhouette or Davies–Bouldin on the vertical axis. It asks what happens as grouping becomes more detailed.

**Observation:** inertia falls throughout; silhouette is highest at k=2 and declines across the tested range; Davies–Bouldin is lowest at k=2. The curves support k=2, while the inertia reduction and raw profiles justify examining k=3. An elbow does not decide the result alone.

**How I can explain this figure in a viva**

> “Increasing k reduces inertia because more centres can fit customers more closely. However, the separation scores favour two clusters, so I combined the curves with stability and profiles rather than selecting the largest k or relying only on an elbow.”

### 12.2 ward_dendrogram.png

[Open the Ward dendrogram](../figures/ward_dendrogram.png).

**What/axes:** the horizontal axis lists merged branches; parentheses show their customer counts. Vertical height is Ward linkage distance: higher joins indicate more costly merges in the tree, not more customers or more recent purchases.

**How/observation:** SciPy displays the final **20 branches** from sklearn's actual full-customer hierarchy. Truncation simplifies the display, not the fitted dataset. The last merge is visibly higher than the earlier major merges, making a coarse two-way split worth examining. The quantitative comparison remains necessary; the tree is not proof of two true customer types.

**How I can explain this figure in a viva**

> “This shows how Ward gradually merged customer groups. I displayed the last twenty branches so it stays readable, but fitted all customers. The high final merge suggests a meaningful coarse split, which I checked using metrics and sizes.”

### 12.3 gmm_information_criteria.png

[Open the GMM information-criteria figure](../figures/gmm_information_criteria.png).

**What/axes:** two panels plot component count against AIC and BIC. Lower scores indicate a more favourable balance of density fit and complexity within this search.

**Observation:** both reach their lowest primary value at eight components. That is the edge of the tested range, so it is not proof of a global optimum. Separation, stability and covariance diagnostics explain why this solution remains diagnostic.

**How I can explain this figure in a viva**

> “The information criteria prefer eight Gaussian components in our tested range. I retained that result honestly, but its groups were less separated and less stable, with very narrow components around repeated frequency values. Lowest BIC was therefore not enough to choose the final segmentation.”

### 12.4 candidate_pca.png

[Open the candidate PCA figure](../figures/candidate_pca.png).

**What:** PCA forms new linear directions that capture as much variation as possible. PC1 captures the most; PC2 captures the next most in a perpendicular direction. This gives a two-dimensional picture of our three-dimensional X.

**Actual axes:** PC1 explains **73.12%**, PC2 **19.86%**, combined **92.98%**. The plot axes themselves display one-decimal percentages. This is feature variance, not accuracy or proportion of customers classified correctly.

**Panels:** K-Means k=2, K-Means k=3, Ward k=2 and GMM k=2. Each dot is a customer, coloured by the panel's assignment. The cloud is the same; algorithms draw different partitions through it. Colours are local: Cluster 0 in one panel need not be Cluster 0 in another.

**Meaning:** useful for seeing coarse/finer partitions, overlap and extreme points, but the projection can hide differences in the remaining dimension. **All clustering and metrics used the three original scaled RFM features; PCA was visualization only.**

**How I can explain this figure in a viva**

> “I used PCA only to display the customer groups in two dimensions, capturing about 92.98% of the feature variance. The models still used all three scaled RFM features. Matching colours across panels do not imply matching groups, and visible separation is supporting evidence rather than a validation result.”

### 12.5 candidate_sizes.png

[Open the candidate size figure](../figures/candidate_sizes.png).

**What/axes:** six panels compare the baseline, K-Means k=2/k=3, Ward k=2, GMM k=2 and GMM k=8 diagnostic. The horizontal axis is the local group ID; the vertical axis is customer count. Bars also show percentages.

**Observation:** the baseline has four fairly balanced groups, while the two-group candidates retain large populations. K-Means k=3's groups are all substantial. GMM k=8 ranges from **95** to **772** customers, making a more fragmented alternative.

A tiny cluster may represent a genuine niche or an outlier-driven split; a huge cluster may conceal useful differences. Unequal sizes are not automatically wrong. The purpose is to judge practical scale alongside profiles and metrics, not force equal group sizes.

**How I can explain this figure in a viva**

> “This checks whether candidate solutions create groups large enough to interpret and compare. The coarse candidates have substantial groups, while the eight-component GMM fragments the population more. I used these counts as evidence, not as a rule that every group must have the same size.”

## 13. Result CSV / JSON files

All paths below are inside [members/member-3/results](../results). These are evidence files, not extra models.

| File | Purpose | Important contents | Who uses it |
| --- | --- | --- | --- |
| [baseline_score_ranges.csv](../results/baseline_score_ranges.csv) | Makes score rules observable | 14 populated feature/score rows; observed min, max and count; shows missing Frequency score 1 | Me explaining ties; Member 4 explaining baseline |
| [candidate_comparison.csv](../results/candidate_comparison.csv) | Summarizes the retained choices | Six candidates/reference/diagnostic rows, metrics, ARI, configuration, strengths, limitations and role | Me, Member 4, group report |
| [cluster_profiles.csv](../results/cluster_profiles.csv) | Connects technical labels to original behaviour | 109 cluster rows across 22 configurations; counts, percentages, raw means and medians | Member 4 and my profile explanations |
| [experiments.csv](../results/experiments.csv) | Keeps the entire primary search visible | 22 rows: baseline plus seven counts for each ML method; sizes, objectives, metrics, convergence and merged stability | Anyone checking selection was not cherry-picked |
| [gmm_component_diagnostics.csv](../results/gmm_component_diagnostics.csv) | Explains the density-fit concern | 35 component rows across the seven GMM counts; Frequency diversity/range and minimum covariance eigenvalue | Me explaining why BIC is insufficient |
| [input_audit.json](../results/input_audit.json) | Records the handoff contract checks | IDs, quality checks, dtypes, date bounds, transformation errors, scaling, one-order count and correlation | Me/Member 2 checking provenance |
| [input_summary.csv](../results/input_summary.csv) | Provides numerical context for each input column | Nine rows with dtype, count, mean, sample standard deviation and quantiles/ranges | Me explaining feature scales |
| [run_manifest.json](../results/run_manifest.json) | Identifies the exact run setup and artifacts | Python/package versions, input/source/output SHA-256 hashes, model parameters, seeds, PCA variance and check flags | Anyone reproducing or auditing |
| [stability_pairs.csv](../results/stability_pairs.csv) | Retains individual agreement comparisons | 140 rows, each method/k/seed-pair and ARI | Me defending stability claims |
| [stability_runs.csv](../results/stability_runs.csv) | Shows each stochastic run's evidence | 70 rows with seed, sizes, metrics and appropriate objective/diagnostic fields | Anyone investigating initialization sensitivity |
| [stability_summary.csv](../results/stability_summary.csv) | Makes stability easy to compare | 14 configurations; ARI mean, minimum, maximum, standard deviation and pair count | Me and Member 4 shortlisting |

A blank inertia for GMM or blank BIC for Ward is expected: those objectives are not shared across methods. These blanks should not be confused with missing modelling input.

## 14. cluster_assignments.csv

[Open cluster_assignments.csv](../../../data/processed/cluster_assignments.csv).

This is one of my most important handoff files: **3,362 rows and 12 columns**. One row is one historical customer, with alternative assignments kept together.

| Column(s) | Meaning |
| --- | --- |
| CustomerID | Unique customer join key |
| Recency, Frequency, Monetary | Original historical raw RFM |
| RFM_Baseline_Group | Fixed neutral rule-based group, 0–3 |
| KMeans_k2 | Leading candidate's primary seed-42 assignment |
| KMeans_k3 | Secondary candidate's primary seed-42 assignment |
| Hierarchical_k2 | Deterministic Ward comparator assignment |
| GMM_k2 | Retained GMM comparator's hard assignment |
| GMM_k8_Diagnostic | Eight-component density diagnostic's hard assignment |
| GMM_k2_MaxPosterior | Largest membership probability under GMM k=2 |
| GMM_k8_Diagnostic_MaxPosterior | Largest membership probability under GMM k=8 |

The posterior columns are not a complete matrix of probabilities for every component. They summarize confidence in the exported maximum-posterior assignment.

Member 4 should aggregate future transactions to **one outcome row per CustomerID**, then left-join those outcomes onto this file. That keeps the original historical cohort, including customers who never return. A raw transaction-level join would duplicate customer rows and can corrupt denominators.

Only after checking coverage and transaction rules should absent purchases be treated as zero. Future-only customers should be reported separately. Labels and historical features stay frozen; the future period must not be used to refit this segmentation.

## 15. cluster_profiles.csv

A **cluster profile** summarizes the customers assigned to one group. [cluster_profiles.csv](../results/cluster_profiles.csv) contains one row per configuration/cluster, not one row per customer.

| Field(s) | Interpretation |
| --- | --- |
| solution, cluster | Which configuration and its local group ID |
| customers, percentage | Number/share of the historical cohort in that group |
| median_recency, median_frequency, median_monetary | Middle customer values in original RFM units |
| mean_recency, mean_frequency, mean_monetary | Arithmetic averages, which expose the effect of large values |

We use raw values because “125 days since the last purchase” is easier to explain than a standardized coordinate. The **median** is less influenced by extreme spend; the **mean** helps show concentration. Neither alone describes every individual customer.

### Reading the leading candidates

The following median table uses three decimals to preserve half-penny values present in the stored summaries.

| Solution | Cluster | Customers | % | Median R (days) | Median F (invoices) | Median M (GBP) |
| --- | --- | --- | --- | --- | --- | --- |
| KMeans_k2 | 0 | 1967 | 58.51 | 125.000 | 1.000 | 304.100 |
| KMeans_k2 | 1 | 1395 | 41.49 | 27.000 | 4.000 | 1561.060 |
| KMeans_k3 | 0 | 866 | 25.76 | 22.000 | 6.000 | 2284.955 |
| KMeans_k3 | 1 | 976 | 29.03 | 192.000 | 1.000 | 246.275 |
| KMeans_k3 | 2 | 1520 | 45.21 | 58.000 | 2.000 | 498.480 |

For K-Means k=2, Cluster 0 has higher median recency and lower median orders/spend. Cluster 1 has more recent and more frequent purchasing. For k=3, the intermediate Cluster 2 has median Recency **58 days** and Frequency **2**, between the other two profiles.

A concrete skew example: K-Means k=2 Cluster 1 has median Monetary **£1,561.06**, but mean Monetary **£3,299.77**. The higher mean shows why the median is useful for describing a typical group member.

The full CSV covers **22 configurations** and **109 cluster rows**. Counts sum to the cohort within each solution; summing across solutions would count the same people repeatedly.

“Cluster 0” is a neutral identifier, not a ranking or final business name. Calling a group “lost” from high historical recency alone would assume the future behaviour Member 4 still needs to check.

## 16. Important notebook

[03_modelling_and_comparison.ipynb](../notebooks/03_modelling_and_comparison.ipynb) is the main executed Member 3 analysis.

The saved file has **26 cells: 13 Markdown and 13 code cells**, with code execution counts **1–13** and no error outputs. Its source matches the adjacent percent-cell Python file.

The sections are:

1. Objective, scope and setup.
2. Load and audit Member 2's handoff.
3. Define X.
4. Rule-based baseline.
5. K-Means search.
6. Ward search and dendrogram.
7. GMM search and criteria.
8. Seed stability.
9. Candidate comparison.
10. Raw profiles and GMM diagnostics.
11. PCA display and group sizes.
12. Export assignments and check the CSV round trip.
13. Generate handoff/decision documents and run manifest; verify input hashes.

In one flow:

**Load → audit → X → baseline → K-Means → Ward → GMM → stability → compare → profile/visualize → export → handoff/checks.**

The final saved output says all modelling, profile, export and input-preservation checks passed. The notebook and manifest provide the saved execution evidence; the runner separately supports a two-fresh-kernel artifact comparison. Creating this guide did not rerun either workflow.

## 17. Important Python files

| File | Actual purpose and important logic |
| --- | --- |
| [03_modelling_and_comparison.py](../notebooks/03_modelling_and_comparison.py) | Readable source of the full analysis; `# %%` markers separate cells. Loads existing data, runs searches and stability, draws figures, selects documented candidates, exports assignments and builds the manifest. Assertions guard the leading-candidate claims if results change. |
| [modelling_helpers.py](../notebooks/modelling_helpers.py) | Reusable data-audit, scoring, model, metric, stability, profile and Markdown-table functions; keeps the main workflow readable. |
| [modelling_report.py](../notebooks/modelling_report.py) | `write_reports(...)` generates model_comparison.md, member3_handoff.md and modelling_decisions.md from computed data. It does not generate modelling_strategy.md or this study guide. |
| [run_notebook.py](../notebooks/run_notebook.py) | Converts percent cells into notebook cells, launches a fresh kernel using the invoking Python interpreter, executes sequentially and saves the notebook. With `--verify-reproducible`, it runs twice and compares manifests and output hashes. |
| [verify_outputs.py](../notebooks/verify_outputs.py) | Independently checks saved inputs, source/output hashes, assignments, metrics, counts, baseline behaviour, stability coverage and notebook execution without fitting new models. |

### Helpers I should recognize

- `fingerprint`: computes a SHA-256 content hash.
- `audit_input`: rejects unexpected schema, invalid IDs/domains or non-finite features; verifies transformation relationships and historical date/customer metadata.
- `rfm_baseline`: performs the tie-preserving scoring and fixed group assignment.
- `make_model`: centralizes each method's exact parameters.
- `cluster_metrics`: calculates separation metrics and ordered size summaries.
- `experiment`: fits one method/configuration, returns its metrics, labels and model, and adds method-specific evidence. Convergence warnings become errors.
- `repeat_experiments`: runs the five-seed comparisons and creates run/pair/summary tables.
- `profiles`: groups raw RFM by assigned label and computes counts, percentages, means and medians.
- `markdown_table`: formats computed tables without an extra table-formatting dependency.

The main script's `save_table` and `save_figure` save evidence consistently. The report helper converts that evidence into readable documents; it does not independently fit models.

### What the independent verifier actually checks

It compares exported raw columns with the source, recalculates candidate geometric metrics from **saved labels**, checks profile counts, and verifies baseline ties, scoring direction and row-order independence. It also checks the **70 runs / 140 pairs / 14 stability summaries** and sequential notebook execution.

Recalculating a score from existing assignments is not the same as fitting a new clustering model. However, I did not need to execute that verifier again to write this guide.

## 18. Modelling documentation files

| Document | Question it answers |
| --- | --- |
| [modelling_strategy.md](modelling_strategy.md) | Why these methods, features, parameters and metrics? Explains task/data/interpretability/constraints, scope and limitations. |
| [modelling_decisions.md](modelling_decisions.md) | Which options were considered, what was chosen, and what evidence supports it? It is an integration draft because the shared Decision Log is a placeholder. |
| [model_comparison.md](model_comparison.md) | What actually happened in the experiments? Contains complete search evidence, stability, profiles, diagnostics and the technical recommendation. |
| [member3_handoff.md](member3_handoff.md) | What does Member 4 receive, what do the columns mean, and how should future evaluation proceed? |

This study guide adds a learning explanation; it does not replace those technical records.

## 19. Reproducibility

**Meaning:** another person should be able to follow the same steps on the same inputs and check how the results were obtained.

| Mechanism | Why it matters here |
| --- | --- |
| Existing input hashes | Detects whether the RFM/history files changed |
| Fixed primary seed and repeat seeds | Makes stochastic initialization choices explicit |
| Explicit model parameters | Avoids depending silently on changing library defaults |
| Same X for all methods | Keeps the comparison consistent |
| Numerical thread counts set to one | Controls numerical execution conditions |
| Project-root discovery and relative paths | Avoids author-specific absolute paths in Member 3 code |
| Notebook generated from source and executed in a fresh kernel | Avoids stale variables and hidden cell-order dependencies |
| Saved output/source hashes and verifier | Makes changed artifacts detectable |
| Two-run verification option | Checks byte-for-byte reproducibility of generated artifacts within the same environment |

[requirements.txt](../../../requirements.txt) lists pandas, numpy, matplotlib, seaborn, openpyxl, scikit-learn and scipy. These are shared project requirements; not every package is directly imported by the Member 3 workflow. **SciPy is directly used for the truncated dendrogram.**

[requirements-notebook.txt](../requirements-notebook.txt) contains the actual notebook-generation/execution tooling: **nbformat 5.11.1, nbclient 0.11.0 and ipykernel 7.3.0**.

The recorded run used Python **3.11.3**, numpy **1.26.3**, pandas **2.3.3**, scikit-learn **1.4.0**, scipy **1.11.4** and matplotlib **3.9.4**. The manifest lists additional relevant versions. Root requirements are unpinned, so a future install may differ; exact numerical reproduction should use the recorded versions.

The existing run instructions include:

```powershell
.venv/Scripts/python.exe members/member-3/notebooks/run_notebook.py --verify-reproducible
.venv/Scripts/python.exe members/member-3/notebooks/verify_outputs.py
```

These are reference commands, **not commands executed to produce this study guide**. The runner regenerates derived Member 3 outputs. It compares assignments, result tables, figures, documents and manifests, not incidental notebook execution-time metadata.

The saved manifest's check flags are true for unchanged inputs, export round trip, finite features, historical-only input and stochastic convergence. A manifest identifies recorded artifacts; it does not guarantee business validity or by itself record how many independent reruns occurred. This newly added study.md is not part of the original manifest, which remains unchanged.

**How I can explain this in a viva**

> “I fixed the seeds and parameters, used the same historical input, and saved the versions and file hashes. The notebook runner can restart from a clean kernel and compare two runs, and the verifier checks that the exported evidence is consistent.”

## 20. Boundary between Member 3 and Member 4

| My Member 3 work | Member 4's next work |
| --- | --- |
| Technical segmentation from historical RFM | Evaluate later behaviour of those groups |
| Algorithm/parameter comparison | Assess usefulness beyond internal geometric scores |
| Initialization stability | Stronger reserved-period behavioural validation |
| Neutral profiles and frozen assignments | Final interpretation and segment naming |
| Leading/secondary technical candidates | Final method justification, recommendations and stakeholder value |

The reserved future period starts at **2011-09-09 00:00:00**. It was excluded from model fitting, PCA and candidate selection. The interim cleaned file contains history only; it is not Member 4's future-outcome dataset.

The group must agree the exact evaluation endpoint. **90 days is not identical to three calendar months**, and the original dataset ends partway through 9 December 2011. The validation period includes Christmas-season activity, so Member 4 must avoid treating it as a typical quarter.

If Member 4 uses future outcomes to choose the final model, that period has informed selection; it should not simultaneously be claimed as untouched confirmation of the chosen model.

## 21. What Member 4 receives from me

| Handoff | How Member 4 uses it |
| --- | --- |
| [cluster_assignments.csv](../../../data/processed/cluster_assignments.csv) | Join future customer outcomes to the frozen baseline/candidate labels |
| [model_comparison.md](model_comparison.md) and candidate_comparison.csv | Understand the technical trade-offs and why alternatives were retained |
| stability_runs.csv, stability_pairs.csv, stability_summary.csv | Check which configurations depend on initialization |
| [cluster_profiles.csv](../results/cluster_profiles.csv) | Start interpretation in original units without prematurely naming groups |
| K-Means k=2/k=3 recommendation | Compare coarse and finer groups against actual later behaviour |
| [member3_handoff.md](member3_handoff.md) | Follow the schema, temporal boundary, joining guidance and outstanding questions |

Member 4 can compare future repeat-purchase proportions, invoice counts and spend, retaining group denominators and uncertainty. These are **evaluation outcomes**, not targets I trained a supervised model to predict.

## 22. Key results I should remember

All values below come from the saved result files; four-decimal metric formatting is consistent with Section 10.

| Item | Verified value |
| --- | --- |
| Historical customers / model dimensions | 3,362 / 3 |
| Baseline groups; sizes in label order | 4; 935, 905, 803, 719 |
| K-Means k=2 silhouette / DB / CH | 0.4114 / 0.9045 / 3136.7877 |
| K-Means k=2 group sizes; minimum ARI | 1,967 / 1,395; 1.0000 |
| K-Means k=3 silhouette / DB / CH | 0.3728 / 0.9158 / 3135.7482 |
| K-Means k=3 sizes; minimum ARI | 866 / 976 / 1,520; 0.9982 |
| Ward k=2 silhouette / DB / CH | 0.3928 / 0.9468 / 2896.4257 |
| Ward k=2 sizes; seed ARI | 1,717 / 1,645; N/A |
| GMM k=2 silhouette / DB / CH | 0.3802 / 0.9724 / 2723.7079 |
| GMM k=2 sizes; minimum ARI | 1,650 / 1,712; 1.0000 |
| GMM k=2 posterior below 0.70 | 637 customers; 18.95% |
| GMM k=8 AIC / BIC | -7830.1966 / -7346.6936 |
| GMM k=8 silhouette; minimum ARI | 0.1597; 0.6517 |
| Stochastic seeds | 42, 7, 21, 99, 123 |
| Stochastic runs / pairwise ARIs | 70 / 140 |
| Primary experiment rows | 22: one baseline + 21 ML configurations |
| PCA variance, PC1 / PC2 / combined | 73.12% / 19.86% / 92.98%, visualization only |
| Assignment table | 3,362 rows × 12 columns |

## 23. Important modelling decisions

| Decision | Alternatives | Reason | Evidence |
| --- | --- | --- | --- |
| Use Member 2's scaled RFM | Raw units; extra features; PCA-reduced fitting | Comparable prepared dimensions without redoing another member's work | Passed input audit and transformation checks |
| Exclude CustomerID; keep raw RFM | Treat all numeric columns as inputs | ID distance is meaningless; raw values are needed to explain profiles | Explicit three-column FEATURES constant |
| Midrank baseline and four bands | Duplicate-edge quantiles; tie splitting; many RFM combinations | Equal values keep equal scores; small explainable reference | 1,366 one-order ties; four substantial groups |
| Search 2–8 across ML methods | Select one k; search many more groups | Compare coarse/finer partitions within bounded scope | All 21 ML configurations retained |
| Ward linkage | Other hierarchical linkages | Variance-based merging matches scaled Euclidean inputs | Actual hierarchy and Ward search evidence |
| K-Means k=2 leads | Other tested counts/methods | Best combined primary geometry, stable substantial groups | Section 10 scores; minimum ARI 1.0000 |
| Keep K-Means k=3 | Export only the leading solution | Stable additional behavioural detail could matter later | Near-equal CH to k=2; three raw profiles |
| Keep GMM k=8 as diagnostic only | Choose the minimum BIC automatically | Weak geometry, initialization sensitivity, covariance-floor components | AIC/BIC table, minimum ARI 0.6517 and component diagnostics |
| Use label-invariant pairwise ARI | Compare numeric labels directly | Group IDs can permute across runs | Ten pairs for each of 14 stochastic configurations |
| PCA for display only | Fit on two PCs | Avoid silently discarding a modelling dimension | Three-column X; PCA occurs after fitting/comparison |
| Hand technical choices to Member 4 | Name final business groups now | Internal evidence does not establish future usefulness | Frozen exports and explicit handoff boundary |

The shared Decision Log was not overwritten. [modelling_decisions.md](modelling_decisions.md) is the actual draft for integration.

## 24. Limitations of my Member 3 work

- **RFM is a restricted view:** it has three purchasing dimensions and does not describe all customer motivations or preferences.
- **No known correct labels:** internal metrics are not classification accuracy or proof of natural customer types.
- **Method assumptions matter:** compact K-Means groups, greedy Ward merges and Gaussian density components can partition the same people differently.
- **Feature redundancy remains:** raw F/M correlation is 0.7842; scaling does not remove overlap.
- **Discrete Frequency affects GMM:** the observed narrow components and seed sensitivity limit the k=8 interpretation. Other covariance/regularization choices were not exhaustively tested.
- **k=2 may be coarse:** strong geometry does not prove sufficient detail for retailer decisions.
- **Seed tests have limited scope:** no sample-resampling or temporal robustness claim follows from them.
- **Cohort and accounting limits:** unidentified customers and later entrants are outside the historical model. Retained positive spend excludes cancellation rows rather than netting returns; valid manual M lines remain. Monetary is not profit or net revenue.
- **Upstream wording qualification:** Member 2's log says cancellation conditions use AND, while the implementation uses OR, as documented in the Member 3 strategy. This was recorded, not silently “fixed.”
- **Scope of search:** 2–8 is a bounded investigation; an endpoint BIC minimum does not prove a global optimum.
- **Further validation is pending:** future period seasonality and endpoint definitions affect Member 4's claims.

The original submitted Assignment Descriptor/Initial Submission files are absent from this checkout. Our strategy records that the implementation follows the supplied commitments; final wording should be checked against the originals.

## 25. Common viva questions and answers

### 1. Why did you use clustering?

We wanted groups of customers with similar historical purchasing behaviour, and we had no known segment labels. Clustering fits that unsupervised objective.

### 2. Why didn't you use classification or predict future purchases?

That would require a different supervised target and answer a different question. My output is segmentation; future purchasing is reserved for Member 4's evaluation.

### 3. Why transform and scale RFM?

Member 2 logged skewed Frequency/Monetary and standardized the prepared features so units of spend would not dominate distance. I used their output; I did not repeat preprocessing.

### 4. Why exclude CustomerID?

It is only an identifier. Two nearby ID numbers do not imply similar purchasing behaviour.

### 5. Why compare four approaches?

The agreed study requires a simple baseline and three alternatives. Comparing different assumptions makes the recommendation more defensible than relying on one algorithm.

### 6. How did your baseline handle ties?

It used average ranks and midrank quintile scores. The 1,366 customers with one order stayed tied at Frequency score 2, rather than being split by row order.

### 7. Why use K-Means?

It is suitable for a manageable numeric, scaled dataset and provides centroid-based groups that can be explained using raw profiles. Its compact-group assumption remains a limitation.

### 8. What does silhouette mean?

It compares a customer's average distance to its own group with the nearest alternative group. Higher is generally better; our leading score of 0.4114 is a separation measure, not accuracy.

### 9. Why is lower Davies–Bouldin better?

It means group spread is smaller relative to separation between group centres. I used it alongside other metrics because it does not measure business usefulness.

### 10. What is Ward linkage?

It merges the pair of groups producing the smallest increase in within-group squared variation. It gives a nested hierarchy on our scaled Euclidean data.

### 11. Why include GMM, and how is it different from K-Means?

GMM models overlapping Gaussian clouds and gives membership probabilities. K-Means uses nearest-centroid hard assignment; GMM's exported labels select the largest posterior probability.

### 12. Why wasn't the lowest-BIC GMM automatically selected?

Eight components had the lowest primary BIC, but poorer geometric separation, minimum seed ARI 0.6517 and covariance-floor components associated with repeated Frequency values. I retained it as a diagnostic rather than hiding it or declaring it the winner.

### 13. What is ARI?

Adjusted Rand Index compares two partitions while correcting for chance agreement. It ignores arbitrary label numbering, so swapped cluster IDs do not count as a different grouping.

### 14. Why test several seeds?

Initialization can change stochastic solutions. I tested five seeds for every K-Means/GMM count, producing 70 run records and 140 pairwise comparisons.

### 15. Why is k=2 your leading technical candidate?

It has the best primary silhouette, Davies–Bouldin and Calinski–Harabasz scores, substantial groups and ARI 1.0000 across every tested seed pair.

### 16. Why retain k=3?

Its CH score is close to k=2, its minimum ARI is 0.9982, and the raw profiles show a stable intermediate group. Member 4 can assess whether the extra detail is useful.

### 17. Why use PCA, and did it change clustering?

It made the three-dimensional feature space viewable in two dimensions. It captured 92.98% of variance for display; all fitting and metrics still used the three scaled RFM features.

### 18. Why show means and medians in profiles?

Spend is skewed. The median describes the middle customer more robustly, while the mean shows how large spenders pull up the group average.

### 19. Why not repeat Ward with five seeds?

Ward has no random initialization for the same data/order/linkage. Seed ARI is N/A; that does not prove robustness to changed data.

### 20. What exactly does Member 4 do next?

Member 4 joins future outcomes to frozen customer assignments, evaluates behavioural differences, then gives final names, method justification and recommendations. They must retain non-returners and respect the future-period boundary.

## 26. 1-minute explanation of my part

> “I was responsible for the modelling and method comparison in our customer segmentation project. Member 2 gave me a model-ready RFM table for 3,362 customers, using only purchases before 9 September 2011. I checked the handoff and used the three existing scaled RFM features, excluding CustomerID.
>
> I compared a rule-based RFM baseline with K-Means, Ward hierarchical clustering and a Gaussian Mixture Model. For the clustering methods I tested two to eight groups. I considered separation scores, group sizes and raw behaviour profiles, and repeated the stochastic methods with five seeds.
>
> K-Means with two clusters became the leading technical candidate. Three clusters remained a stable, more detailed alternative. I also documented why GMM's lowest BIC did not make it the strongest segmentation. I exported the assignments and evidence for Member 4 to validate against future behaviour and make the final business interpretation.”

## 27. 30-second explanation of my part

> “Member 2 prepared the historical RFM data, and I turned it into candidate customer groups. I compared a rule-based baseline, K-Means, Ward clustering and GMM using separation metrics, sizes, profiles and seed stability. K-Means with two clusters led technically, while three clusters remained a finer alternative. I handed the frozen assignments to Member 4 for future behavioural validation, final naming and business recommendations.”

## 28. Final memory summary

**Member 3 in one sentence**

I converted Member 2's historical model-ready RFM table into compared, stability-checked customer groupings and handed technical candidates to Member 4.

**5 things I must remember**

1. This is unsupervised clustering; no future-purchase prediction model was trained.
2. Member 2 prepared the RFM features; I audited and modelled their existing output.
3. The four approaches are RFM rules, K-Means, Ward and GMM.
4. K-Means k=2 leads technically; k=3 remains a stable finer candidate.
5. Member 4 validates future behaviour and owns the final business interpretation.

