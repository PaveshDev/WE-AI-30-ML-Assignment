# Member 3 model / method comparison

Generated from executed code on the unchanged historical RFM table. Values in this
document come from the notebook's in-memory results; full precision is retained in
`../results/`. The strategy document explains parameter choices and the rubric mapping.

## Technical conclusion

**K-Means k=2 is the leading technical candidate; K-Means k=3 is a finer secondary
candidate for Member 4's behavioural validation.** Ward k=2 and GMM k=2 remain
cross-method comparators. The rule-based four-group baseline is retained. GMM k=8 is
exported explicitly as a density-criterion diagnostic, not the recommended segmentation.
No final business winner or final segment names have been assigned.

K-Means k=2 leads the primary experiments on silhouette (0.4114),
Davies-Bouldin (0.9045) and Calinski-Harabasz
(3136.79), with smallest/largest groups
1395/1967 and minimum seed-pair ARI 1.0000.
This agreement is strong internal evidence for a coarse segmentation, but it does not
establish whether two groups provide enough distinct decisions for the retailer.

## Handoff audit

- Rows and unique IDs: **3,362**; duplicate IDs: 0;
  duplicate full rows: 0; duplicate feature vectors:
  0 (equal behaviours would have been retained).
- Missing / infinite values: 0 / 0.
- Historical transaction metadata: 231,806 rows, dates
  2010-12-01 08:26:00 through 2011-09-08 19:58:00; reserved-period rows: 0.
- Existing transformations agree with Member 2's log1p/scaling formula within maximum
  absolute error 4.59e-09.
- One-order customers: 1,366
  (40.63%); raw Spearman(F,M):
  0.7842. Standardization does not remove that overlap.
- No downstream correction, filtering or preprocessing replacement was necessary.

| feature | dtype | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- |
| CustomerID | int64 | 12346.0000 | 13786.2500 | 15234.5000 | 16766.7500 | 18287.0000 |
| Recency | int64 | 0.0000 | 25.0000 | 73.0000 | 150.0000 | 281.0000 |
| Frequency | int64 | 1.0000 | 1.0000 | 2.0000 | 4.0000 | 131.0000 |
| Monetary | float64 | 2.9000 | 260.6575 | 555.0150 | 1363.2625 | 177729.6200 |
| Log_Frequency | float64 | 0.6931 | 0.6931 | 1.0986 | 1.6094 | 4.8828 |
| Log_Monetary | float64 | 1.3610 | 5.5670 | 6.3208 | 7.2184 | 12.0880 |
| Recency_scaled | float64 | -1.1792 | -0.8651 | -0.2621 | 0.7052 | 2.3508 |
| Frequency_scaled | float64 | -0.8688 | -0.8688 | -0.2217 | 0.5936 | 5.8180 |
| Monetary_scaled | float64 | -4.1265 | -0.6847 | -0.0679 | 0.6666 | 4.6514 |

Population standard deviations are one to numerical/CSV precision. The sample standard
deviation is slightly greater because of its n-1 denominator. Exact means, deviations,
dtypes, errors and dates are retained in `input_audit.json` and `input_summary.csv`.

## Candidate comparison

Higher silhouette / Calinski-Harabasz and lower Davies-Bouldin indicate stronger geometric
separation. ARI applies only to stochastic seed repeats, not deterministic methods.
The baseline's geometric metrics evaluate fixed rules, not learned clustering.

| method | k | silhouette | davies_bouldin | calinski_harabasz | smallest_cluster | largest_cluster | ari_mean | ari_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFM_Baseline | 4 | 0.2220 | 1.4174 | 2097.5180 | 719 | 935 | N/A | N/A |
| KMeans | 2 | 0.4114 | 0.9045 | 3136.7877 | 1395 | 1967 | 1.0000 | 1.0000 |
| KMeans | 3 | 0.3728 | 0.9158 | 3135.7482 | 866 | 1520 | 0.9989 | 0.9982 |
| Hierarchical | 2 | 0.3928 | 0.9468 | 2896.4257 | 1645 | 1717 | N/A | N/A |
| GMM | 2 | 0.3802 | 0.9724 | 2723.7079 | 1650 | 1712 | 1.0000 | 1.0000 |
| GMM | 8 | 0.1597 | 1.9549 | 1265.3859 | 95 | 772 | 0.7864 | 0.6517 |

| method | k | configuration | stability_note | interpretability | strengths | limitations | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RFM_Baseline | 4 | Fixed midrank score bands | Deterministic; seed ARI N/A | Transparent scores | Simple operational reference | Compensates across R/F/M; ties unbalance bins | Rule-based reference |
| KMeans | 2 | k-means++; n_init=20; Lloyd | Five-seed ARI below | Centroids and raw profiles | Compact groups; straightforward assignment | Spherical bias; residual extremes | Leading technical |
| KMeans | 3 | k-means++; n_init=20; Lloyd | Five-seed ARI below | Centroids and raw profiles | Compact groups; straightforward assignment | Spherical bias; residual extremes | Finer secondary |
| Hierarchical | 2 | Ward; Euclidean | Deterministic; seed ARI N/A | Merge tree and raw profiles | Nested partition; no random starts | Greedy merges; quadratic scaling | Cross-method comparator |
| GMM | 2 | Full covariance; n_init=10 | Five-seed ARI below | Posterior probabilities and profiles | Allows elliptical overlap | Density fit can follow discrete Frequency; local optima | Cross-method comparator |
| GMM | 8 | Full covariance; n_init=10 | Five-seed ARI below | Posterior probabilities and profiles | Allows elliptical overlap | Density fit can follow discrete Frequency; local optima | Density diagnostic only |

## Baseline evidence

Midrank quintile scores preserve all ties. Sum equally weighted scores and use fixed
total-score bands: Group 0 = 3–6; Group 1 = 7–9; Group 2 = 10–12; Group 3 = 13–15.
There is no score 1 for Frequency in this cohort: the tied one-order population has
its midrank in the second quintile. This is disclosed, not broken apart using IDs.
Scores can range from 1 to 5; the procedure does not guarantee every level is populated.

| feature | score | min | max | count |
| --- | --- | --- | --- | --- |
| Recency | 1 | 170.0000 | 281.0000 | 665 |
| Recency | 2 | 98.0000 | 169.0000 | 673 |
| Recency | 3 | 51.0000 | 97.0000 | 680 |
| Recency | 4 | 20.0000 | 50.0000 | 670 |
| Recency | 5 | 0.0000 | 18.0000 | 674 |
| Frequency | 2 | 1.0000 | 1.0000 | 1366 |
| Frequency | 3 | 2.0000 | 2.0000 | 668 |
| Frequency | 4 | 3.0000 | 4.0000 | 654 |
| Frequency | 5 | 5.0000 | 131.0000 | 674 |
| Monetary | 1 | 2.9000 | 213.5500 | 672 |
| Monetary | 2 | 213.7000 | 401.6000 | 673 |
| Monetary | 3 | 401.9000 | 765.5500 | 672 |
| Monetary | 4 | 765.6500 | 1651.2500 | 673 |
| Monetary | 5 | 1651.7500 | 177729.6200 | 672 |

The baseline has silhouette 0.2220, Davies-Bouldin 1.4174
and Calinski-Harabasz 2097.52. Its balanced groups and transparent
rules remain useful despite lower geometric separation. Correlated F and M influence
the total twice, while compensation can put different R/F/M patterns in one band.

| solution | cluster | customers | percentage | median_recency | median_frequency | median_monetary | mean_recency | mean_frequency | mean_monetary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFM_Baseline_Group | 0 | 935 | 27.81 | 182.00 | 1.00 | 190.55 | 182.45 | 1.05 | 220.71 |
| RFM_Baseline_Group | 1 | 905 | 26.92 | 88.00 | 2.00 | 409.63 | 95.15 | 1.66 | 606.16 |
| RFM_Baseline_Group | 2 | 803 | 23.88 | 46.00 | 3.00 | 905.60 | 55.34 | 3.13 | 1228.49 |
| RFM_Baseline_Group | 3 | 719 | 21.39 | 15.00 | 7.00 | 2433.22 | 20.08 | 9.48 | 5034.78 |

## Complete primary experiments

All ML primary runs use seed 42 where applicable, k=2..8, and the same three scaled
features. All full-data geometric metrics are retained; no configurations are omitted.
The sizes dictionaries map local numeric cluster IDs to customer counts.

### K-Means

| k | inertia | silhouette | davies_bouldin | calinski_harabasz | smallest_cluster | largest_cluster | sizes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 5216.2813 | 0.4114 | 0.9045 | 3136.7877 | 1395 | 1967 | {"0": 1967, "1": 1395} |
| 3 | 3517.8854 | 0.3728 | 0.9158 | 3135.7482 | 866 | 1520 | {"0": 866, "1": 976, "2": 1520} |
| 4 | 2714.5012 | 0.3460 | 0.9498 | 3039.6614 | 470 | 1112 | {"0": 1112, "1": 893, "2": 470, "3": 887} |
| 5 | 2346.0046 | 0.3239 | 0.9941 | 2768.8901 | 184 | 965 | {"0": 865, "1": 965, "2": 184, "3": 726, "4": 622} |
| 6 | 2030.8302 | 0.3148 | 0.9696 | 2662.2846 | 190 | 758 | {"0": 190, "1": 488, "2": 635, "3": 758, "4": 658, "5": 633} |
| 7 | 1826.6904 | 0.3010 | 0.9809 | 2528.2542 | 184 | 725 | {"0": 600, "1": 563, "2": 377, "3": 725, "4": 512, "5": 401, "6": 184} |
| 8 | 1672.9705 | 0.2812 | 1.0494 | 2409.5458 | 134 | 617 | {"0": 371, "1": 603, "2": 367, "3": 134, "4": 399, "5": 617, "6": 522, "7": 349} |

k=2 has the best separation metrics within this family. k=3 reduces inertia from
5216.28 to 3517.89 and retains substantial groups
({"0": 866, "1": 976, "2": 1520}), while silhouette decreases to 0.3728. Its
Calinski-Harabasz (3135.75) is close to k=2. The profiles below
show the additional differentiation; Member 4 must assess whether it matters in practice.
Increasing k beyond three reduces inertia mechanically but does not improve the reported
separation metrics. k=8 is also more initialization-sensitive (minimum ARI
0.7250). No elbow-only choice was made.

### Ward agglomerative

| k | silhouette | davies_bouldin | calinski_harabasz | smallest_cluster | largest_cluster | sizes |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.3928 | 0.9468 | 2896.4257 | 1645 | 1717 | {"0": 1717, "1": 1645} |
| 3 | 0.3389 | 0.9317 | 2565.9930 | 380 | 1645 | {"0": 1645, "1": 1337, "2": 380} |
| 4 | 0.3147 | 0.9432 | 2676.9050 | 380 | 1337 | {"0": 1337, "1": 939, "2": 380, "3": 706} |
| 5 | 0.2621 | 1.1377 | 2432.1274 | 380 | 939 | {"0": 939, "1": 628, "2": 380, "3": 706, "4": 709} |
| 6 | 0.2577 | 1.1683 | 2305.9385 | 251 | 709 | {"0": 380, "1": 628, "2": 688, "3": 706, "4": 709, "5": 251} |
| 7 | 0.2593 | 1.0806 | 2137.5331 | 23 | 709 | {"0": 688, "1": 628, "2": 357, "3": 706, "4": 709, "5": 251, "6": 23} |
| 8 | 0.2385 | 1.0777 | 2002.6363 | 23 | 709 | {"0": 628, "1": 709, "2": 357, "3": 706, "4": 447, "5": 251, "6": 23, "7": 241} |

Ward k=2 has the highest silhouette and Calinski-Harabasz in its family, and nearly
equal groups ({"0": 1717, "1": 1645}). Ward k=3 has slightly better Davies-Bouldin
(0.9317 versus 0.9468) but lower silhouette
(0.3389) and more unequal groups ({"0": 1645, "1": 1337, "2": 380}). k=2 is retained
as the stronger coarse comparator; the DB disagreement is not hidden. At k=7 and k=8
the smallest group has 23 customers; these
fine splits require more explanation without stronger overall separation. Ward is
deterministic for fixed input/order/linkage; no fabricated random-seed ARI is reported.

### Gaussian Mixture

| k | aic | bic | silhouette | davies_bouldin | calinski_harabasz | smallest_cluster | largest_cluster | sizes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 21957.1243 | 22073.4098 | 0.3802 | 0.9724 | 2723.7079 | 1650 | 1712 | {"0": 1650, "1": 1712} |
| 3 | 6139.1253 | 6316.6137 | 0.2105 | 1.3961 | 1758.0719 | 992 | 1366 | {"0": 992, "1": 1366, "2": 1004} |
| 4 | -1003.2557 | -764.5643 | 0.1347 | 1.6826 | 1389.0621 | 570 | 1366 | {"0": 758, "1": 668, "2": 1366, "3": 570} |
| 5 | -5132.6964 | -4832.8021 | 0.0973 | 2.0052 | 1181.8494 | 396 | 1366 | {"0": 668, "1": 524, "2": 408, "3": 396, "4": 1366} |
| 6 | -5341.8335 | -4980.7363 | 0.1021 | 1.8952 | 1026.9445 | 181 | 1366 | {"0": 447, "1": 1366, "2": 408, "3": 668, "4": 181, "5": 292} |
| 7 | -5671.3987 | -5249.0986 | 0.0856 | 1.7453 | 1083.7457 | 165 | 1201 | {"0": 165, "1": 460, "2": 1201, "3": 260, "4": 668, "5": 200, "6": 408} |
| 8 | -7830.1966 | -7346.6936 | 0.1597 | 1.9549 | 1265.3859 | 95 | 772 | {"0": 772, "1": 95, "2": 213, "3": 594, "4": 408, "5": 366, "6": 668, "7": 246} |

| k | converged | iterations | mean_confidence | low_confidence_count | low_confidence_pct | min_covariance_eigenvalue |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | True | 6.000000 | 0.887989 | 637.000000 | 18.947055 | 0.104234 |
| 3 | True | 9.000000 | 0.921541 | 486.000000 | 14.455681 | 0.000001 |
| 4 | True | 14.000000 | 0.945982 | 297.000000 | 8.834027 | 0.000001 |
| 5 | True | 27.000000 | 0.959935 | 231.000000 | 6.870910 | 0.000001 |
| 6 | True | 23.000000 | 0.945069 | 288.000000 | 8.566330 | 0.000001 |
| 7 | True | 30.000000 | 0.937728 | 296.000000 | 8.804283 | 0.000001 |
| 8 | True | 21.000000 | 0.919211 | 406.000000 | 12.076145 | 0.000001 |

GMM k=2 has the best geometric metrics within the family and balanced groups
({"0": 1650, "1": 1712}); 637 customers (18.95%)
have maximum posterior below 0.70. Its probabilities expose overlap, not calibrated
probabilities of buying again or of belonging to a proven true customer type.

Both AIC and BIC favour k=8 within the tested range (AIC -7830.20, BIC
-7346.69), but silhouette is only 0.1597 and minimum seed-pair ARI
is 0.6517. All k>=3 primary fits have at least one eigenvalue at the
1e-6 regularization floor. The table below verifies components restricted to discrete
Frequency values. This can strongly improve continuous density fit without producing
useful behavioural groups. Higher posterior confidence alone does not resolve it.
Covariance/regularization sensitivity was not exhaustively searched; the result must not
be described as globally optimal GMM selection or a failure of all mixture models.

| k | cluster | customers | frequency_unique | frequency_min | frequency_max | min_covariance_eigenvalue |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 | 1650 | 46 | 1 | 131 | 0.167358 |
| 2 | 1 | 1712 | 6 | 1 | 6 | 0.104234 |
| 8 | 0 | 772 | 1 | 1 | 1 | 0.000001 |
| 8 | 1 | 95 | 7 | 5 | 34 | 0.089750 |
| 8 | 2 | 213 | 41 | 5 | 131 | 0.007180 |
| 8 | 3 | 594 | 1 | 1 | 1 | 0.000001 |
| 8 | 4 | 408 | 1 | 3 | 3 | 0.000001 |
| 8 | 5 | 366 | 14 | 5 | 24 | 0.043285 |
| 8 | 6 | 668 | 1 | 2 | 2 | 0.000001 |
| 8 | 7 | 246 | 1 | 4 | 4 | 0.000001 |

## Stability: all searched stochastic configurations

Five seeds (42, 7, 21, 99, 123), fixed parameters, ten pairwise ARIs per configuration.
The 70 stochastic fits include the primary seed-42 fits once.
All completed without convergence warnings. Hierarchical/baseline seed tests are not
applicable. ARI is label-permutation invariant; the exported IDs remain primary seed 42.

| method | k | ari_mean | ari_min | ari_max | ari_std | pairs |
| --- | --- | --- | --- | --- | --- | --- |
| GMM | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 10 |
| GMM | 3 | 0.9995 | 0.9992 | 1.0000 | 0.0004 | 10 |
| GMM | 4 | 0.9987 | 0.9977 | 1.0000 | 0.0010 | 10 |
| GMM | 5 | 0.9607 | 0.9173 | 1.0000 | 0.0319 | 10 |
| GMM | 6 | 0.9952 | 0.9903 | 0.9984 | 0.0028 | 10 |
| GMM | 7 | 0.9962 | 0.9920 | 1.0000 | 0.0025 | 10 |
| GMM | 8 | 0.7864 | 0.6517 | 0.9978 | 0.1250 | 10 |
| KMeans | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 10 |
| KMeans | 3 | 0.9989 | 0.9982 | 1.0000 | 0.0010 | 10 |
| KMeans | 4 | 0.9927 | 0.9834 | 0.9991 | 0.0052 | 10 |
| KMeans | 5 | 0.9955 | 0.9905 | 0.9992 | 0.0032 | 10 |
| KMeans | 6 | 0.9749 | 0.9581 | 0.9992 | 0.0138 | 10 |
| KMeans | 7 | 0.9941 | 0.9871 | 1.0000 | 0.0048 | 10 |
| KMeans | 8 | 0.8688 | 0.7250 | 0.9795 | 0.0986 | 10 |

| method | k | silhouette_min | silhouette_max | inertia_min | inertia_max | bic_min | bic_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | 2 | 0.3802 | 0.3802 | N/A | N/A | 22073.4098 | 22073.4098 |
| GMM | 3 | 0.2105 | 0.2106 | N/A | N/A | 6316.6137 | 6316.6347 |
| GMM | 4 | 0.1347 | 0.1349 | N/A | N/A | -764.5643 | -764.3967 |
| GMM | 5 | 0.0887 | 0.1008 | N/A | N/A | -4835.4102 | -1016.0492 |
| GMM | 6 | 0.1020 | 0.1042 | N/A | N/A | -4982.4188 | -4976.1631 |
| GMM | 7 | 0.0843 | 0.0868 | N/A | N/A | -5249.3761 | -5248.0093 |
| GMM | 8 | 0.0761 | 0.1610 | N/A | N/A | -7542.4608 | -5174.8488 |
| KMeans | 2 | 0.4114 | 0.4114 | 5216.2640 | 5216.2813 | N/A | N/A |
| KMeans | 3 | 0.3728 | 0.3728 | 3517.8699 | 3517.8854 | N/A | N/A |
| KMeans | 4 | 0.3460 | 0.3464 | 2714.5012 | 2714.5615 | N/A | N/A |
| KMeans | 5 | 0.3231 | 0.3239 | 2345.9632 | 2346.0272 | N/A | N/A |
| KMeans | 6 | 0.3148 | 0.3163 | 2030.8150 | 2030.9212 | N/A | N/A |
| KMeans | 7 | 0.3007 | 0.3011 | 1826.6904 | 1826.8445 | N/A | N/A |
| KMeans | 8 | 0.2789 | 0.2863 | 1672.8505 | 1673.4912 | N/A | N/A |

ARI summarizes repeatability under initialization, not robustness to resampling,
outliers, different scalings or future periods. A stable algorithm can reproduce an
unhelpful segmentation. Per-run results and all pairs remain in the accompanying CSVs.

## Raw profiles for candidates and density diagnostic

All figures are in original units (Recency days, Frequency invoices, Monetary GBP).
Medians are emphasized for skew; means retain evidence of concentration. Cluster IDs
are neutral and local to each solution, not matched across methods. The machine-readable
profile file also covers every other searched configuration.

| solution | cluster | customers | percentage | median_recency | median_frequency | median_monetary | mean_recency | mean_frequency | mean_monetary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFM_Baseline_Group | 0 | 935 | 27.81 | 182.00 | 1.00 | 190.55 | 182.45 | 1.05 | 220.71 |
| RFM_Baseline_Group | 1 | 905 | 26.92 | 88.00 | 2.00 | 409.63 | 95.15 | 1.66 | 606.16 |
| RFM_Baseline_Group | 2 | 803 | 23.88 | 46.00 | 3.00 | 905.60 | 55.34 | 3.13 | 1228.49 |
| RFM_Baseline_Group | 3 | 719 | 21.39 | 15.00 | 7.00 | 2433.22 | 20.08 | 9.48 | 5034.78 |
| KMeans_k2 | 0 | 1967 | 58.51 | 125.00 | 1.00 | 304.10 | 133.93 | 1.39 | 385.48 |
| KMeans_k2 | 1 | 1395 | 41.49 | 27.00 | 4.00 | 1561.06 | 37.38 | 6.51 | 3299.77 |
| KMeans_k3 | 0 | 866 | 25.76 | 22.00 | 6.00 | 2284.95 | 33.30 | 8.75 | 4720.23 |
| KMeans_k3 | 1 | 976 | 29.03 | 192.00 | 1.00 | 246.27 | 198.57 | 1.24 | 340.63 |
| KMeans_k3 | 2 | 1520 | 45.21 | 58.00 | 2.00 | 498.48 | 61.14 | 1.99 | 619.25 |
| Hierarchical_k2 | 0 | 1717 | 51.07 | 32.00 | 4.00 | 1228.19 | 41.12 | 5.63 | 2780.91 |
| Hierarchical_k2 | 1 | 1645 | 48.93 | 150.00 | 1.00 | 258.28 | 148.92 | 1.31 | 356.60 |
| GMM_k2 | 0 | 1650 | 49.08 | 29.00 | 4.00 | 1193.72 | 36.99 | 5.78 | 2699.24 |
| GMM_k2 | 1 | 1712 | 50.92 | 147.00 | 1.00 | 275.11 | 148.68 | 1.33 | 530.19 |
| GMM_k8_Diagnostic | 0 | 772 | 22.96 | 84.00 | 1.00 | 262.36 | 81.99 | 1.00 | 488.67 |
| GMM_k8_Diagnostic | 1 | 95 | 2.83 | 92.00 | 6.00 | 1495.79 | 103.18 | 6.29 | 1734.96 |
| GMM_k8_Diagnostic | 2 | 213 | 6.34 | 6.00 | 13.00 | 4685.17 | 7.16 | 17.55 | 10658.97 |
| GMM_k8_Diagnostic | 3 | 594 | 17.67 | 218.00 | 1.00 | 224.43 | 220.94 | 1.00 | 288.66 |
| GMM_k8_Diagnostic | 4 | 408 | 12.14 | 53.00 | 3.00 | 866.13 | 65.47 | 3.00 | 1267.54 |
| GMM_k8_Diagnostic | 5 | 366 | 10.89 | 24.00 | 6.00 | 2144.55 | 26.01 | 7.02 | 2879.16 |
| GMM_k8_Diagnostic | 6 | 668 | 19.87 | 79.00 | 2.00 | 539.47 | 89.68 | 2.00 | 693.44 |
| GMM_k8_Diagnostic | 7 | 246 | 7.32 | 45.00 | 4.00 | 1204.19 | 55.22 | 4.00 | 1395.86 |

## Visual evidence

- [K-Means search](../figures/kmeans_search.png)
- [Truncated Ward tree](../figures/ward_dendrogram.png)
- [GMM AIC/BIC](../figures/gmm_information_criteria.png)
- [Candidate PCA display](../figures/candidate_pca.png)
- [Candidate group sizes](../figures/candidate_sizes.png)

PC1 explains 73.12% and PC2 19.86%
(combined 92.98%). **PCA was not used for model fitting or metrics.**
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
6. **Computational constraints:** 3,362 rows and three features permit
   full-cohort fits and silhouette. A dense n-by-n float64 distance matrix would occupy
   approximately 86.2 MiB; hierarchical/pairwise work has
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
