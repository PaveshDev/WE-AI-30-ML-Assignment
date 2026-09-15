# Member 3 modelling decisions — integration draft

The shared `docs/decision-log/` contains only `.gitkeep`. This draft preserves
existing work and is intended for the group lead to integrate after review.
Numerical evidence is generated from the executed modelling notebook.

| Decision | Options considered | Selected option | Reason | Supporting evidence |
| --- | --- | --- | --- | --- |
| M3-01: Feature matrix | Raw RFM; three scaled RFM; extra features/PCA | Use existing three scaled RFM; exclude ID; retain raw profiles | Respect handoff; comparable numeric distance dimensions; no future data | Audit: 3362 complete customers; transformation checks passed |
| M3-02: Baseline | Quantile edges; row-order tie breaking; midrank scores | Tie-preserving midrank 1–5 scores; sum; four fixed bands | Same values get same scores; small practical baseline without tuning to ML | 1366 one-order ties; groups {"0": 935, "1": 905, "2": 803, "3": 719} |
| M3-03: Search range | Single k; 2..8; much larger search | All ML methods use 2..8 | Compare coarse and finer groups within bounded assignment scope | Complete experiments.csv; no hidden configurations or future-outcome selection |
| M3-04: K-Means candidates | 2..8 clusters | k=2 leading; k=3 finer secondary | k=2 leads geometry; k=3 adds profile detail with strong repeatability | k2 silhouette 0.4114; k3 0.3728; ARI minima 1.0000/0.9982 |
| M3-05: Hierarchy | Ward; alternative linkages; counts 2..8 | Ward Euclidean, retain k=2 comparator | Numeric scaled data suits variance minimization; k2 best silhouette/CH and balanced; acknowledge k3 DB | k2 sizes {"0": 1717, "1": 1645}; DB k2 0.9468, k3 0.9317 |
| M3-06: GMM | Full covariance 2..8; select by geometry or AIC/BIC | k=2 comparator; k=8 separately labelled density diagnostic | Retain disagreement; do not equate best continuous density fit with best customer groups | k8 BIC -7346.69, silhouette 0.1597, min ARI 0.6517; covariance floor diagnostics |
| M3-07: Stability | Compare numeric IDs; pairwise ARI; resampling/temporal checks | Five seeds; all ten label-invariant ARI pairs for every KMeans/GMM k | Measure initialization sensitivity of fixed multistart procedure; temporal work belongs to M4 | 70 stochastic runs; deterministic seed ARI marked N/A |
| M3-08: PCA and display | Fit models on PCA; full 3D features; display only PCA | Full 3D models; two-PC display; truncated actual Ward tree | Avoid dropping model information; keep hierarchy legible | PC1/PC2 explain 73.12%/19.86% |
| M3-09: Business decision boundary | Choose final winner now; technical shortlist then future validation | Technical shortlist and frozen assignments to M4 | Internal metrics cannot establish business usefulness or future behaviour | Candidate profiles, assignments and member3_handoff.md; no future rows fitted |
| M3-10: Upstream issues | Silently repair/cap; replace preprocessing; preserve and document | No downstream data adjustment; record AND/OR wording, gross-spend and manual-line qualifications | Handoff is numerically usable; business accounting choices need explicit group ownership | M2 code uses OR; log says AND; M1 Q8 sale/reversal evidence; original hashes unchanged |

Group discussion: agree the validation endpoint, qualify retained positive spend
and manual transactions, reconcile the cancellation-log wording, and check the
original submission/descriptor before final reporting. These do not prevent the
completed historical modelling handoff. Final segment naming belongs to Member 4.
