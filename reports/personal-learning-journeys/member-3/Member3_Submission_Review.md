# Member 3 submission review and viva preparation

Reviewed on 9 October 2026. This is review guidance, not a replacement personal submission or a prediction of marks.

## Scope and verdict

Reviewed the supplied `ML_Group30_Final_Report.docx`, the two-page `IT3091_Assignment_Descriptor_V1.pdf`, the repository's `Pavesh_Personal_Learning_Journey.md`, and Member 3's modelling source, strategy, decision log and saved results.

The implementation meets the required baseline plus three alternative methods. The supporting technical documentation is stronger and more cautious than the condensed final report. Correct the report's factual overstatements, complete the AI declaration, and verify the final submission package before calling it ready.

The personal report reviewed is the repository Markdown version. A separate final Word/PDF personal report was not supplied, so its pagination and any later edits remain unverified.

## Assignment alignment

The descriptor, page 2, allocates 20 marks to model/data-mining strategy and comparison, 20 to evaluation/validation/critical judgement, and 10 to reproducibility/documentation/AI transparency. These are group criteria, not a separate 50-mark allocation to Member 3.

| Requirement | Evidence | Review |
|---|---|---|
| Sensible baseline and at least three alternatives | Rule-based RFM, K-Means, Ward agglomerative, GMM | Met. Different k values are configurations, not additional method families. |
| Justification by task, data, interpretability and constraints | Detailed strategy covers all four dimensions | Strong in repository; expand final report Table 13 with assumptions and constraints. |
| Appropriate comparison evidence | Three internal metrics, sizes, raw profiles, k=2–8, GMM diagnostics | Present; correct claims that all configurations outperform the baseline. |
| Validation logic and leakage prevention | Historical features strictly before 2011-09-09; five-seed repeats; frozen labels | Good Member 3 boundary. Future-period selection must be described as validation, not an untouched final test. |
| Decision log | Member 3 decisions included in Appendix A | Present; repair PCA rationale and clarify ownership of stability work. |
| Reproducibility and report agreement | Saved-output verifier passes | Member 3 evidence checked; integrated final notebook unavailable locally. |
| Honest AI disclosure | Personal report declares substantial Codex help | Group declaration remains incomplete. |
| Individual submission: one A4 Personal Learning Journey | Repository report contains approximately 639 whitespace-delimited words including headings/identity | Content appropriate; actual one-page layout unverified. |

## Corrections to the final report

### 1. Section 5.3: baseline superiority is overstated

The text says every ML method beats the baseline on all three scores. Table 14 itself includes GMM k=8, which is worse: silhouette 0.1597 versus 0.2220, Davies–Bouldin 1.9549 versus 1.4174, and Calinski–Harabasz 1265.4 versus 2097.5. Several other GMM configurations in the complete experiment file are also worse.

Suggested replacement: “The shortlisted K-Means k=2/k=3, Ward k=2 and GMM k=2 solutions outperform the rule-based baseline on all three internal metrics. This supports their geometric separation; it does not establish greater commercial value. GMM k=8 is retained separately as a density-fit diagnostic.”

### 2. Section 6.6: the justification for excluding two alternatives from future evaluation is weak

The report says Ward k=2 and GMM k=2 give nearly the same split as K-Means k=2. Comparing the actual exported assignments gives ARI 0.6231 and 0.6146 respectively. These scores show agreement but do not justify treating the partitions as interchangeable. ARI is not a percentage of matching customers.

The descriptor does not explicitly require a future test for every candidate, but omitting these candidates weakens the claim of a complete comparison beyond internal geometry. Prefer running the same future-outcome evaluation for all five practical candidates, excluding GMM k=8 from winner selection because it is diagnostic. Otherwise retain the limitation and remove the assertion that the omitted tests could not change the outcome.

### 3. Section 6.5: acknowledge selection on the future validation period

The report uses future behaviour to choose k=3. This can be a valid validation-based decision, but the same period is no longer an untouched final test of the selected model. Add that there is no additional independent temporal test. Do not confuse this selection limitation with contaminating the historical fitting features: they are different issues.

### 4. Table 13 and Appendix A: improve the technical descriptions

- Ward merges the pair of clusters giving the smallest increase in within-cluster sum of squares. “Joins the two closest” does not identify its actual merge criterion.
- GMM models a mixture of Gaussian distributions and gives posterior membership probabilities over its components. It is not limited to partial membership in two groups.
- M3-08 says PCA fitting would prevent mapping clusters to R/F/M. That is incorrect: customers can still be profiled in original units after PCA clustering. State the actual rationale: retain all three features to avoid discarding information; use two PCs only for display. The saved display retains about 92.98% of variance.
- M3-06 should distinguish Ward's Euclidean criterion from a practical choice to standardize variables; standardization is useful here, not a formal requirement that all inputs must already have unit variance.

These corrections agree with the version-matched [scikit-learn clustering documentation](https://scikit-learn.org/1.4/modules/clustering.html) and [PCA documentation](https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html).

### 5. Sections 5.1–5.2: make the baseline reproducible

The three learned methods fit the scaled matrix. The baseline assigns scores from original RFM values; its geometric metrics are then evaluated on the same scaled matrix as the learned methods. State that distinction explicitly.

Add the implemented rule: average ranks preserve ties; lower Recency is better, higher Frequency/Monetary is better; convert ranks to scores 1–5 and sum with equal weights. Total bands are 3–6, 7–9, 10–12 and 13–15. Unequal band sizes are expected. Equal values must not be split just to force equal-sized bins.

### 6. Section 5.2 and decision M3-05: k=2–8 does not guarantee actionable group sizes

Ward k=7 and k=8 each contain a group of only 23 customers. Replace the claim that the search avoids small groups with: “We used a bounded search from two to eight groups and inspected the resulting sizes; some finer configurations produced small groups.”

### 7. Table 16 and Section 6.3: qualify conclusions about truth and stability

Ward k=2 does not confirm that two genuine natural customer types exist. Say it provides another coarse partition with broadly similar profiles. Also replace the claim that k=3 gives the same memberships regardless of initialization with “highly consistent across the five tested seeds”; mean ARI 0.9989 is close to, but below, one.

Use “identical assignments across these five tested seeds” for ARI=1 results. Neither that result nor Ward's deterministic behaviour establishes robustness to resampling, different preprocessing or future periods.

### 8. Table 13: restore task/data/constraint justification

The strategy file already contains material to summarize:

- K-Means: compact groups in three scaled numeric dimensions; easy centroid/profile interpretation; compactness assumptions and residual outlier sensitivity.
- Ward: a nested variance-based hierarchy; feasible for 3,362 customers, but quadratic-scale work and irreversible greedy merges limit scaling.
- GMM: full covariance permits correlated, elliptical components and membership ambiguity; more parameters and discrete Frequency create density-fit risks.
- Baseline: transparent deterministic rules; equal-weight compensation and correlated F/M can obscure different behaviours.

Add the actual principal settings or a precise reference to them: K-Means n_init=20, max_iter=500; Ward Euclidean linkage; GMM full covariance, n_init=10, reg_covar=1e-6, max_iter=500; primary seed 42 and the five stability seeds. These settings already exist in code and the manifest.

### 9. Section 5.5: retain the good GMM reasoning, soften the certainty

The numerical diagnostic is supported: five k=8 components have minimum covariance eigenvalues near 1e-6 and contain a single Frequency value. Say this suggests density fitting to discrete Frequency structure, rather than proving there are no genuine customer types. The smallest AIC/BIC is within the tested GMM family/range; it is not a score comparison with K-Means or proof of global optimality.

### 10. Section 8.5: complete the AI declaration

Table 22 still has “Member 3” / “COMPLETE” in the modelling row. Replace these with the actual assistance and actual human verification. The personal report's declaration of substantial coding, review and documentation assistance is a suitable starting point.

The group report says no AI was used for Personal Learning Journeys and leaves “List only what is actually true.” This review is itself AI assistance in reviewing that report. Distinguish independent reflection from AI drafting/editing/review accurately; do not assert that the original personal text was AI-generated without evidence. Delete drafting prompts and complete the other placeholders before signing an accuracy declaration.

### 11. Reproducibility and integration need a final package check

The named `notebooks/final/final_integrated_notebook.ipynb` is absent from the current Member 3 checkout and from the locally available main tree. This does not establish that it is absent from teammates' computers or an external submission package. It prevents this review from verifying the group report's integrated-run and future-cleaning claims.

Obtain the actual final notebook, restart its kernel, execute it against the declared data, and compare outputs with the final report. In particular, verify the claimed duplicate-cleaning correction and updated future totals. Do not substitute a successful Member 3 verifier run for that full pipeline check.

### 12. Shared consistency issues affecting your viva

- The report says 39 decisions in Section 2.3 but 33 in Appendix A. Counting the listed non-empty IDs gives 39.
- Appendix A assigns the stability decision to M4-02, while your learning report and code include substantial Member 3 stability work. Clarify implementation versus interpretation/shared decision ownership with the group.
- Monetary is retained positive spend after excluding cancellation rows; it is not net revenue or profit. Use that qualification when explaining profiles or future outcomes.
- Full-year EDA informed preprocessing decisions. The historical fitting boundary is sound, but avoid claiming the future period was completely unseen by the entire project. Its full-year descriptive patterns had been explored.
- Section 5.4's 70 fitted configurations and 140 ARI pairs are supported. Explain that the 70 include primary seed-42 fits; each estimator call also performs its configured internal restarts.

## Personal Learning Journey review

The existing report is well aligned with the individual requirement: it identifies your responsibility, explains decisions and learning, gives specific challenges, describes collaboration, discloses AI assistance and proposes future improvements. Its main numbers and method claims match the checked Member 3 artifacts. I found no clear technical contradiction in that personal report.

Before submitting:

1. Export the actual personal document and check it is one A4 page. Approximately 639 words plus headings may be tight depending on formatting; no rendered version was checked. Shorten repeated context before reducing legibility.
2. Preserve the distinction between AI-assisted implementation and your personal understanding. Only claim checks you actually performed or can explain. The saved-output verifier was run by this review, which does not establish who ran earlier checks.
3. Add one truthful, specific personal observation about how your understanding changed, if you can do so within the page limit. Explain an actual misconception, what result changed it, and what you checked; do not invent an experience.
4. Optionally mention the final group outcome: your stage shortlisted k=2 and k=3, and the final report records k=3 after Member 4's evaluation. This is context, not a claim that you performed that evaluation.
5. Be prepared to explain every advanced term you retain: midrank scoring, ARI, density versus segmentation, covariance floor, fresh kernel and input hashes. Simplify wording that you cannot yet explain accurately.
6. Resolve the group AI-use and stability-attribution wording so both submissions describe the same division of work.

## Viva practice tied to your implementation

| Likely question | Points to understand and explain in your own words |
|---|---|
| What exactly was your contribution? | Consumed Member 2's historical RFM; checked it; implemented an AI-assisted baseline and three method families; compared k values, metrics, profiles and stability; exported assignments for Member 4. Distinguish implementation assistance from your own verification and decisions. |
| Why is this unsupervised? | No known correct customer-segment labels. Silhouette is not prediction accuracy; 0.4114 does not mean 41.14% accuracy. |
| Which columns entered the models? | Recency_scaled, Frequency_scaled and Monetary_scaled. The latter two are scaled log1p values. CustomerID only joins records; raw RFM explains profiles. |
| How does the baseline work? | Average ranks for ties; recency direction reversed; 1–5 scores; equal-weight sum; four fixed bands. With many one-order customers, forcing equal bins would arbitrarily separate identical values. |
| Why exactly these methods? | A transparent rule reference, compact centroid groups, a nested variance hierarchy, and an overlapping Gaussian density model. Explain each one's limitations as well as strengths. |
| Why k=2 through 8? | A practical bounded comparison of coarse/finer segmentation, not proof the true k lies there. Inspect sizes and profiles; high k can create small groups. |
| Why was k=2 your technical leader? | Highest silhouette and Calinski–Harabasz and lowest Davies–Bouldin in the primary comparison; large groups and consistent tested seed assignments. |
| Why does the final report choose k=3? | A finer business interpretation with three profiles; your technical shortlist enabled Member 4's validation-based decision. Internal quality is slightly lower. Future validation used for selection is not an independent final test. |
| Explain the three internal metrics. | Silhouette compares within-group distances with the nearest other group; DB summarizes each group's worst relative dispersion/separation; CH compares between-group and within-group dispersion. Higher silhouette/CH and lower DB are preferred. They evaluate geometry, not campaign profit. |
| Why ARI instead of comparing label numbers? | Numeric cluster labels can swap without changing membership. ARI compares partitions with chance adjustment. Five seeds yield 10 pairs; 2 methods × 7 k values × 5 seeds = 70 runs, and 14 configurations × 10 pairs = 140 comparisons. |
| Is seed 42 special? | It is a reproducible fixed choice, not the best seed discovered through selection. Internal restarts and comparisons across other seeds reduce reliance on one initialization. |
| Why reject GMM k=8 despite its AIC/BIC? | Better continuous density fit did not imply better customer segmentation. Its silhouette was 0.1597, mean ARI about 0.7864, and five components were narrow around discrete Frequency values. This does not disprove all GMM configurations. |
| What is the covariance floor? | A small positive diagonal regularization helps prevent singular covariance matrices. Near-floor eigenvalues indicate extremely narrow directions; explain the evidence without claiming covariance is zero. |
| How did you prevent temporal contamination? | Transactions strictly before 2011-09-09 create fitting features and scaling; freeze historical labels; later outcomes belong to validation. Explain the separate limitations of full-year EDA and validation-based selection. |
| Why use PCA only for display? | Two components provide a readable plot but drop some variance. Fitting used all three dimensions. PCA would not prevent profiling assigned customers in raw RFM units. |
| What have you not proved? | Natural customer types, causal benefit of marketing, net profit, robustness to resampling, optimality across all method settings, or performance in an independent later period. |
| What did AI do? | Substantial coding, review and documentation assistance. Identify the specific decisions and output checks you personally understand; do not claim independent authorship for generated work. |

## Checks actually performed in this review

- Ran `.venv/Scripts/python.exe members/member-3/notebooks/verify_outputs.py`: PASS.
- It checked recorded source/input/output hashes, six exported candidate metric/label sets, all 22 experiment/profile records, baseline tie/order/direction properties, 70 saved stochastic runs, 140 saved ARI pairs and the saved notebook's execution sequence/error absence.
- Independently computed cross-method ARI for Ward/GMM k=2 against K-Means k=2 and inspected the small Ward groups.
- Inspected report text and tables, the assignment requirements and the repository personal report.
- Did not rerun every model in fresh kernels, execute the missing final integrated notebook, inspect a rendered one-page personal report, or verify the learner's personal comprehension. These remain distinct from the checks above.

The original Word report, learning report and model files were not edited by this review.
