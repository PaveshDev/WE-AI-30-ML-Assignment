# %% [markdown]
"""
# Member 3: ML strategy, modelling and comparison

**IT3091 | WE-AI-30 | Guided Data Track | Retail & E-commerce**

## 1. Objective and modelling scope

Group identified customers by historical purchasing behaviour using the promised
rule-based RFM baseline, K-Means, Ward hierarchical clustering and Gaussian Mixture.
This is unsupervised segmentation: there is no prediction target or purchase classifier.
Member 2 has already cleaned, engineered, transformed and scaled the input.
Member 4 owns future behavioural validation, final business names and recommendations.

All fitting uses the existing three scaled RFM columns. Transactions on or after
**2011-09-09 00:00:00** are reserved. No Member 1 EDA or Member 2 preprocessing is rerun.
The audit reads only customer/date metadata from the already cleaned historical file.

This notebook is generated from the adjacent percent-cell Python source. Run
`python members/member-3/notebooks/run_notebook.py` from the project root to regenerate
and execute it in a fresh kernel. See the Member 3 README for dependencies.
"""
# %%
import os
# Fix numerical thread counts before importing numpy/sklearn in a fresh process.
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

from pathlib import Path
import sys
import json
import platform
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "data/processed/rfm_table.csv").exists() and (p / "members").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Run from the cloned repository or a directory inside it")
WORK = ROOT / "members/member-3"
sys.path.insert(0, str(WORK / "notebooks"))
from modelling_helpers import (
    FEATURES, RAW, SEEDS, COUNTS, fingerprint, audit_input, rfm_baseline,
    cluster_metrics, experiment, repeat_experiments, profiles, markdown_table,
)

DOCS, FIGURES, RESULTS = WORK / "docs", WORK / "figures", WORK / "results"
for folder in [DOCS, FIGURES, RESULTS]:
    folder.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

def save_figure(fig, name):
    fig.tight_layout()
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "Member 3 modelling"})
    plt.close(fig)
    display(Image(filename=str(path)))

def save_table(frame, name):
    frame.to_csv(RESULTS / f"{name}.csv", index=False, float_format="%.12g")

# %% [markdown]
"""
## 2. Load model-ready RFM and verify the Member 2 handoff

The input is read unchanged. We check schema, IDs, missing/infinite values, raw domains,
log1p relationships, and the existing scaler's algebra to CSV precision (1e-7 absolute
tolerance). That algebra is an audit only; it never replaces model features or fits a scaler.
Population standard deviation (`ddof=0`) should be one; the sample standard deviation
is slightly larger. Repeated feature vectors are valid distinct customers and are retained.

The transaction date audit and Member 2's `< 2011-09-09` source filter support historical
provenance. Hashes bind this run to the exact input artifacts. The future raw file is not
read by the modelling workflow. Full-data EDA statistics are context, not modelling input.
"""
# %%
RFM_PATH = ROOT / "data/processed/rfm_table.csv"
TRANSACTIONS_PATH = ROOT / "data/interim/cleaned_transactions.csv"
input_hashes = {p.relative_to(ROOT).as_posix(): fingerprint(p)
                for p in [RFM_PATH, TRANSACTIONS_PATH]}
rfm = pd.read_csv(RFM_PATH)
audit = audit_input(rfm, TRANSACTIONS_PATH)
(RESULTS / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
input_summary = rfm.describe().T.reset_index(names="feature")
input_summary["dtype"] = input_summary.feature.map(rfm.dtypes.astype(str))
save_table(input_summary, "input_summary")
display(Markdown(f"**Handoff passed:** {audit['rows']:,} unique customer rows; "
                 f"no missing/infinite values or duplicate IDs. "
                 f"Latest historical transaction: {audit['date_max']}."))
display(input_summary)
display(pd.DataFrame({"mean": rfm[FEATURES].mean(),
                      "population_std": rfm[FEATURES].std(ddof=0),
                      "sample_std": rfm[FEATURES].std(ddof=1)}))
print("Input fingerprints:", json.dumps(input_hashes, indent=2))

# %% [markdown]
"""
## 3. Define the feature matrix

Only `Recency_scaled`, `Frequency_scaled`, `Monetary_scaled` enter the models and
distance metrics. The last two represent scaled **log1p** Frequency and Monetary.
CustomerID is only a join key. Raw RFM remains available for profiles in days, invoice
counts and GBP. Standardization equalizes marginal variance but does not remove the
Frequency/Monetary correlation or guarantee normally distributed features.
"""
# %%
X = rfm[FEATURES].to_numpy(dtype=float, copy=True)
assert X.shape == (len(rfm), 3)
X.setflags(write=False)
print("Feature matrix:", X.shape, FEATURES)
print("Historical Spearman(F, M):", audit["frequency_monetary_spearman"])

# %% [markdown]
"""
## 4. Rule-based RFM baseline

Rank each variable worst-to-best (Recency descending, Frequency/Monetary ascending),
using the average rank for ties. Let `p = (average_rank - 0.5) / n`, then
`score = floor(5*p) + 1`, clipped to 1..5. This is an empirical midrank quintile rule:
ties never split by row order or CustomerID. Large ties can leave some scores unused;
we do not force five equal-sized bins or discard duplicate quantile edges.

Combine scores with equal weights: `total = R + F + M` (3..15).
Fixed neutral groups are 0: total 3..6, 1: 7..9, 2: 10..12, 3: 13..15.
Equivalently, mean scores are <=2, >2..3, >3..4, and >4. These thresholds are fixed
before inspecting ML results. They provide four explainable groups without 125 tiny
RFM combinations. Compensation between dimensions and correlated F/M are limitations.

Silhouette, Davies-Bouldin and Calinski-Harabasz evaluate these rule-based labels
on the exact same scaled matrix. They do not turn the baseline into an ML model.
"""
# %%
baseline = rfm_baseline(rfm)
baseline_row = {"method": "RFM_Baseline", "k": baseline.RFM_Baseline_Group.nunique(),
                "seed": np.nan, **cluster_metrics(X, baseline.RFM_Baseline_Group)}
baseline_profiles = profiles(rfm, baseline.RFM_Baseline_Group, "RFM_Baseline_Group")
score_ranges = pd.concat([
    rfm.assign(score=baseline[f"{short}_Score"]).groupby("score")[raw]
       .agg(["min", "max", "count"]).reset_index().assign(feature=raw)
    for raw, short in zip(RAW, ["R", "F", "M"])
], ignore_index=True)[["feature", "score", "min", "max", "count"]]
save_table(score_ranges, "baseline_score_ranges")
display(score_ranges)
display(baseline_profiles)

# %% [markdown]
"""
## 5. K-Means experiments

Search k=2..8 with seed 42, k-means++ initialization, 20 restarts, Lloyd's algorithm,
500 maximum iterations and tolerance 1e-4. Explicit restarts avoid relying on changing
library defaults. Retain inertia, full-data silhouette, Davies-Bouldin (lower is better),
Calinski-Harabasz (higher is better) and all cluster sizes. Full-data silhouette is feasible
for this cohort; no changing random metric sample obscures comparisons.

An elbow alone cannot establish k. We examine separation, stability, raw profiles and
group sizes together. No minimum group threshold is imposed as a hidden exclusion rule.
"""
# %%
primary = {}
for k in COUNTS:
    primary[("KMeans", k)] = experiment("KMeans", k, X)
kmeans_results = pd.DataFrame([primary[("KMeans", k)][0] for k in COUNTS])
display(kmeans_results)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
for ax, metric, title in zip(axes, ["inertia", "silhouette", "davies_bouldin"],
                            ["Inertia (lower)", "Silhouette (higher)", "Davies-Bouldin (lower)"]):
    ax.plot(kmeans_results.k, kmeans_results[metric], marker="o", color="#356b8c")
    ax.set(xlabel="Number of clusters", ylabel=title, xticks=list(COUNTS))
fig.suptitle("K-Means: historical scaled RFM")
save_figure(fig, "kmeans_search")

# %% [markdown]
"""
## 6. Hierarchical / Agglomerative experiments

Ward linkage minimizes increases in within-group squared Euclidean dispersion.
Search k=2..8 on all customers and the same three features. No random initialization
is involved for a fixed dataset/order, distance and linkage; seed repetitions are not
meaningful. Determinism does not prove robustness to changed samples or time windows.

The dendrogram is built from the fitted sklearn model's actual merge tree, using SciPy
only to display the final 20 merged branches. Parentheses indicate branch populations,
not individual customer names. There is no dendrogram-only sampling or different fit.
"""
# %%
for k in COUNTS:
    primary[("Hierarchical", k)] = experiment("Hierarchical", k, X)
hierarchical_results = pd.DataFrame([primary[("Hierarchical", k)][0] for k in COUNTS])
display(hierarchical_results)
tree = primary[("Hierarchical", 2)][2]
counts = np.zeros(len(tree.children_))
for i, children in enumerate(tree.children_):
    counts[i] = sum(1 if child < len(X) else counts[child - len(X)] for child in children)
linkage_matrix = np.column_stack([tree.children_, tree.distances_, counts]).astype(float)
fig, ax = plt.subplots(figsize=(11, 4.5))
dendrogram(linkage_matrix, truncate_mode="lastp", p=20, show_leaf_counts=True,
           leaf_rotation=45, leaf_font_size=9, color_threshold=0, ax=ax)
ax.set(title="Ward hierarchy: last 20 branches (all customers fitted)",
       xlabel="Merged branch (parentheses show customer count)", ylabel="Ward linkage distance")
save_figure(fig, "ward_dendrogram")

# %% [markdown]
"""
## 7. Gaussian Mixture experiments

Search 2..8 full-covariance Gaussian components with seed 42, 10 initializations,
500 maximum EM iterations, tolerance 1e-3 and diagonal regularization 1e-6.
Full covariance can model correlated, elliptical groups; posterior probabilities
describe overlap. Labels are maximum-posterior assignments on the same three features.

AIC and BIC (lower preferred within these GMM fits) assess penalized in-sample density
fit. They are not interchangeable with geometric separation or future validation.
We retain covariance minimum eigenvalues to detect components compressed against the
regularization floor, particularly because Frequency has many ties. We report maximum
posterior <0.70 as a descriptive ambiguity threshold, not a ground-truth error rate.
Convergence warnings fail the run rather than disappearing silently.
"""
# %%
for k in COUNTS:
    primary[("GMM", k)] = experiment("GMM", k, X)
gmm_results = pd.DataFrame([primary[("GMM", k)][0] for k in COUNTS])
display(gmm_results)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, metric in zip(axes, ["aic", "bic"]):
    ax.plot(gmm_results.k, gmm_results[metric], marker="o", color="#35795c")
    ax.set(xlabel="Number of components", ylabel=metric.upper() + " (lower)", xticks=list(COUNTS))
fig.suptitle("GMM density criteria: compare alongside geometry and profiles")
save_figure(fig, "gmm_information_criteria")

# %% [markdown]
"""
## 8. Stability analysis

Repeat every stochastic configuration at seeds 42, 7, 21, 99, 123, keeping all other
parameters fixed. This assesses the actual multistart procedure, not single-start runs.
Report all ten pairwise Adjusted Rand Index (ARI) values per configuration. ARI is
invariant to label permutation: 1 means the same partition; near 0 means chance-level
agreement under its adjustment. Numeric cluster IDs need not match between runs.

Also retain per-seed metrics and objective values. Reusing the primary seed-42 run
gives five distinct seeds, not six trials. These are initialization checks on the same
customers, not confidence intervals, resampling tests or three-month temporal validation.
"""
# %%
seed_runs, stability_pairs, stability_summary = repeat_experiments(X, primary)
save_table(seed_runs, "stability_runs")
save_table(stability_pairs, "stability_pairs")
save_table(stability_summary, "stability_summary")
experiments = pd.DataFrame([baseline_row] + [result[0] for result in primary.values()])
experiments = experiments.merge(stability_summary, on=["method", "k"], how="left")
save_table(experiments, "experiments")
display(stability_summary)
print("Primary fits and stability complete; results saved under members/member-3/results")

# %% [markdown]
"""
## 9. Cross-method comparison and candidate solutions

The executed search supports **K-Means k=2 as the leading technical candidate**:
it leads the primary search on silhouette, Davies-Bouldin and Calinski-Harabasz,
with stable and substantial groups. Retain **K-Means k=3** as a finer alternative;
raw profiles below must show what the additional group contributes. Ward k=2 and
GMM k=2 are cross-method comparators. None is a final business winner.

Retain GMM k=8 explicitly as a **density-criterion diagnostic**, because AIC/BIC favour
it within the tested range. Its lower geometric separation, covariance-floor behaviour
and initialization sensitivity prevent declaring it the best customer segmentation.
The endpoint optimum does not prove that eight components is a global density optimum.

The table includes strength and limitation notes, with deterministic stability marked
not applicable. See the generated comparison document for all primary configurations,
the complete stability summary, and the seven decision dimensions used in the discussion.
"""
# %%
# These choices were made after reviewing the complete executed 2..8 search.
# Guards below prevent stale recommendations if input or library behaviour changes.
leading = experiments[(experiments.method == "KMeans") & (experiments.k == 2)].iloc[0]
assert np.isclose(leading.silhouette, experiments.silhouette.max())
assert np.isclose(leading.davies_bouldin, experiments.davies_bouldin.min())
assert np.isclose(leading.calinski_harabasz, experiments.calinski_harabasz.max())
assert int(gmm_results.loc[gmm_results.bic.idxmin(), "k"]) == 8
assert int(gmm_results.loc[gmm_results.aic.idxmin(), "k"]) == 8
selected = [("KMeans", 2), ("KMeans", 3), ("Hierarchical", 2), ("GMM", 2), ("GMM", 8)]
names = {(method, k): f"{method}_k{k}" for method, k in selected}
names[("GMM", 8)] = "GMM_k8_Diagnostic"
candidate_labels = {"RFM_Baseline_Group": baseline.RFM_Baseline_Group.to_numpy()}
candidate_labels.update({names[key]: primary[key][1] for key in selected})
notes = {
    "RFM_Baseline": ("Fixed midrank score bands", "Deterministic; seed ARI N/A",
                     "Transparent scores", "Simple operational reference", "Compensates across R/F/M; ties unbalance bins"),
    "KMeans": ("k-means++; n_init=20; Lloyd", "Five-seed ARI below",
               "Centroids and raw profiles", "Compact groups; straightforward assignment", "Spherical bias; residual extremes"),
    "Hierarchical": ("Ward; Euclidean", "Deterministic; seed ARI N/A",
                     "Merge tree and raw profiles", "Nested partition; no random starts", "Greedy merges; quadratic scaling"),
    "GMM": ("Full covariance; n_init=10", "Five-seed ARI below",
            "Posterior probabilities and profiles", "Allows elliptical overlap", "Density fit can follow discrete Frequency; local optima"),
}
comparison = experiments.loc[
    [(row.method, row.k) in selected or row.method == "RFM_Baseline"
     for row in experiments.itertuples()]
].copy()
for i, field in enumerate(["configuration", "stability_note", "interpretability", "strengths", "limitations"]):
    comparison[field] = comparison.method.map(lambda method: notes[method][i])
comparison["role"] = ["Rule-based reference", "Leading technical", "Finer secondary",
                      "Cross-method comparator", "Cross-method comparator", "Density diagnostic only"]
save_table(comparison, "candidate_comparison")
display(comparison[["method", "k", "silhouette", "davies_bouldin", "calinski_harabasz",
                    "smallest_cluster", "largest_cluster", "ari_mean", "ari_min", "role"]])

# %% [markdown]
"""
## 10. Raw RFM profiles and GMM diagnostics

Every searched configuration is profiled in original units; shortlisted profiles are
displayed here. Counts and percentages describe customers, not transaction rows.
Means accompany medians to show skew. All cluster numbers are neutral and local to
their configuration. Member 4 will assign final segment names after behavioural validation.

The GMM diagnostic table records each component's number and range of observed raw
Frequency values and minimum covariance eigenvalue. A high posterior can reflect a
narrow component fitted to a repeated discrete value; it is not proof of a real customer type.
"""
# %%
all_profiles = pd.concat([baseline_profiles] + [
    profiles(rfm, result[1], names.get(key, f"{key[0]}_k{key[1]}"))
    for key, result in primary.items()
], ignore_index=True)
save_table(all_profiles, "cluster_profiles")
candidate_profiles = all_profiles[all_profiles.solution.isin(candidate_labels)].copy()
display(candidate_profiles)
gmm_diagnostics = []
for k in COUNTS:
    _, labels, model = primary[("GMM", k)]
    for cluster in range(k):
        frequency = rfm.loc[labels == cluster, "Frequency"]
        gmm_diagnostics.append({"k": k, "cluster": cluster, "customers": len(frequency),
                                "frequency_unique": frequency.nunique(),
                                "frequency_min": frequency.min(), "frequency_max": frequency.max(),
                                "min_covariance_eigenvalue": np.linalg.eigvalsh(model.covariances_[cluster]).min()})
gmm_diagnostics = pd.DataFrame(gmm_diagnostics)
save_table(gmm_diagnostics, "gmm_component_diagnostics")
display(gmm_diagnostics[gmm_diagnostics.k.isin([2, 8])])

# %% [markdown]
"""
## 11. Visualise candidate geometry and group sizes

PCA is fitted only to show the three scaled features in two dimensions. Every model
and metric above used all three scaled RFM features. Axis labels report explained
variance. Colour/cluster number is local to each panel; matching colours across methods
do not imply matching customer sets. Size charts include the rule baseline and GMM
density diagnostic so those alternatives remain visible.
"""
# %%
pca = PCA(n_components=2, svd_solver="full")
coordinates = pca.fit_transform(X)
pca_variance = pca.explained_variance_ratio_
display(Markdown(f"PC1: **{100*pca_variance[0]:.2f}%**, PC2: **{100*pca_variance[1]:.2f}%**; "
                 f"combined **{100*pca_variance.sum():.2f}%** of variance (visualisation only)."))
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, name in zip(axes.flat, ["KMeans_k2", "KMeans_k3", "Hierarchical_k2", "GMM_k2"]):
    labels = candidate_labels[name]
    for cluster in np.unique(labels):
        points = coordinates[labels == cluster]
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=.4,
                   color=plt.get_cmap("tab10")(int(cluster)), label=f"Cluster {cluster}", rasterized=True)
    ax.set(title=name, xlabel=f"PC1 ({100*pca_variance[0]:.1f}%)",
           ylabel=f"PC2 ({100*pca_variance[1]:.1f}%)")
    ax.legend(markerscale=2, fontsize=8)
fig.suptitle("Historical customer groups: PCA display only")
save_figure(fig, "candidate_pca")
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
for ax, (name, labels) in zip(axes.flat, candidate_labels.items()):
    sizes = pd.Series(labels).value_counts().sort_index()
    ax.bar(sizes.index, sizes, color="#356b8c")
    ax.set(title=name, xlabel="Neutral cluster/group ID", ylabel="Customers",
           xticks=list(sizes.index), ylim=(0, len(rfm)*.67))
    for cluster, size in sizes.items():
        ax.text(cluster, size+25, f"{size}\n{100*size/len(rfm):.1f}%", ha="center", fontsize=8)
save_figure(fig, "candidate_sizes")

# %% [markdown]
"""
## 12. Export frozen historical assignments

One row per original CustomerID, retaining raw RFM and clearly named candidate columns.
GMM k=8 is explicitly marked diagnostic. Posterior maxima for GMM k=2 and k=8 support
ambiguity analysis, not purchase-probability prediction. No ambiguous generic `Cluster`
column or final business name is exported. The original RFM file is never written.
"""
# %%
assignments = rfm[["CustomerID", *RAW]].copy()
for name, labels in candidate_labels.items():
    assignments[name] = labels
for k in [2, 8]:
    name = "GMM_k2_MaxPosterior" if k == 2 else "GMM_k8_Diagnostic_MaxPosterior"
    assignments[name] = primary[("GMM", k)][2].predict_proba(X).max(axis=1)
OUTPUT = ROOT / "data/processed/cluster_assignments.csv"
assignments.to_csv(OUTPUT, index=False)
reloaded = pd.read_csv(OUTPUT)
pd.testing.assert_frame_equal(reloaded[["CustomerID", *RAW]], rfm[["CustomerID", *RAW]])
assert len(reloaded) == len(rfm) and reloaded.CustomerID.is_unique and not reloaded.isna().any().any()
for name, labels in candidate_labels.items():
    np.testing.assert_array_equal(reloaded[name], labels)
    profile = candidate_profiles[candidate_profiles.solution == name]
    assert profile.customers.sum() == len(rfm)
    assert np.isclose(profile.percentage.sum(), 100)
print("Verified assignment export:", OUTPUT.relative_to(ROOT).as_posix(), reloaded.shape)
display(reloaded.head())

# %% [markdown]
"""
## 13. Member 4 handoff, decisions and reproducibility record

Generate the numerical comparison, audit summary and handoff directly from this run.
The decision log is a Member 3 draft because the shared decision-log directory currently
contains only a placeholder. Generated documentation is overwritten only within Member 3;
review edits should be made in the source/report helper so reruns preserve them.

Member 4 should join future outcomes to these frozen IDs/assignments, keep customers with
no future purchases in the denominator, and compare repeat purchasing, order counts and
spend by candidate group. Those are later evaluation measures, not targets trained here.
Agree the precise three-month endpoint before computing outcomes: the full source ends
partway through 9 December 2011, and 90 days is not identical to three calendar months.

The comparison is provisional technical evidence. Business names, final winner,
recommendations and claims of future usefulness remain with Member 4.
"""
# %%
from modelling_report import write_reports
write_reports(DOCS, audit, input_summary, experiments, comparison, candidate_profiles,
              stability_summary, seed_runs, gmm_diagnostics, score_ranges, pca_variance,
              input_hashes, assignments)
for relative, original_hash in input_hashes.items():
    assert fingerprint(ROOT / relative) == original_hash, f"Input changed: {relative}"
packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "openpyxl",
            "nbformat", "nbclient", "ipykernel", "ipython", "jupyter_client", "threadpoolctl"]
manifest = {
    "python": platform.python_version(), "packages": {p: metadata.version(p) for p in packages},
    "input_sha256": input_hashes, "feature_columns": FEATURES, "primary_seed": 42,
    "stability_seeds": SEEDS, "cluster_counts": list(COUNTS),
    "pca_explained_variance_ratio": pca_variance.tolist(),
    "primary_model_parameters": {f"{m}_k{k}": result[2].get_params()
                                 for (m, k), result in primary.items()},
    "assignment_sha256": fingerprint(OUTPUT),
    "source_sha256": {p.name: fingerprint(p) for p in sorted((WORK / "notebooks").glob("*.py"))},
    "output_sha256": {p.relative_to(ROOT).as_posix(): fingerprint(p)
                      for folder in [RESULTS, FIGURES, DOCS] for p in sorted(folder.iterdir())
                      if p.is_file() and p.name not in ["run_manifest.json", ".gitkeep"]},
    "checks": {"source_inputs_unchanged": True, "export_roundtrip": True,
               "finite_features": True, "historical_only": True,
               "all_stochastic_fits_converged": True},
}
(RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
display(Markdown((DOCS / "member3_handoff.md").read_text(encoding="utf-8")))
print("All modelling, profile, export and input-preservation checks passed.")
