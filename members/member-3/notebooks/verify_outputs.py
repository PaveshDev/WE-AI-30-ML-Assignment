"""Independently verify Member 3's saved evidence without fitting new models."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from modelling_helpers import (
    FEATURES, RAW, SEEDS, audit_input, cluster_metrics, fingerprint, rfm_baseline,
)


def main():
    work = Path(__file__).resolve().parents[1]
    root = work.parents[1]
    results = work / "results"
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    rfm = pd.read_csv(root / "data/processed/rfm_table.csv")
    assignments = pd.read_csv(root / "data/processed/cluster_assignments.csv")
    audit_input(rfm, root / "data/interim/cleaned_transactions.csv")
    for relative, expected in {**manifest["input_sha256"], **manifest["output_sha256"]}.items():
        assert fingerprint(root / relative) == expected, f"Changed artifact: {relative}"
    for name, expected in manifest["source_sha256"].items():
        assert fingerprint(work / "notebooks" / name) == expected, f"Changed source: {name}"
    assert fingerprint(root / "data/processed/cluster_assignments.csv") == manifest["assignment_sha256"]
    assert len(assignments) == len(rfm) and assignments.CustomerID.is_unique
    assert not assignments.isna().any().any()
    pd.testing.assert_frame_equal(assignments[["CustomerID", *RAW]], rfm[["CustomerID", *RAW]])

    experiments = pd.read_csv(results / "experiments.csv")
    assert len(experiments) == 22
    for method in ["KMeans", "Hierarchical", "GMM"]:
        assert set(experiments.loc[experiments.method == method, "k"]) == set(range(2, 9))
    configs = {"RFM_Baseline_Group": ("RFM_Baseline", 4), "KMeans_k2": ("KMeans", 2),
               "KMeans_k3": ("KMeans", 3), "Hierarchical_k2": ("Hierarchical", 2),
               "GMM_k2": ("GMM", 2), "GMM_k8_Diagnostic": ("GMM", 8)}
    profiles = pd.read_csv(results / "cluster_profiles.csv")
    assert profiles.solution.nunique() == 22
    for _, profile in profiles.groupby("solution"):
        assert profile.customers.sum() == len(rfm)
        assert np.isclose(profile.percentage.sum(), 100)
    for column, (method, k) in configs.items():
        row = experiments[(experiments.method == method) & (experiments.k == k)].iloc[0]
        calculated = cluster_metrics(rfm[FEATURES].to_numpy(), assignments[column])
        for metric in ["silhouette", "davies_bouldin", "calinski_harabasz", "smallest_cluster", "largest_cluster"]:
            np.testing.assert_allclose(calculated[metric], row[metric], rtol=1e-9, atol=1e-9)
        counts = assignments[column].value_counts().sort_index()
        recorded = profiles[profiles.solution == column].set_index("cluster").customers.sort_index()
        np.testing.assert_array_equal(counts, recorded)
        assert counts.size == k
    for column in ["GMM_k2_MaxPosterior", "GMM_k8_Diagnostic_MaxPosterior"]:
        assert assignments[column].between(0, 1).all()

    # Behavioural properties catch tie splitting, reversed Recency and row-order dependence.
    scores = rfm_baseline(rfm)
    np.testing.assert_array_equal(scores.RFM_Baseline_Group, assignments.RFM_Baseline_Group)
    shuffled = rfm.sample(frac=1, random_state=19)
    pd.testing.assert_frame_equal(rfm_baseline(shuffled).sort_index(), scores)
    for raw, short in zip(RAW, ["R", "F", "M"]):
        values = pd.DataFrame({"raw": rfm[raw], "score": scores[f"{short}_Score"]})
        assert values.groupby("raw").score.nunique().max() == 1
        unique = values.drop_duplicates().sort_values("raw").score
        assert unique.is_monotonic_decreasing if raw == "Recency" else unique.is_monotonic_increasing
        assert unique.between(1, 5).all()

    runs = pd.read_csv(results / "stability_runs.csv")
    pairs = pd.read_csv(results / "stability_pairs.csv")
    summary = pd.read_csv(results / "stability_summary.csv")
    assert len(runs) == 70 and len(pairs) == 140 and len(summary) == 14
    assert set(runs.method) == {"KMeans", "GMM"}
    for (method, k), group in runs.groupby(["method", "k"]):
        assert set(group.seed) == set(SEEDS) and len(group) == 5
        evidence = pairs[(pairs.method == method) & (pairs.k == k)]
        assert len(evidence) == 10 and not evidence.duplicated(["seed_a", "seed_b"]).any()
        row = summary[(summary.method == method) & (summary.k == k)].iloc[0]
        np.testing.assert_allclose([row.ari_mean, row.ari_min], [evidence.ari.mean(), evidence.ari.min()], atol=1e-10)
    assert experiments.loc[experiments.method.isin(["Hierarchical", "RFM_Baseline"]), "ari_mean"].isna().all()

    notebook = nbformat.read(work / "notebooks/03_modelling_and_comparison.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code)+1))
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    assert len([cell for cell in notebook.cells if cell.cell_type == "markdown"]) >= 13
    print("PASS: input preservation, hashes, 22 experiments/profiles, exported metrics and labels,")
    print("baseline tie/order/direction properties, 70 seed runs, 140 ARI pairs, and sequential notebook execution.")


if __name__ == "__main__":
    main()
