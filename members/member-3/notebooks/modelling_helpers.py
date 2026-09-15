"""Small shared helpers for Member 3's notebook and readable Python source."""
from hashlib import sha256
from itertools import combinations
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.mixture import GaussianMixture

FEATURES = ["Recency_scaled", "Frequency_scaled", "Monetary_scaled"]
RAW = ["Recency", "Frequency", "Monetary"]
SEEDS = [42, 7, 21, 99, 123]
COUNTS = range(2, 9)


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def audit_input(rfm, transactions_path):
    """Read-only contract checks; no cleaning, refitting scaler, or replacement data."""
    required = ["CustomerID", *RAW, "Log_Frequency", "Log_Monetary", *FEATURES]
    if list(rfm.columns) != required:
        raise ValueError(f"Review changed Member 2 schema: {list(rfm.columns)}")
    if len(rfm) < 10 or rfm.isna().any().any():
        raise ValueError("Missing values or insufficient customers: consult Member 2")
    if not all(pd.api.types.is_numeric_dtype(rfm[c]) for c in required):
        raise ValueError("Expected numeric handoff columns")
    if not np.isfinite(rfm.to_numpy()).all() or not rfm.CustomerID.is_unique:
        raise ValueError("Non-finite values or duplicate CustomerID: consult Member 2")
    if not ((rfm.Recency >= 0) & (rfm.Frequency >= 1) & (rfm.Monetary > 0)).all():
        raise ValueError("Invalid raw RFM domain")
    if not np.equal(rfm.CustomerID, np.floor(rfm.CustomerID)).all():
        raise ValueError("CustomerID must be an integer identifier")
    errors = {}
    for raw, logged in [("Frequency", "Log_Frequency"), ("Monetary", "Log_Monetary")]:
        errors[logged] = float(np.max(np.abs(np.log1p(rfm[raw]) - rfm[logged])))
    # Algebraic verification of the existing values, never substituted into X.
    for original, scaled in zip(["Recency", "Log_Frequency", "Log_Monetary"], FEATURES):
        expected = (rfm[original] - rfm[original].mean()) / rfm[original].std(ddof=0)
        errors[scaled] = float(np.max(np.abs(expected - rfm[scaled])))
    if max(errors.values()) > 1e-7:
        raise ValueError(f"Transformation mismatch beyond CSV precision: {errors}")
    np.testing.assert_allclose(rfm[FEATURES].mean(), 0, atol=1e-7)
    np.testing.assert_allclose(rfm[FEATURES].std(ddof=0), 1, atol=1e-7)
    dates = pd.read_csv(transactions_path, usecols=["CustomerID", "InvoiceDate"],
                        parse_dates=["InvoiceDate"])
    if dates.isna().any().any() or not (dates.InvoiceDate < pd.Timestamp("2011-09-09")).all():
        raise ValueError("Missing transaction metadata or reserved-period rows in handoff")
    if set(dates.CustomerID) != set(rfm.CustomerID):
        raise ValueError("RFM customer population differs from cleaned transactions")
    return {
        "rows": len(rfm), "unique_customers": int(rfm.CustomerID.nunique()),
        "duplicate_ids": int(rfm.CustomerID.duplicated().sum()),
        "duplicate_rows": int(rfm.duplicated().sum()),
        "duplicate_feature_vectors": int(rfm[FEATURES].duplicated().sum()),
        "missing_values": int(rfm.isna().sum().sum()), "infinite_values": 0,
        "dtypes": rfm.dtypes.astype(str).to_dict(),
        "scaled_means": rfm[FEATURES].mean().to_dict(),
        "scaled_population_std": rfm[FEATURES].std(ddof=0).to_dict(),
        "transformation_max_absolute_errors": errors,
        "one_order_customers": int((rfm.Frequency == 1).sum()),
        "frequency_monetary_spearman": float(rfm[RAW].corr(method="spearman").loc["Frequency", "Monetary"]),
        "transaction_rows": len(dates), "date_min": str(dates.InvoiceDate.min()),
        "date_max": str(dates.InvoiceDate.max()), "reserved_period_rows": 0,
    }


def rfm_baseline(rfm):
    """Midrank quintile scores preserve ties; fixed total-score bands form 4 groups."""
    scores = pd.DataFrame(index=rfm.index)
    for raw, short in zip(RAW, ["R", "F", "M"]):
        # Worst-to-best ordering makes every score increase with desirability.
        rank = rfm[raw].rank(method="average", ascending=(raw != "Recency"))
        percentile = (rank - 0.5) / len(rfm)
        scores[f"{short}_Score"] = (np.floor(5 * percentile) + 1).clip(1, 5).astype(int)
    scores["RFM_Total"] = scores.sum(axis=1)
    # Equal-weight average: <=2, >2 to 3, >3 to 4, >4.
    scores["RFM_Baseline_Group"] = np.searchsorted([6, 9, 12], scores.RFM_Total, side="left")
    return scores


def make_model(method, k, seed=42):
    if method == "KMeans":
        return KMeans(n_clusters=k, init="k-means++", n_init=20,
                      max_iter=500, tol=1e-4, algorithm="lloyd", random_state=seed)
    if method == "Hierarchical":
        return AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean",
                                       compute_full_tree=True, compute_distances=True)
    if method == "GMM":
        return GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                               max_iter=500, tol=1e-3, reg_covar=1e-6,
                               init_params="kmeans", random_state=seed)
    raise ValueError(method)


def cluster_metrics(X, labels):
    sizes = pd.Series(labels).value_counts().sort_index()
    if not 1 < len(sizes) < len(X):
        raise ValueError("Metrics require 2 to n-1 populated groups")
    return {
        "groups": len(sizes),
        "silhouette": silhouette_score(X, labels, metric="euclidean"),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max()),
        "smallest_pct": float(100 * sizes.min() / len(X)),
        "sizes": json.dumps({str(k): int(v) for k, v in sizes.items()}),
    }


def experiment(method, k, X, seed=42):
    model = make_model(method, k, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        labels = model.fit_predict(X)
    if len(np.unique(labels)) != k:
        raise ValueError(f"{method} k={k}: fewer populated groups than requested")
    row = {"method": method, "k": k, "seed": np.nan if method == "Hierarchical" else seed,
           **cluster_metrics(X, labels)}
    if method == "KMeans":
        row.update(inertia=model.inertia_, iterations=model.n_iter_)
        if model.n_iter_ >= model.max_iter:
            raise ValueError("KMeans reached iteration cap")
    elif method == "GMM":
        if not model.converged_:
            raise ValueError("GMM did not converge")
        confidence = model.predict_proba(X).max(axis=1)
        row.update(aic=model.aic(X), bic=model.bic(X), iterations=model.n_iter_,
                   converged=bool(model.converged_), mean_confidence=float(confidence.mean()),
                   low_confidence_count=int((confidence < .7).sum()),
                   low_confidence_pct=float(100 * (confidence < .7).mean()),
                   min_covariance_eigenvalue=float(np.linalg.eigvalsh(model.covariances_).min()))
    return row, labels, model


def repeat_experiments(X, primary):
    """All five seeds, all seven k values; ten label-invariant pairs per config."""
    runs, pairs = [], []
    for method in ["KMeans", "GMM"]:
        for k in COUNTS:
            labels_by_seed = {}
            for seed in SEEDS:
                row, labels, _ = primary[(method, k)] if seed == 42 else experiment(method, k, X, seed)
                runs.append(row)
                labels_by_seed[seed] = labels
            for a, b in combinations(SEEDS, 2):
                pairs.append({"method": method, "k": k, "seed_a": a, "seed_b": b,
                              "ari": adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])})
            print(f"Stability complete: {method} k={k}", flush=True)
    pairs = pd.DataFrame(pairs)
    summary = pairs.groupby(["method", "k"], as_index=False).agg(
        ari_mean=("ari", "mean"), ari_min=("ari", "min"),
        ari_max=("ari", "max"), ari_std=("ari", "std"), pairs=("ari", "size"))
    return pd.DataFrame(runs), pairs, summary


def profiles(rfm, labels, solution):
    result = rfm.assign(cluster=labels).groupby("cluster").agg(
        customers=("CustomerID", "size"),
        median_recency=("Recency", "median"), median_frequency=("Frequency", "median"),
        median_monetary=("Monetary", "median"), mean_recency=("Recency", "mean"),
        mean_frequency=("Frequency", "mean"), mean_monetary=("Monetary", "mean"),
    ).reset_index()
    result.insert(0, "solution", solution)
    result.insert(3, "percentage", 100 * result.customers / len(rfm))
    return result


def markdown_table(frame, decimals=4):
    """Simple Markdown output without adding tabulate as a dependency."""
    def fmt(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value).replace("|", "/").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)
