"""Clustering Module for Supermarket Data Mining Platform.
Supports: K-Means, Agglomerative (Hierarchical) Clustering, DBSCAN.
Computes cluster assignments, centroids, Silhouette Score, 2D PCA/feature projection, and cluster profiles.
Supports both Transaction/Item Level and Customer RFM (Recency, Frequency, Monetary) clustering.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def prepare_clustering_data(df, mode="rows", feature_columns=None):
    """
    Prepares features for clustering.
    mode='rows': uses selected row-level columns.
    mode='rfm': computes customer-level Recency, Frequency, Monetary dataset.
    """
    if mode == "rfm":
        if "CustomerID" not in df.columns:
            raise ValueError("Dataset does not have CustomerID column for RFM clustering.")
        
        has_dates = "Date" in df.columns and df["Date"].notna().any()
        date_series = pd.to_datetime(df["Date"], errors="coerce") if has_dates else None
        has_valid_dates = date_series is not None and date_series.notna().any()
        ref_date = date_series.max() if has_valid_dates else None

        df_rfm = df.copy()
        if has_valid_dates:
            df_rfm["_ParsedDate"] = date_series

        grouped = df_rfm.groupby("CustomerID")
        rfm_list = []
        
        for cid, g in grouped:
            # Frequency
            tx_count = g["TransactionID"].nunique() if "TransactionID" in g.columns else len(g)
            # Monetary
            monetary = float(g["TotalAmount"].sum()) if "TotalAmount" in g.columns else float(len(g))
            # Recency
            if has_valid_dates and g["_ParsedDate"].notna().any():
                recency = int((ref_date - g["_ParsedDate"].max()).days)
            else:
                recency = 0
            
            rfm_list.append({
                "CustomerID": str(cid),
                "Recency": recency,
                "Frequency": int(tx_count),
                "Monetary": round(monetary, 2)
            })
        
        rfm_df = pd.DataFrame(rfm_list)
        available_feats = ["Recency", "Frequency", "Monetary"]
        selected_feats = [f for f in (feature_columns or available_feats) if f in rfm_df.columns]
        if not selected_feats:
            selected_feats = available_feats
        return rfm_df, selected_feats, "CustomerID"
    else:
        # Row level
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        selected_feats = [f for f in (feature_columns or num_cols) if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if not selected_feats:
            raise ValueError("No numeric features found for clustering. Please select at least 2 numerical features.")
        return df[selected_feats].copy(), selected_feats, None

def run_clustering(df, algorithm, feature_columns=None, mode="rows", params=None):
    """
    Executes clustering on df with the chosen algorithm and parameters.
    Returns cluster metrics, silhouette score, 2D coordinates for scatter visualization, and cluster profiles.
    """
    if params is None:
        params = {}
    t0 = time.time()

    data_df, feats, id_col = prepare_clustering_data(df, mode=mode, feature_columns=feature_columns)
    if len(feats) < 2:
        raise ValueError(f"Please select at least 2 numeric features for clustering. Selected: {feats}")

    # Clean missing values
    data_df = data_df.dropna(subset=feats).copy()
    if len(data_df) < 15:
        raise ValueError("Dataset has too few records (<15) for clustering.")

    # Sample data to keep Hierarchical / DBSCAN distance computations snappy (<1s)
    sample_size = int(params.get("sample_size", 3000))
    if sample_size > 0 and len(data_df) > sample_size:
        data_df = data_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    X_raw = data_df[feats].values
    
    # Scale features
    scale = bool(params.get("scale_features", True))
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X = X_raw

    algo = str(algorithm).lower().strip()
    algo_display_name = "Clustering"
    inertia = None
    centroids = []
    
    if algo == "kmeans":
        n_clusters = int(params.get("n_clusters", 3))
        n_clusters = max(2, min(10, n_clusters))
        max_iter = int(params.get("max_iter", 300))
        model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        algo_display_name = f"K-Means (k={n_clusters})"
        inertia = round(float(model.inertia_), 2)

        # Centroids in original scale
        raw_centers = scaler.inverse_transform(model.cluster_centers_) if scale else model.cluster_centers_
        for idx, center in enumerate(raw_centers):
            c_dict = {"cluster": f"Cluster {idx}"}
            for f_i, f_name in enumerate(feats):
                c_dict[f_name] = round(float(center[f_i]), 2)
            centroids.append(c_dict)

    elif algo == "agglomerative":
        n_clusters = int(params.get("n_clusters", 3))
        n_clusters = max(2, min(10, n_clusters))
        linkage = params.get("linkage", "ward")
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
        algo_display_name = f"Hierarchical Agglomerative ({linkage} linkage, k={n_clusters})"

    elif algo == "dbscan":
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        algo_display_name = f"DBSCAN (eps={eps}, min_samples={min_samples})"

    else:
        raise ValueError(f"Unsupported clustering algorithm '{algorithm}'.")

    train_time = round(time.time() - t0, 3)

    # Unique clusters
    unique_labels = sorted(list(set(labels)))
    non_noise_labels = [lbl for lbl in unique_labels if lbl != -1]
    n_clusters_found = len(non_noise_labels)
    n_noise = int((labels == -1).sum()) if -1 in unique_labels else 0

    # Silhouette score
    sil_score = None
    if n_clusters_found >= 2:
        try:
            # For DBSCAN or sampled data, calculate on non-noise points if possible
            if n_noise > 0:
                mask = labels != -1
                if mask.sum() > 20 and len(set(labels[mask])) >= 2:
                    sil_score = round(float(silhouette_score(X[mask], labels[mask])), 4)
            else:
                sil_score = round(float(silhouette_score(X, labels)), 4)
        except Exception:
            sil_score = None

    # Cluster sizes
    cluster_counts = {}
    for lbl in unique_labels:
        c_name = f"Cluster {lbl}" if lbl != -1 else "Noise / Outliers"
        cluster_counts[c_name] = int((labels == lbl).sum())

    # Cluster profiles (mean values of features per cluster)
    cluster_profiles = []
    for lbl in unique_labels:
        mask = labels == lbl
        pts = X_raw[mask]
        c_name = f"Cluster {lbl}" if lbl != -1 else "Noise (-1)"
        prof = {
            "cluster": c_name,
            "count": int(mask.sum()),
            "pct": round(float(mask.sum() / len(data_df) * 100), 1)
        }
        for f_i, f_name in enumerate(feats):
            prof[f_name] = round(float(np.mean(pts[:, f_i])), 2)
        cluster_profiles.append(prof)

    # 2D projection for visualization
    # If 2 features: use directly; If > 2 features: use PCA(2)
    if len(feats) == 2:
        x_pts = X_raw[:, 0]
        y_pts = X_raw[:, 1]
        x_label = feats[0]
        y_label = feats[1]
        pca_explained = None
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        x_pts = X_pca[:, 0]
        y_pts = X_pca[:, 1]
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"
        pca_explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # Sample points for scatter visualization (up to 400 points)
    scatter_data = []
    step = max(1, len(data_df) // 400)
    for i in range(0, len(data_df), step):
        lbl = int(labels[i])
        c_name = f"Cluster {lbl}" if lbl != -1 else "Noise"
        scatter_data.append({
            "x": round(float(x_pts[i]), 2),
            "y": round(float(y_pts[i]), 2),
            "cluster": c_name,
            "cluster_id": lbl
        })

    # Sample table (first 25 rows)
    sample_assignments = []
    for i in range(min(25, len(data_df))):
        row_dict = {f: round(float(X_raw[i, idx]), 2) for idx, f in enumerate(feats)}
        if id_col and id_col in data_df.columns:
            row_dict[id_col] = str(data_df[id_col].iloc[i])
        lbl = int(labels[i])
        row_dict["AssignedCluster"] = f"Cluster {lbl}" if lbl != -1 else "Noise (-1)"
        sample_assignments.append(row_dict)

    summary_bullets = [
        f"Executed <b>{algo_display_name}</b> on {len(data_df):,} instances using features: <code>{', '.join(feats)}</code>.",
        f"Identified <b>{n_clusters_found} distinct clusters</b>" + (f" with {n_noise:,} noise points." if n_noise else "."),
    ]
    if sil_score is not None:
        summary_bullets.append(f"Silhouette Score: <b>{sil_score:.4f}</b> ({'Strong' if sil_score > 0.5 else 'Reasonable' if sil_score > 0.25 else 'Weak/Overlapping'} clustering separation).")
    if inertia is not None:
        summary_bullets.append(f"Within-cluster sum of squares (Inertia): <b>{inertia:,.1f}</b>.")

    return {
        "algorithm": algo,
        "algorithm_name": algo_display_name,
        "mode": mode,
        "features": feats,
        "total_samples": int(len(data_df)),
        "n_clusters": n_clusters_found,
        "n_noise": n_noise,
        "silhouette_score": sil_score,
        "inertia": inertia,
        "training_time_s": train_time,
        "cluster_sizes": cluster_counts,
        "centroids": centroids,
        "cluster_profiles": cluster_profiles,
        "scatter_data": scatter_data,
        "scatter_axes": {"x_label": x_label, "y_label": y_label},
        "pca_explained": pca_explained,
        "sample_assignments": sample_assignments,
        "summary": summary_bullets
    }
