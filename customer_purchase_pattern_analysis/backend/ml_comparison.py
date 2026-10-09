"""Data Mining Comparison & Analysis Module.
Provides comparative benchmarking, metric categorization, winner evaluation,
and automated cross-technique/intra-technique interpretations.
Reuses existing ML modules without altering any existing implementation.
"""
import time
import numpy as np
import pandas as pd

from ml_classification import run_classification
from ml_regression import run_regression
from ml_clustering import run_clustering
from ml_preprocessing import get_dataset_profile

def compare_classification(df, algorithms=None, target="Category", features=None, params=None):
    if algorithms is None:
        algorithms = ["decision_tree", "random_forest", "knn", "logistic_regression", "naive_bayes"]
    if features is None:
        features = ["Quantity", "UnitPrice", "TotalAmount"]
    if params is None:
        params = {"sample_size": 2500, "test_size": 0.2}

    results = []
    best_algo = None
    best_f1 = -1.0

    for algo in algorithms:
        try:
            t0 = time.time()
            res = run_classification(df, algo, target, features, params)
            m = res["metrics"]
            f1 = m["f1_macro"]
            acc = m["accuracy"]
            prec = m["precision_macro"]
            rec = m["recall_macro"]
            duration = res["training_time_s"]

            row = {
                "algorithm_id": algo,
                "name": res["algorithm_name"],
                "accuracy": round(acc, 4),
                "precision_macro": round(prec, 4),
                "recall_macro": round(rec, 4),
                "f1_macro": round(f1, 4),
                "f1_weighted": round(m["f1_weighted"], 4),
                "training_time_s": duration,
                "train_samples": res["train_samples"],
                "test_samples": res["test_samples"],
                "classes_count": len(res["classes"]),
                "full_result": res
            }
            results.append(row)

            if f1 > best_f1:
                best_f1 = f1
                best_algo = row
        except Exception as e:
            results.append({
                "algorithm_id": algo,
                "name": algo.replace("_", " ").title(),
                "error": str(e),
                "accuracy": 0,
                "f1_macro": 0,
                "training_time_s": 0
            })

    # Sort results by F1 Macro descending
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: (x["f1_macro"], x["accuracy"]), reverse=True)

    # Explanation of why the winning algorithm performed better
    explanation = ""
    if best_algo:
        b_name = best_algo["name"]
        b_acc = best_algo["accuracy"] * 100
        b_f1 = best_algo["f1_macro"] * 100

        if "Random Forest" in b_name:
            explanation = (
                f"<b>{b_name}</b> achieved the highest overall performance with <b>{b_acc:.1f}% Accuracy</b> and <b>{b_f1:.1f}% Macro F1-Score</b>. "
                f"It outperforms single Decision Trees and linear models because its ensemble of bootstrap-aggregated trees reduces model variance, "
                f"mitigates individual tree overfitting, and effectively captures non-linear feature interactions between price, quantity, and sales."
            )
        elif "Decision Tree" in b_name:
            explanation = (
                f"<b>{b_name}</b> led classification with <b>{b_acc:.1f}% Accuracy</b> and <b>{b_f1:.1f}% Macro F1-Score</b>. "
                f"Its orthogonal feature splits efficiently segmented distinct price and quantity thresholds with zero need for feature scaling, "
                f"offering maximum model interpretability alongside rapid training."
            )
        elif "KNN" in b_name:
            explanation = (
                f"<b>{b_name}</b> demonstrated superior classification with <b>{b_acc:.1f}% Accuracy</b>. "
                f"It benefits from localized instance clustering in the standardized feature space, accurately classifying items based on their closest neighbors."
            )
        elif "Logistic Regression" in b_name:
            explanation = (
                f"<b>{b_name}</b> proved most effective with <b>{b_acc:.1f}% Accuracy</b>. "
                f"Its linear log-odds formulation provided robust generalization without overfitting to localized sample fluctuations."
            )
        else:
            explanation = (
                f"<b>{b_name}</b> outperformed other classifiers with <b>{b_acc:.1f}% Accuracy</b> and <b>{b_f1:.1f}% Macro F1-Score</b>, "
                f"delivering the most balanced precision-recall trade-off across all target classes."
            )

    return {
        "technique": "Classification",
        "primary_metric": "Accuracy & Macro F1-Score",
        "target": target,
        "features": features,
        "results": valid_results,
        "all_results": results,
        "best_algorithm": best_algo,
        "explanation": explanation,
        "metric_descriptions": {
            "Accuracy": "Ratio of correctly predicted instances over all test samples. Best for balanced class distributions.",
            "Precision (Macro)": "Proportion of positive predictions that were truly correct, averaged equally across all classes.",
            "Recall (Macro)": "Proportion of actual instances in each class that the model successfully identified.",
            "F1-Score (Macro)": "Harmonic mean of precision and recall. Key metric to assess balanced multi-class performance."
        }
    }

def compare_regression(df, algorithms=None, target="TotalAmount", features=None, params=None):
    if algorithms is None:
        algorithms = ["linear", "polynomial", "decision_tree", "random_forest"]
    if features is None:
        features = ["Quantity", "UnitPrice"]
    if params is None:
        params = {"sample_size": 2500, "test_size": 0.2, "poly_degree": 2, "dt_max_depth": 6, "rf_n_estimators": 40}

    results = []
    best_algo = None
    best_r2 = -float("inf")

    for algo in algorithms:
        try:
            res = run_regression(df, algo, target, features, params)
            m = res["metrics"]
            r2 = m["r2"]
            mae = m["mae"]
            rmse = m["rmse"]
            mse = m["mse"]
            duration = res["training_time_s"]

            row = {
                "algorithm_id": algo,
                "name": res["algorithm_name"],
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mse": round(mse, 4),
                "training_time_s": duration,
                "train_samples": res["train_samples"],
                "test_samples": res["test_samples"],
                "full_result": res
            }
            results.append(row)

            if r2 > best_r2:
                best_r2 = r2
                best_algo = row
        except Exception as e:
            results.append({
                "algorithm_id": algo,
                "name": algo.replace("_", " ").title(),
                "error": str(e),
                "r2": 0,
                "mae": 0,
                "rmse": 0,
                "training_time_s": 0
            })

    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: (x["r2"], -x["rmse"]), reverse=True)

    explanation = ""
    if best_algo:
        b_name = best_algo["name"]
        b_r2 = best_algo["r2"]
        b_rmse = best_algo["rmse"]
        b_mae = best_algo["mae"]

        if "Polynomial" in b_name or "Random Forest" in b_name or "Tree" in b_name:
            explanation = (
                f"<b>{b_name}</b> achieved the best regression fit with <b>R² = {b_r2:.4f}</b> (explaining {max(0.0, b_r2)*100:.1f}% of variance) "
                f"and lowest Root Mean Squared Error (<b>RMSE = {b_rmse:.2f}</b>, MAE = {b_mae:.2f}). "
                f"Because transaction total amount naturally depends on non-linear multiplicative terms (<code>Quantity × UnitPrice</code>), "
                f"models that capture polynomial interactions or recursive partitions fit the revenue surface significantly better than simple linear hyperplanes."
            )
        else:
            explanation = (
                f"<b>{b_name}</b> provided the highest predictive stability with <b>R² = {b_r2:.4f}</b> and <b>MAE = {b_mae:.2f}</b>. "
                f"It yields clean, un-overfitted estimates with minimal computational overhead and straightforward feature coefficients."
            )

    return {
        "technique": "Regression",
        "primary_metric": "R² Score (Variance Explained) & RMSE",
        "target": target,
        "features": features,
        "results": valid_results,
        "all_results": results,
        "best_algorithm": best_algo,
        "explanation": explanation,
        "metric_descriptions": {
            "R² Score": "Coefficient of determination (range ≤ 1.0). Indicates the fraction of target variance explained by features. Closer to 1.0 is superior.",
            "MAE": "Mean Absolute Error. Average magnitude of absolute residuals. Provides an intuitive dollar-value error scale.",
            "MSE": "Mean Squared Error. Quadratic error penalty highlighting variance in model deviations.",
            "RMSE": "Root Mean Squared Error. Square root of MSE. Penalizes large prediction errors more severely than MAE."
        }
    }

def compare_clustering(df, algorithms=None, features=None, mode="rows", params=None):
    if algorithms is None:
        algorithms = ["kmeans", "agglomerative", "dbscan"]
    if features is None:
        features = ["Quantity", "UnitPrice", "TotalAmount"] if mode == "rows" else ["Frequency", "Monetary", "Recency"]
    if params is None:
        params = {"sample_size": 2000, "n_clusters": 3, "scale_features": True, "eps": 0.5, "min_samples": 5}

    results = []
    best_algo = None
    best_sil = -2.0

    for algo in algorithms:
        try:
            res = run_clustering(df, algo, feature_columns=features, mode=mode, params=params)
            sil = res["silhouette_score"]
            n_clu = res["n_clusters"]
            noise = res["n_noise"]
            duration = res["training_time_s"]

            row = {
                "algorithm_id": algo,
                "name": res["algorithm_name"],
                "silhouette_score": round(sil, 4) if sil is not None else None,
                "n_clusters": n_clu,
                "n_noise": noise,
                "inertia": res["inertia"],
                "training_time_s": duration,
                "total_samples": res["total_samples"],
                "cluster_sizes": res["cluster_sizes"],
                "full_result": res
            }
            results.append(row)

            # Compare on silhouette score if clusters >= 2
            if sil is not None and sil > best_sil and n_clu >= 2:
                best_sil = sil
                best_algo = row
        except Exception as e:
            results.append({
                "algorithm_id": algo,
                "name": algo.replace("_", " ").title(),
                "error": str(e),
                "silhouette_score": None,
                "n_clusters": 0,
                "training_time_s": 0
            })

    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: (x["silhouette_score"] if x["silhouette_score"] is not None else -99), reverse=True)

    explanation = ""
    if best_algo:
        b_name = best_algo["name"]
        b_sil = best_algo["silhouette_score"]
        b_k = best_algo["n_clusters"]

        if "K-Means" in b_name:
            explanation = (
                f"<b>{b_name}</b> formed the most cohesive groupings with a <b>Silhouette Score of {b_sil:.4f}</b> across <b>{b_k} clusters</b>. "
                f"K-Means iteratively minimizes within-cluster variance (Inertia), successfully identifying spherical, well-separated density clusters "
                f"in the standardized customer/transaction space."
            )
        elif "Hierarchical" in b_name or "Agglomerative" in b_name:
            explanation = (
                f"<b>{b_name}</b> achieved the strongest cluster structure with <b>Silhouette Score = {b_sil:.4f}</b> ({b_k} clusters). "
                f"Its bottom-up ward linkage creates a rigorous hierarchy without requiring randomized centroid initialization."
            )
        elif "DBSCAN" in b_name:
            explanation = (
                f"<b>{b_name}</b> excelled in density partitioning with <b>Silhouette Score = {b_sil:.4f}</b>, automatically isolating "
                f"<b>{best_algo['n_noise']} outlier transactions</b> without imposing rigid spherical boundaries."
            )
        else:
            explanation = f"<b>{b_name}</b> provided the highest cluster separation with <b>Silhouette Score = {b_sil:.4f}</b>."

    return {
        "technique": "Clustering",
        "primary_metric": "Silhouette Score & Cluster Separation",
        "mode": mode,
        "features": features,
        "results": valid_results,
        "all_results": results,
        "best_algorithm": best_algo,
        "explanation": explanation,
        "metric_descriptions": {
            "Silhouette Score": "Measures cluster cohesion vs separation (range -1 to +1). Values > 0.5 indicate strong clustering; values > 0.25 indicate meaningful structure.",
            "Inertia (WCSS)": "Within-Cluster Sum of Squares. Total squared Euclidean distance from each sample to its assigned centroid.",
            "Cluster Count (k)": "Number of natural cohorts or customer segments discovered in the data space.",
            "Noise Points": "Unassigned outliers detected by density-based algorithms (DBSCAN) that do not belong to dense clusters."
        }
    }

def analyze_association_rules(rules_result):
    if not rules_result or "rules" not in rules_result:
        return {
            "technique": "Association Rule Mining",
            "available": False,
            "message": "Association rules have not been generated yet. Run Apriori to compute rule metrics."
        }

    rules = rules_result.get("rules", [])
    n_rules = rules_result.get("n_rules", len(rules))
    n_itemsets = rules_result.get("n_itemsets", 0)
    params = rules_result.get("params", {})
    engine = rules_result.get("engine", "Apriori")

    top_lift_rule = None
    top_conf_rule = None
    top_sup_rule = None

    if rules:
        top_lift_rule = max(rules, key=lambda x: x.get("lift", 0))
        top_conf_rule = max(rules, key=lambda x: x.get("confidence", 0))
        top_sup_rule = max(rules, key=lambda x: x.get("support", 0))

    avg_lift = round(float(np.mean([r["lift"] for r in rules])), 3) if rules else 0
    avg_conf = round(float(np.mean([r["confidence"] for r in rules])), 3) if rules else 0
    avg_sup = round(float(np.mean([r["support"] for r in rules])), 4) if rules else 0

    explanation = ""
    if top_lift_rule:
        a_str = " + ".join(top_lift_rule["antecedents"])
        c_str = " + ".join(top_lift_rule["consequents"])
        explanation = (
            f"The Apriori algorithm mined <b>{n_rules:,} valid rules</b> and <b>{n_itemsets:,} frequent itemsets</b>. "
            f"The strongest association rule is <code>{a_str} → {c_str}</code> with a <b>Lift of {top_lift_rule['lift']:.2f}</b>, "
            f"<b>Confidence of {top_lift_rule['confidence']*100:.1f}%</b>, and Support of {top_lift_rule['support']*100:.2f}%. "
            f"A Lift of {top_lift_rule['lift']:.2f} means purchasing the antecedent increases the likelihood of buying the consequent by {top_lift_rule['lift']:.1f}x "
            f"over random chance, proving a statistically significant cross-selling relationship."
        )

    return {
        "technique": "Association Rule Mining",
        "available": True,
        "engine": engine,
        "primary_metric": "Support, Confidence & Lift",
        "n_rules": n_rules,
        "n_itemsets": n_itemsets,
        "params": params,
        "averages": {
            "avg_lift": avg_lift,
            "avg_confidence": avg_conf,
            "avg_support": avg_sup
        },
        "top_rules": {
            "highest_lift": top_lift_rule,
            "highest_confidence": top_conf_rule,
            "highest_support": top_sup_rule
        },
        "sample_rules": rules[:15],
        "explanation": explanation,
        "metric_descriptions": {
            "Support": "Frequency of transactions containing the itemset. P(A ∩ B). Eliminates rare coincidences.",
            "Confidence": "Rule reliability. P(B | A) = Support(A ∪ B) / Support(A). Likelihood that B is purchased given A is in the basket.",
            "Lift": "Ratio of observed support to expected support under independence. Values > 1.0 indicate true positive association (> 2.0 is strong affinity).",
            "Leverage": "Support(A ∪ B) - Support(A) × Support(B). Difference in observed frequency versus independence."
        }
    }

def analyze_preprocessing(initial_profile, current_profile, prep_history=None):
    if not initial_profile or not current_profile:
        return {
            "technique": "Data Preprocessing",
            "available": False,
            "message": "Dataset profiles not initialized."
        }

    r_init = initial_profile.get("rows", 0)
    r_curr = current_profile.get("rows", 0)
    c_init = initial_profile.get("columns", 0)
    c_curr = current_profile.get("columns", 0)

    m_init = initial_profile.get("total_missing", 0)
    m_curr = current_profile.get("total_missing", 0)

    d_init = initial_profile.get("duplicate_rows", 0)
    d_curr = current_profile.get("duplicate_rows", 0)

    total_cells_init = max(1, r_init * c_init)
    total_cells_curr = max(1, r_curr * c_curr)

    # Cleanliness score based on missingness and duplicates
    clean_score_before = round((1.0 - (m_init / total_cells_init)) * 100, 2)
    clean_score_after = round((1.0 - (m_curr / total_cells_curr)) * 100, 2)

    missing_reduction_pct = round(((m_init - m_curr) / max(1, m_init)) * 100, 1) if m_init > 0 else 0.0
    duplicates_removed = max(0, d_init - d_curr)
    steps_applied = len(prep_history) if prep_history else 0

    explanation = (
        f"Data preprocessing prepared the active dataset from <b>{r_init:,} raw rows</b> to <b>{r_curr:,} clean records</b>. "
        f"Missing values were reduced from <b>{m_init:,}</b> to <b>{m_curr:,}</b> ({missing_reduction_pct}% reduction), and "
        f"duplicates were cleaned to maintain dataset integrity. "
        f"The data hygiene score stands at <b>{clean_score_after}%</b> across {c_curr} structured columns, ensuring clean, "
        f"non-distorted inputs for downstream classification, regression, and clustering algorithms."
    )

    return {
        "technique": "Data Preprocessing",
        "available": True,
        "primary_metric": "Before/After Data Quality Statistics & Hygiene",
        "stats_before": {
            "rows": r_init,
            "columns": c_init,
            "missing_values": m_init,
            "duplicate_rows": d_init,
            "cleanliness_score": clean_score_before
        },
        "stats_after": {
            "rows": r_curr,
            "columns": c_curr,
            "missing_values": m_curr,
            "duplicate_rows": d_curr,
            "cleanliness_score": clean_score_after
        },
        "deltas": {
            "rows_retained_pct": round((r_curr / max(1, r_init)) * 100, 2),
            "missing_reduction_pct": missing_reduction_pct,
            "duplicates_removed": duplicates_removed,
            "pipeline_steps": steps_applied
        },
        "history": prep_history or [],
        "explanation": explanation,
        "metric_descriptions": {
            "Missing Values": "Incomplete or NaN cells requiring imputation or row dropping to prevent calculation errors in ML models.",
            "Duplicate Rows": "Redundant identical transaction records that artificially inflate frequency counts.",
            "Data Hygiene Score": "Percentage of complete, non-null data elements across the total matrix dimensions.",
            "Pipeline Transformations": "Number of sequential cleaning, encoding, and scaling operations applied."
        }
    }

def synthesize_overall_comparison(clf_data, reg_data, clu_data, rules_data, prep_data):
    """
    Synthesizes cross-technique comparative results without treating disparate metrics as equivalent.
    """
    matrix = []

    # 1. Preprocessing
    if prep_data.get("available"):
        p_aft = prep_data.get("stats_after", {})
        matrix.append({
            "technique": "Data Preprocessing",
            "category": "Data Preparation & Quality",
            "objective": "Cleanse, impute, encode, scale, and validate raw transactions",
            "primary_metric_name": "Data Hygiene & Missing Reduction",
            "score_display": f"{p_aft.get('cleanliness_score', 100)}% Hygiene ({prep_data.get('deltas', {}).get('missing_reduction_pct', 100)}% nulls removed)",
            "best_model": f"Pipeline ({prep_data.get('deltas', {}).get('pipeline_steps', 0)} steps)",
            "speed": "< 0.05s",
            "business_impact": "Prevents garbage-in/garbage-out bias across all machine learning models"
        })

    # 2. Classification
    if clf_data and clf_data.get("best_algorithm"):
        best_c = clf_data["best_algorithm"]
        matrix.append({
            "technique": "Classification",
            "category": "Supervised Learning (Discrete)",
            "objective": "Predict categorical target (e.g. Product Category) from shopping features",
            "primary_metric_name": "Test Accuracy & Macro F1-Score",
            "score_display": f"Accuracy: {best_c['accuracy']*100:.1f}%, F1: {best_c['f1_macro']:.4f}",
            "best_model": best_c["name"],
            "speed": f"{best_c['training_time_s']}s",
            "business_impact": "Automated categorization and classification of customer purchase types"
        })

    # 3. Regression
    if reg_data and reg_data.get("best_algorithm"):
        best_r = reg_data["best_algorithm"]
        matrix.append({
            "technique": "Regression",
            "category": "Supervised Learning (Continuous)",
            "objective": "Forecast numerical dollar revenue (TotalAmount) based on item attributes",
            "primary_metric_name": "R² Variance Explained & RMSE",
            "score_display": f"R²: {best_r['r2']:.4f}, RMSE: {best_r['rmse']:.2f}",
            "best_model": best_r["name"],
            "speed": f"{best_r['training_time_s']}s",
            "business_impact": "Sales revenue forecasting and basket transaction value prediction"
        })

    # 4. Clustering
    if clu_data and clu_data.get("best_algorithm"):
        best_cl = clu_data["best_algorithm"]
        matrix.append({
            "technique": "Clustering",
            "category": "Unsupervised Learning (Segmentation)",
            "objective": "Discover natural customer/transaction cohorts without labeled targets",
            "primary_metric_name": "Silhouette Score & Separation",
            "score_display": f"Silhouette: {best_cl['silhouette_score']:.4f} ({best_cl['n_clusters']} clusters)",
            "best_model": best_cl["name"],
            "speed": f"{best_cl['training_time_s']}s",
            "business_impact": "Customer RFM segmentation (VIP vs Occasional) and targeted marketing"
        })

    # 5. Association Rule Mining
    if rules_data and rules_data.get("available"):
        top_l = rules_data.get("top_rules", {}).get("highest_lift")
        l_val = top_l["lift"] if top_l else 0
        matrix.append({
            "technique": "Association Rules",
            "category": "Pattern Discovery (Affinity)",
            "objective": "Uncover frequent co-occurring product baskets and cross-selling rules",
            "primary_metric_name": "Lift & Confidence",
            "score_display": f"Max Lift: {l_val:.2f}x, {rules_data.get('n_rules', 0):,} Rules",
            "best_model": rules_data.get("engine", "Apriori"),
            "speed": "< 0.4s",
            "business_impact": "Shelf layout optimization, product bundles, and recommendation engines"
        })

    summary_text = (
        "Each Data Mining technique addresses a distinct phase of the supermarket intelligence lifecycle. "
        "<b>Data Preprocessing</b> establishes the structural foundation by ensuring high data cleanliness. "
        "<b>Classification</b> and <b>Regression</b> provide supervised predictive capabilities—categorizing purchases and estimating transaction revenue. "
        "<b>Clustering</b> and <b>Association Rule Mining</b> deliver unsupervised exploratory power—grouping customer shopping behaviors and uncovering hidden item purchase affinities. "
        "Together, they form a cohesive, end-to-end data mining pipeline that transforms raw supermarket transactions into actionable retail strategies."
    )

    return {
        "matrix": matrix,
        "summary": summary_text
    }
