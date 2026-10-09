"""Classification Module for Supermarket Data Mining Platform.
Supports: Decision Tree, KNN, Logistic Regression, Naive Bayes, Random Forest, SVM.
Calculates Accuracy, Precision, Recall, F1, Confusion Matrix, and generates visualizations.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def run_classification(df, algorithm, target_column, feature_columns, params=None):
    """
    Trains and evaluates a classification model.
    Returns metrics, confusion matrix, feature importances, sample predictions, and metadata.
    """
    if params is None:
        params = {}
    t0 = time.time()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset.")
    if not feature_columns or len(feature_columns) == 0:
        raise ValueError("Please select at least one feature column.")
    
    valid_features = [c for c in feature_columns if c in df.columns and c != target_column]
    if not valid_features:
        raise ValueError("Selected features are invalid or match the target column.")

    # Drop rows where target is missing
    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < 20:
        raise ValueError("Dataset has too few non-null rows (<20) for classification.")

    # Sample if dataset is large to guarantee fast sub-second execution
    sample_size = int(params.get("sample_size", 5000))
    if sample_size > 0 and len(clean_df) > sample_size:
        clean_df = clean_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    y_raw = clean_df[target_column]
    # Check number of distinct classes and handle high cardinality
    n_unique_y = y_raw.nunique()
    if n_unique_y < 2:
        raise ValueError(f"Target column '{target_column}' must have at least 2 distinct classes. Found: {n_unique_y}.")
    
    # If target has too many classes (e.g. > 12 categories or continuous values), keep top 8 most frequent and group rest
    cardinality_note = ""
    if n_unique_y > 12:
        top_classes = y_raw.value_counts().head(8).index.tolist()
        y_raw = y_raw.apply(lambda v: str(v) if v in top_classes else "Other")
        cardinality_note = f"Target '{target_column}' had {n_unique_y} classes; grouped to top 8 frequent classes + 'Other' for effective classification."

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]

    # Prepare features: handle missing and encode any categorical features
    X_df = clean_df[valid_features].copy()
    processed_features = []
    
    for col in valid_features:
        s = X_df[col]
        if pd.api.types.is_numeric_dtype(s):
            X_df[col] = s.fillna(s.mean() if s.count() > 0 else 0)
            processed_features.append(col)
        else:
            # Categorical feature: label encode or fill
            le = LabelEncoder()
            X_df[col] = le.fit_transform(s.fillna("Missing").astype(str))
            processed_features.append(col)

    X = X_df[processed_features].values

    # Train / Test split
    test_size = float(params.get("test_size", 0.2))
    test_size = max(0.1, min(0.5, test_size))
    random_state = int(params.get("random_state", 42))

    # Stratified split if each class has at least 2 samples
    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Scaling for distance/gradient based algorithms
    algo = str(algorithm).lower().strip()
    needs_scaling = algo in ("knn", "logistic_regression", "svm", "naive_bayes")
    scaler = None
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize model
    algo_display_name = "Classification"
    feature_importance_list = []
    
    if algo == "decision_tree":
        max_depth = int(params.get("dt_max_depth", 5)) if params.get("dt_max_depth") else None
        criterion = params.get("dt_criterion", "gini")
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
        algo_display_name = "Decision Tree Classifier"

    elif algo == "knn":
        n_neighbors = int(params.get("knn_n_neighbors", 5))
        n_neighbors = min(n_neighbors, len(X_train))
        metric = params.get("knn_metric", "minkowski")
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        algo_display_name = f"K-Nearest Neighbors (k={n_neighbors})"

    elif algo == "logistic_regression":
        c_val = float(params.get("lr_c", 1.0))
        max_iter = int(params.get("lr_max_iter", 300))
        model = LogisticRegression(C=c_val, max_iter=max_iter, random_state=random_state)
        algo_display_name = "Logistic Regression"

    elif algo == "naive_bayes":
        var_smoothing = float(params.get("nb_var_smoothing", 1e-9))
        model = GaussianNB(var_smoothing=var_smoothing)
        algo_display_name = "Gaussian Naive Bayes"

    elif algo == "random_forest":
        n_estimators = int(params.get("rf_n_estimators", 50))
        max_depth = int(params.get("rf_max_depth", 6)) if params.get("rf_max_depth") else None
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        algo_display_name = f"Random Forest ({n_estimators} trees)"

    elif algo == "svm":
        kernel = params.get("svm_kernel", "rbf")
        c_val = float(params.get("svm_c", 1.0))
        # Limit train size for SVM to keep under 1-2 seconds if needed
        if len(X_train) > 3000:
            sub_idx = np.random.RandomState(random_state).choice(len(X_train), size=3000, replace=False)
            X_train_fit = X_train[sub_idx]
            y_train_fit = y_train[sub_idx]
        else:
            X_train_fit = X_train
            y_train_fit = y_train
        model = SVC(kernel=kernel, C=c_val, random_state=random_state)
        algo_display_name = f"Support Vector Machine ({kernel} kernel)"
    else:
        raise ValueError(f"Unsupported classification algorithm '{algorithm}'.")

    # Fit model
    if algo == "svm" and len(X_train) > 3000:
        model.fit(X_train_fit, y_train_fit)
    else:
        model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    train_time = round(time.time() - t0, 3)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()

    # Per class report
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    per_class_metrics = []
    for cls in class_names:
        if cls in report_dict:
            r = report_dict[cls]
            per_class_metrics.append({
                "class_name": cls,
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1-score"], 4),
                "support": int(r["support"])
            })

    # Feature importances or coefficients
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feature_importance_list = sorted(
            [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(processed_features, fi)],
            key=lambda x: x["importance"], reverse=True
        )
    elif hasattr(model, "coef_"):
        # For logistic regression
        coefs = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        feature_importance_list = sorted(
            [{"feature": f, "importance": round(float(c), 4)} for f, c in zip(processed_features, coefs)],
            key=lambda x: x["importance"], reverse=True
        )

    # Sample predictions table (first 25 test samples)
    sample_preds = []
    inv_X_test = scaler.inverse_transform(X_test) if scaler else X_test
    for i in range(min(25, len(y_test))):
        feats = {f: round(float(inv_X_test[i][idx]), 2) for idx, f in enumerate(processed_features)}
        act_class = class_names[y_test[i]]
        pred_class = class_names[y_pred[i]]
        sample_preds.append({
            "features": feats,
            "actual": act_class,
            "predicted": pred_class,
            "correct": bool(act_class == pred_class)
        })

    # Interpretation
    pct = lambda v: f"{v * 100:.1f}%"
    summary_bullets = [
        f"Trained <b>{algo_display_name}</b> on {len(X_train):,} training samples and evaluated on {len(X_test):,} test samples.",
        f"Overall Test Accuracy: <b>{pct(acc)}</b> with F1-Score (macro): <b>{pct(f1_macro)}</b>.",
        f"Macro Precision: <b>{pct(prec_macro)}</b>, Macro Recall: <b>{pct(rec_macro)}</b>."
    ]
    if cardinality_note:
        summary_bullets.insert(0, cardinality_note)
    if feature_importance_list:
        top_f = feature_importance_list[0]
        summary_bullets.append(f"Most influential feature: <b>{top_f['feature']}</b> ({pct(top_f['importance'])} relative importance).")

    return {
        "algorithm": algo,
        "algorithm_name": algo_display_name,
        "target_column": target_column,
        "feature_columns": processed_features,
        "classes": class_names,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "training_time_s": train_time,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4)
        },
        "confusion_matrix": cm_list,
        "per_class_metrics": per_class_metrics,
        "feature_importances": feature_importance_list,
        "sample_predictions": sample_preds,
        "summary": summary_bullets
    }
