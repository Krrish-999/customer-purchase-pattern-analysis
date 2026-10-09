"""Regression Module for Supermarket Data Mining Platform.
Supports: Linear Regression, Multiple Linear Regression, Polynomial Regression,
Decision Tree Regressor, Random Forest Regressor.
Calculates MAE, MSE, RMSE, R² Score, Actual vs Predicted scatter data, and feature importances.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_regression(df, algorithm, target_column, feature_columns, params=None):
    """
    Trains and evaluates a regression model.
    Returns metrics (MAE, MSE, RMSE, R²), actual vs predicted chart points, feature coefficients, and sample predictions.
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

    # Target must be numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric for regression.")

    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < 20:
        raise ValueError("Dataset has too few rows (<20) for regression.")

    sample_size = int(params.get("sample_size", 5000))
    if sample_size > 0 and len(clean_df) > sample_size:
        clean_df = clean_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    y = clean_df[target_column].astype(float).values

    # Prepare features
    X_df = clean_df[valid_features].copy()
    processed_features = []
    
    for col in valid_features:
        s = X_df[col]
        if pd.api.types.is_numeric_dtype(s):
            X_df[col] = s.fillna(s.mean() if s.count() > 0 else 0)
            processed_features.append(col)
        else:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(s.fillna("Missing").astype(str))
            processed_features.append(col)

    X = X_df[processed_features].values

    test_size = float(params.get("test_size", 0.2))
    test_size = max(0.1, min(0.5, test_size))
    random_state = int(params.get("random_state", 42))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    algo = str(algorithm).lower().strip()
    algo_display_name = "Regression"
    feature_importance_list = []
    poly_transformer = None

    if algo in ("linear", "multiple_linear"):
        model = LinearRegression()
        algo_display_name = "Linear Regression" if len(processed_features) == 1 else "Multiple Linear Regression"
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Coefficients
        for f, c in zip(processed_features, model.coef_):
            feature_importance_list.append({"feature": f, "importance": round(float(c), 4), "is_coef": True})

    elif algo == "polynomial":
        degree = int(params.get("poly_degree", 2))
        degree = max(2, min(4, degree))
        poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_transformer.fit_transform(X_train)
        X_test_poly = poly_transformer.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
        algo_display_name = f"Polynomial Regression (Degree {degree})"

        feature_names = poly_transformer.get_feature_names_out(processed_features)
        for f, c in zip(feature_names[:15], model.coef_[:15]):
            feature_importance_list.append({"feature": f, "importance": round(float(c), 4), "is_coef": True})

    elif algo == "decision_tree":
        max_depth = int(params.get("dt_max_depth", 6)) if params.get("dt_max_depth") else None
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        algo_display_name = f"Decision Tree Regressor (depth={max_depth if max_depth else 'None'})"

        for f, imp in zip(processed_features, model.feature_importances_):
            feature_importance_list.append({"feature": f, "importance": round(float(imp), 4), "is_coef": False})

    elif algo == "random_forest":
        n_estimators = int(params.get("rf_n_estimators", 50))
        max_depth = int(params.get("rf_max_depth", 6)) if params.get("rf_max_depth") else None
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        algo_display_name = f"Random Forest Regressor ({n_estimators} trees)"

        for f, imp in zip(processed_features, model.feature_importances_):
            feature_importance_list.append({"feature": f, "importance": round(float(imp), 4), "is_coef": False})

    else:
        raise ValueError(f"Unsupported regression algorithm '{algorithm}'.")

    train_time = round(time.time() - t0, 3)

    # Calculate metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Sample points for Actual vs Predicted scatter chart (up to 120 points for clean rendering)
    chart_points = []
    step = max(1, len(y_test) // 120)
    for i in range(0, len(y_test), step):
        act = float(y_test[i])
        prd = float(y_pred[i])
        chart_points.append({
            "x": round(act, 2),
            "y": round(prd, 2),
            "residual": round(act - prd, 2)
        })

    min_val = round(float(min(np.min(y_test), np.min(y_pred))), 2)
    max_val = round(float(max(np.max(y_test), np.max(y_pred))), 2)

    # Sample predictions table (first 25 test samples)
    sample_preds = []
    for i in range(min(25, len(y_test))):
        feats = {f: round(float(X_test[i][idx]), 2) for idx, f in enumerate(processed_features)}
        act_val = round(float(y_test[i]), 2)
        pred_val = round(float(y_pred[i]), 2)
        err = round(abs(act_val - pred_val), 2)
        pct_err = round((err / max(abs(act_val), 1e-4)) * 100, 1)
        sample_preds.append({
            "features": feats,
            "actual": act_val,
            "predicted": pred_val,
            "error": err,
            "pct_error": min(pct_err, 999.9)
        })

    summary_bullets = [
        f"Trained <b>{algo_display_name}</b> to predict <code>{target_column}</code> on {len(X_train):,} samples.",
        f"R² Score: <b>{r2:.4f}</b> (explains {max(0.0, r2) * 100:.1f}% of target variance).",
        f"Mean Absolute Error (MAE): <b>{mae:.2f}</b>, Root Mean Squared Error (RMSE): <b>{rmse:.2f}</b>."
    ]

    return {
        "algorithm": algo,
        "algorithm_name": algo_display_name,
        "target_column": target_column,
        "feature_columns": processed_features,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "training_time_s": train_time,
        "metrics": {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4)
        },
        "chart_points": chart_points,
        "reference_line": {"min": min_val, "max": max_val},
        "feature_importances": sorted(feature_importance_list, key=lambda x: abs(x["importance"]), reverse=True),
        "sample_predictions": sample_preds,
        "summary": summary_bullets
    }
