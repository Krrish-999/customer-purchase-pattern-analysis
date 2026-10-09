"""Data Preprocessing Module for Supermarket Data Mining Platform.
Handles missing values, duplicates, type conversions, categorical encoding,
feature scaling, outlier detection/handling, data cleaning, and feature selection.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold

def sanitize_value(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) or v is pd.NaT:
        return None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S") if (v.hour or v.minute or v.second) else v.strftime("%Y-%m-%d")
    return str(v)

def get_dataset_profile(df):
    """Computes comprehensive profile of the dataset: shape, column stats, types, nulls."""
    if df is None or len(df) == 0:
        return {"rows": 0, "columns": 0, "total_missing": 0, "duplicate_rows": 0, "columns_info": []}
    
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    cols_info = []

    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        dtype_str = str(s.dtype)
        
        is_num = pd.api.types.is_numeric_dtype(s)
        is_date = pd.api.types.is_datetime64_any_dtype(s)
        is_bool = pd.api.types.is_bool_dtype(s)
        
        if is_num:
            group = "numeric"
        elif is_date:
            group = "datetime"
        elif is_bool:
            group = "boolean"
        else:
            group = "categorical"

        samples = [sanitize_value(x) for x in s.dropna().unique()[:3]]
        
        col_stat = {
            "name": str(col),
            "dtype": dtype_str,
            "group": group,
            "missing": n_missing,
            "missing_pct": round((n_missing / len(df)) * 100, 2),
            "unique": n_unique,
            "samples": samples
        }

        if is_num and s.count() > 0:
            col_stat["min"] = sanitize_value(s.min())
            col_stat["max"] = sanitize_value(s.max())
            col_stat["mean"] = sanitize_value(s.mean())
            col_stat["median"] = sanitize_value(s.median())
            col_stat["std"] = sanitize_value(s.std())
        
        cols_info.append(col_stat)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "columns_info": cols_info
    }

def apply_preprocessing_step(df, op, params=None):
    """
    Applies a single preprocessing operation to df.
    Returns (new_df, message, changes_dict).
    """
    if params is None:
        params = {}
    
    work_df = df.copy()
    initial_rows = len(work_df)
    initial_cols = len(work_df.columns)
    initial_missing = int(work_df.isna().sum().sum())
    
    op = str(op).lower()

    if op == "impute_missing":
        column = params.get("column", "all")
        strategy = params.get("strategy", "mean").lower()
        fill_val = params.get("fill_value", "")

        target_cols = [column] if column != "all" else list(work_df.columns)
        changes = []

        if strategy == "drop_rows":
            if column == "all":
                work_df = work_df.dropna().reset_index(drop=True)
            else:
                work_df = work_df.dropna(subset=[column]).reset_index(drop=True)
            dropped = initial_rows - len(work_df)
            msg = f"Dropped {dropped:,} row(s) containing missing values in {column}."
        else:
            for c in target_cols:
                s = work_df[c]
                n_na = int(s.isna().sum())
                if n_na == 0:
                    continue
                
                is_num = pd.api.types.is_numeric_dtype(s)
                if strategy == "mean" and is_num:
                    val = s.mean()
                    work_df[c] = s.fillna(val)
                    changes.append(f"{c}: filled {n_na} missing with mean ({val:.2f})")
                elif strategy == "median" and is_num:
                    val = s.median()
                    work_df[c] = s.fillna(val)
                    changes.append(f"{c}: filled {n_na} missing with median ({val:.2f})")
                elif strategy == "mode":
                    mode_val = s.mode(dropna=True)
                    val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                    work_df[c] = s.fillna(val)
                    changes.append(f"{c}: filled {n_na} missing with mode ('{val}')")
                elif strategy == "constant":
                    val = fill_val
                    if is_num:
                        try:
                            val = float(fill_val)
                        except (ValueError, TypeError):
                            val = 0
                    work_df[c] = s.fillna(val)
                    changes.append(f"{c}: filled {n_na} missing with constant ({val})")
            
            msg = "; ".join(changes) if changes else f"No missing values found in selected column(s)."

    elif op == "remove_duplicates":
        subset = params.get("subset")
        if subset and isinstance(subset, list) and len(subset) > 0:
            valid_sub = [c for c in subset if c in work_df.columns]
            work_df = work_df.drop_duplicates(subset=valid_sub).reset_index(drop=True)
            dropped = initial_rows - len(work_df)
            msg = f"Removed {dropped:,} duplicate row(s) based on columns: {', '.join(valid_sub)}."
        else:
            work_df = work_df.drop_duplicates().reset_index(drop=True)
            dropped = initial_rows - len(work_df)
            msg = f"Removed {dropped:,} duplicate row(s) across all columns."

    elif op == "convert_type":
        column = params.get("column")
        target_type = params.get("target_type", "numeric").lower()
        if not column or column not in work_df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if target_type == "numeric":
            work_df[column] = pd.to_numeric(work_df[column], errors="coerce")
            msg = f"Converted column '{column}' to Numeric (errors coerced to NaN)."
        elif target_type in ("string", "category"):
            work_df[column] = work_df[column].astype(str)
            msg = f"Converted column '{column}' to String/Category."
        elif target_type == "datetime":
            work_df[column] = pd.to_datetime(work_df[column], errors="coerce")
            msg = f"Converted column '{column}' to DateTime (errors coerced to NaT)."
        else:
            raise ValueError(f"Unsupported target type '{target_type}'.")

    elif op == "encode_categorical":
        columns = params.get("columns", [])
        method = params.get("method", "label").lower()
        drop_first = bool(params.get("drop_first", False))

        if isinstance(columns, str):
            columns = [columns]
        target_cols = [c for c in columns if c in work_df.columns]
        if not target_cols:
            raise ValueError("No valid columns selected for encoding.")

        encoded_names = []
        if method == "label":
            for c in target_cols:
                le = LabelEncoder()
                work_df[c] = le.fit_transform(work_df[c].astype(str))
                encoded_names.append(c)
            msg = f"Label Encoded column(s): {', '.join(encoded_names)} into integer classes."
        elif method == "one_hot":
            work_df = pd.get_dummies(work_df, columns=target_cols, drop_first=drop_first, dtype=int)
            msg = f"One-Hot Encoded column(s): {', '.join(target_cols)}. Dataset now has {len(work_df.columns)} columns."
        else:
            raise ValueError(f"Unknown encoding method '{method}'.")

    elif op == "scale_features":
        columns = params.get("columns", [])
        method = params.get("method", "min_max").lower()

        if isinstance(columns, str):
            columns = [columns]
        target_cols = [c for c in columns if c in work_df.columns and pd.api.types.is_numeric_dtype(work_df[c])]
        if not target_cols:
            raise ValueError("Select at least one numerical column to scale.")

        if method == "min_max":
            scaler = MinMaxScaler()
            scaler_name = "Min-Max Normalization [0, 1]"
        elif method == "standard":
            scaler = StandardScaler()
            scaler_name = "Standardization (Z-score, mean=0, std=1)"
        elif method == "robust":
            scaler = RobustScaler()
            scaler_name = "Robust Scaling (IQR based)"
        else:
            raise ValueError(f"Unknown scaling method '{method}'.")

        scaled_vals = scaler.fit_transform(work_df[target_cols].fillna(work_df[target_cols].mean()))
        for i, c in enumerate(target_cols):
            work_df[c] = np.round(scaled_vals[:, i], 4)

        msg = f"Applied {scaler_name} on column(s): {', '.join(target_cols)}."

    elif op == "handle_outliers":
        columns = params.get("columns", [])
        method = params.get("method", "iqr").lower()
        action = params.get("action", "clip").lower()  # 'clip' or 'remove'
        threshold = float(params.get("threshold", 1.5 if method == "iqr" else 3.0))

        if isinstance(columns, str):
            columns = [columns]
        target_cols = [c for c in columns if c in work_df.columns and pd.api.types.is_numeric_dtype(work_df[c])]
        if not target_cols:
            raise ValueError("Select at least one numerical column for outlier detection.")

        total_outliers = 0
        if method == "iqr":
            if action == "clip":
                for c in target_cols:
                    q1 = work_df[c].quantile(0.25)
                    q3 = work_df[c].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    out_mask = (work_df[c] < lower) | (work_df[c] > upper)
                    n_out = int(out_mask.sum())
                    total_outliers += n_out
                    work_df[c] = work_df[c].clip(lower=lower, upper=upper)
                msg = f"Capped {total_outliers:,} outlier values using IQR method (bounds: Q1/Q3 ± {threshold}*IQR) on {', '.join(target_cols)}."
            else: # remove
                mask = pd.Series(True, index=work_df.index)
                for c in target_cols:
                    q1 = work_df[c].quantile(0.25)
                    q3 = work_df[c].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    mask &= (work_df[c] >= lower) & (work_df[c] <= upper)
                work_df = work_df[mask].reset_index(drop=True)
                dropped = initial_rows - len(work_df)
                msg = f"Removed {dropped:,} outlier row(s) using IQR method on {', '.join(target_cols)}."
        elif method == "z_score":
            if action == "clip":
                for c in target_cols:
                    mean = work_df[c].mean()
                    std = work_df[c].std()
                    if std > 0:
                        lower = mean - threshold * std
                        upper = mean + threshold * std
                        out_mask = (work_df[c] < lower) | (work_df[c] > upper)
                        total_outliers += int(out_mask.sum())
                        work_df[c] = work_df[c].clip(lower=lower, upper=upper)
                msg = f"Capped {total_outliers:,} outlier values using Z-score method (|z| > {threshold}) on {', '.join(target_cols)}."
            else:
                mask = pd.Series(True, index=work_df.index)
                for c in target_cols:
                    mean = work_df[c].mean()
                    std = work_df[c].std()
                    if std > 0:
                        z = (work_df[c] - mean) / std
                        mask &= z.abs() <= threshold
                work_df = work_df[mask].reset_index(drop=True)
                dropped = initial_rows - len(work_df)
                msg = f"Removed {dropped:,} outlier row(s) using Z-score (|z| > {threshold}) on {', '.join(target_cols)}."
        else:
            raise ValueError(f"Unknown outlier method '{method}'.")

    elif op == "drop_columns":
        columns = params.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]
        to_drop = [c for c in columns if c in work_df.columns]
        if not to_drop:
            raise ValueError("No valid columns selected to drop.")
        if len(to_drop) >= len(work_df.columns):
            raise ValueError("Cannot drop all columns from dataset.")
        work_df = work_df.drop(columns=to_drop)
        msg = f"Dropped column(s): {', '.join(to_drop)}. Remaining: {len(work_df.columns)} columns."

    elif op == "feature_selection":
        method = params.get("method", "select_k_best").lower()
        target_col = params.get("target_column")
        k = int(params.get("k", 3))

        num_cols = [c for c in work_df.columns if pd.api.types.is_numeric_dtype(work_df[c]) and c != target_col]
        if len(num_cols) <= 1:
            raise ValueError("Feature selection requires at least 2 numerical candidate features.")

        if method == "variance_threshold":
            thresh = float(params.get("threshold", 0.01))
            selector = VarianceThreshold(threshold=thresh)
            X = work_df[num_cols].fillna(work_df[num_cols].mean())
            selector.fit(X)
            kept = [col for col, keep in zip(num_cols, selector.get_support()) if keep]
            dropped = [c for c in num_cols if c not in kept]
            if not kept:
                raise ValueError("Variance threshold dropped all features. Try a lower threshold.")
            non_candidate_cols = [c for c in work_df.columns if c not in num_cols]
            work_df = work_df[non_candidate_cols + kept]
            msg = f"Variance threshold ({thresh}): kept {len(kept)} feature(s), dropped {len(dropped)} low-variance feature(s) ({', '.join(dropped) if dropped else 'none'})."
        
        elif method == "select_k_best":
            if not target_col or target_col not in work_df.columns:
                raise ValueError("Target column is required for SelectKBest.")
            k = min(k, len(num_cols))
            X = work_df[num_cols].fillna(work_df[num_cols].mean())
            y = work_df[target_col]
            
            # If target is categorical, use f_classif, else f_regression
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
                selector = SelectKBest(score_func=f_regression, k=k)
            else:
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                selector = SelectKBest(score_func=f_classif, k=k)
            
            selector.fit(X, y if 'y_enc' not in locals() else y_enc)
            kept = [col for col, keep in zip(num_cols, selector.get_support()) if keep]
            dropped = [c for c in num_cols if c not in kept]
            non_candidate_cols = [c for c in work_df.columns if c not in num_cols]
            work_df = work_df[non_candidate_cols + kept]
            msg = f"SelectKBest (k={k}) selected top features: {', '.join(kept)}."
        
        else:
            raise ValueError(f"Unknown feature selection method '{method}'.")

    elif op == "clean_text":
        columns = params.get("columns", [])
        trim = bool(params.get("trim", True))
        lower = bool(params.get("lowercase", False))
        
        if isinstance(columns, str):
            columns = [columns]
        target_cols = [c for c in columns if c in work_df.columns and not pd.api.types.is_numeric_dtype(work_df[c])]
        for c in target_cols:
            if trim:
                work_df[c] = work_df[c].astype(str).str.strip()
            if lower:
                work_df[c] = work_df[c].astype(str).str.lower()
        msg = f"Text cleaning applied on: {', '.join(target_cols)} (trim: {trim}, lowercase: {lower})."
    else:
        raise ValueError(f"Unrecognized preprocessing operation: '{op}'")

    changes = {
        "operation": op,
        "summary": msg,
        "rows_before": initial_rows,
        "rows_after": len(work_df),
        "cols_before": initial_cols,
        "cols_after": len(work_df.columns),
        "missing_before": initial_missing,
        "missing_after": int(work_df.isna().sum().sum())
    }

    return work_df, msg, changes
