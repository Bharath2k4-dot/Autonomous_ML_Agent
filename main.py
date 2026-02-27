import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------
# Helpers: clean prints
# ----------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ----------------------------
# Step 1: Load data
# ----------------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    df = pd.read_csv(path)
    if df.shape[0] < 20:
        log("Warning: dataset has very few rows. Model quality will be weak.")
    return df


# ----------------------------
# Step 2: Target detection (simple + practical)
# ----------------------------
def guess_target_column(df: pd.DataFrame) -> str:
    """
    Heuristic:
    - Prefer columns named like: target, label, churn, y, outcome, class
    - Otherwise: last column
    """
    candidates = ["target", "label", "churn", "y", "outcome", "class"]
    lowered = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lowered:
            return lowered[key]
    return df.columns[-1]


# ----------------------------
# Step 3: Task type detection
# ----------------------------
def detect_task_type(y: pd.Series) -> str:
    """
    classification if:
    - y is object/bool
    - OR unique values small (<= 15) AND integers-like
    regression otherwise
    """
    if y.dtype == "bool" or y.dtype == "object":
        return "classification"

    y_no_na = y.dropna()
    if y_no_na.empty:
        return "regression"

    nunique = y_no_na.nunique()

    # if looks like classes
    if nunique <= 15:
        # if all values are integers (or very close)
        vals = y_no_na.values
        if np.all(np.isclose(vals, np.round(vals))):
            return "classification"

    return "regression"


# ----------------------------
# Step 4: Build preprocessing
# ----------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ----------------------------
# Step 5: Candidate models
# ----------------------------
def get_candidate_models(task_type: str):
    if task_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        }


# ----------------------------
# Step 6: Evaluate
# ----------------------------
def evaluate_classification(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def evaluate_regression(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def choose_best_model(task_type: str, scores: dict) -> str:
    """
    For classification: maximize f1, tie-break with accuracy
    For regression: minimize rmse, tie-break with mae
    """
    if task_type == "classification":
        return sorted(
            scores.keys(),
            key=lambda k: (scores[k]["f1"], scores[k]["accuracy"]),
            reverse=True,
        )[0]
    else:
        return sorted(
            scores.keys(),
            key=lambda k: (scores[k]["rmse"], scores[k]["mae"]),
        )[0]


# ----------------------------
# Step 7: Business summary (simple, interview-friendly)
# ----------------------------
def build_summary(task_type: str, target: str, best_model_name: str, best_metrics: dict) -> str:
    if task_type == "classification":
        return (
            f"Task: Classification\n"
            f"Target column: {target}\n"
            f"Best model: {best_model_name}\n"
            f"Key metrics:\n"
            f"  - Accuracy: {best_metrics['accuracy']:.4f}\n"
            f"  - Weighted F1: {best_metrics['f1']:.4f}\n\n"
            f"Business use: This model can be used to automatically classify outcomes for new records, "
            f"helping teams take faster actions (e.g., churn prevention, risk flags, approvals)."
        )
    else:
        return (
            f"Task: Regression\n"
            f"Target column: {target}\n"
            f"Best model: {best_model_name}\n"
            f"Key metrics:\n"
            f"  - RMSE: {best_metrics['rmse']:.4f}\n"
            f"  - MAE: {best_metrics['mae']:.4f}\n"
            f"  - R2: {best_metrics['r2']:.4f}\n\n"
            f"Business use: This model can predict continuous values for new records, enabling forecasting "
            f"and better planning (e.g., sales prediction, demand estimation, pricing)."
        )


# ----------------------------
# Main Agent Orchestrator
# ----------------------------
def run_agent(data_path: str, target: str | None) -> None:
    log("Loading dataset...")
    df = load_csv(data_path)

    if target is None:
        target = guess_target_column(df)
        log(f"No target provided. Auto-selected target column: '{target}'")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # Basic cleaning: drop fully empty rows
    df = df.dropna(how="all")

    X = df.drop(columns=[target])
    y = df[target]

    # Detect task
    task_type = detect_task_type(y)
    log(f"Detected task type: {task_type}")

    # Build preprocessing
    preprocessor = build_preprocessor(X)

    # Train/test split
    log("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if task_type == "classification" and y.nunique() <= 50 else None,
    )

    # Candidate models
    models = get_candidate_models(task_type)
    scores = {}
    fitted_pipelines = {}

    log(f"Training {len(models)} candidate models...")
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)

        if task_type == "classification":
            metrics = evaluate_classification(y_test, preds)
        else:
            # regression predictions must be numeric
            preds = np.array(preds, dtype=float)
            y_true = np.array(y_test, dtype=float)
            metrics = evaluate_regression(y_true, preds)

        scores[name] = metrics
        fitted_pipelines[name] = pipe
        log(f"Model: {name} | Metrics: {metrics}")

    # Choose best
    best_name = choose_best_model(task_type, scores)
    best_pipe = fitted_pipelines[best_name]
    best_metrics = scores[best_name]
    log(f"Best model selected: {best_name}")

    # Save outputs
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = os.path.join("models", "best_model.joblib")
    joblib.dump(best_pipe, model_path)
    log(f"Saved best model pipeline to: {model_path}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "target": target,
        "task_type": task_type,
        "candidate_scores": scores,
        "best_model": best_name,
        "best_metrics": best_metrics,
    }

    json_path = os.path.join("reports", "run_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Saved report JSON to: {json_path}")

    summary_text = build_summary(task_type, target, best_name, best_metrics)
    txt_path = os.path.join("reports", "run_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log(f"Saved summary TXT to: {txt_path}")

    log("\nDONE. Your autonomous ML lifecycle agent executed end-to-end.")


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous ML Lifecycle Agent (Version 1)")
    parser.add_argument("--data", required=True, help="Path to CSV file (example: data/mydata.csv)")
    parser.add_argument("--target", required=False, help="Target column name (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_agent(data_path=args.data, target=args.target)