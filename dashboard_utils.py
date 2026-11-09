"""Data validation and metrics for a firm-level credit-default Streamlit dashboard.

The application does not assume that a reported default_label has been constructed
using a forward, next-12-month horizon; that is the researcher's responsibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ALIASES = {
    "true_label": ["true_label", "true_default", "y_true", "y_test", "default_label", "target", "actual"],
    "predicted_probability": ["predicted_probability", "bayes_prob", "mc_mean", "mean_probability", "default_probability", "probability", "p_default", "y_prob"],
    "mc_std": ["mc_std", "bayes_std", "predictive_std", "predictive_standard_deviation", "uncertainty_std"],
    "predictive_variance": ["predictive_variance", "bayes_var", "mc_variance"],
    "firm_id": ["firm_id", "firm", "company_id"],
    "year": ["year", "prediction_year", "observation_year"],
    "model": ["model", "model_name"],
}

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "nll", "ece"]


def _clean_col(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalise_predictions(source: pd.DataFrame) -> pd.DataFrame:
    """Map documented aliases and reject malformed labels/probabilities.

    Optional uncertainty can be expressed as standard deviation or variance.
    Missing uncertainty is kept as missing; it is never fabricated.
    """
    if source.empty:
        raise ValueError("The prediction file has no rows.")
    df = source.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("The uploaded file has duplicate column names after normalisation.")
    for canonical, aliases in ALIASES.items():
        if canonical not in df.columns:
            matching = [c for c in aliases if c in df.columns]
            if matching:
                df = df.rename(columns={matching[0]: canonical})

    for col in ("true_label", "predicted_probability"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. See the included CSV template.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["true_label", "predicted_probability"]].isna().any().any():
        raise ValueError("Labels and predicted probabilities must be numeric and nonmissing on every row.")
    if not df["true_label"].isin([0, 1]).all():
        raise ValueError("true_label must contain only 0 (non-default) or 1 (default).")
    if not np.isfinite(df["predicted_probability"].to_numpy()).all():
        raise ValueError("All predicted probabilities must be finite.")
    if not df["predicted_probability"].between(0, 1).all():
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    df["true_label"] = df["true_label"].astype(int)

    if "mc_std" in df:
        df["mc_std"] = pd.to_numeric(df["mc_std"], errors="coerce")
        if df["mc_std"].notna().any() and ((df["mc_std"].dropna() < 0).any() or
                                            (~np.isfinite(df["mc_std"].dropna())).any()):
            raise ValueError("MC standard deviation must be finite and nonnegative.")
    elif "predictive_variance" in df:
        variance = pd.to_numeric(df["predictive_variance"], errors="coerce")
        if variance.notna().any() and ((variance.dropna() < 0).any() or
                                       (~np.isfinite(variance.dropna())).any()):
            raise ValueError("Predictive variance must be finite and nonnegative.")
        df["mc_std"] = np.sqrt(variance)
    else:
        df["mc_std"] = np.nan

    if "model" not in df:
        df["model"] = "Bayesian Bidirectional Transformer"
    else:
        df["model"] = df["model"].fillna("Unspecified model").astype(str)
    if "firm_id" in df:
        df["firm_id"] = df["firm_id"].astype("string")
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def calibration_bins(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bin calibration; ECE uses every nonempty bin."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be nonempty, equally sized arrays.")
    if n_bins < 2:
        raise ValueError("At least two bins are required.")
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append({
                "bin": b + 1,
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
                "bin_lower": b / n_bins,
                "bin_upper": (b + 1) / n_bins,
            })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    bins = calibration_bins(y_true, prob, n_bins)
    return float((bins["count"] * (bins["mean_probability"] - bins["observed_frequency"]).abs()).sum() / bins["count"].sum())


def evaluation_metrics(df: pd.DataFrame, threshold: float = 0.54, n_bins: int = 10) -> dict:
    y = df["true_label"].to_numpy(dtype=int)
    p = df["predicted_probability"].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    metrics = {
        "n": len(y),
        "positives": int(y.sum()),
        "default_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "ece": expected_calibration_error(y, p, n_bins),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]),
    }
    return metrics


def enrich_predictions(df: pd.DataFrame, threshold: float = 0.54,
                       low_risk: float = 0.33, high_risk: float = 0.66) -> pd.DataFrame:
    out = df.copy()
    p = out["predicted_probability"]
    out["predicted_label"] = (p >= threshold).astype(int)
    out["error"] = (out["predicted_label"] != out["true_label"]).astype(int)
    out["absolute_probability_error"] = (p - out["true_label"]).abs()
    out["risk_category"] = np.select(
        [p < low_risk, p < high_risk], ["Low", "Medium"], default="High"
    )
    # These categories are defined relative to THIS uploaded cohort only.
    out["uncertainty_category"] = "Not available"
    available = out["mc_std"].notna()
    if available.any():
        rank = out.loc[available, "mc_std"].rank(pct=True, method="average")
        out.loc[available, "uncertainty_category"] = np.select(
            [rank <= 1/3, rank <= 2/3], ["Low", "Medium"], default="High"
        )
    return out


def uncertainty_summary(df: pd.DataFrame) -> dict:
    """Uncertainty is a useful error signal only when supported by observations."""
    sub = df.dropna(subset=["mc_std"]).copy()
    if len(sub) < 2:
        return {}
    u = sub["mc_std"]
    error = sub["error"]
    vals = {
        "count": len(sub),
        "mean_std": float(u.mean()),
        "median_std": float(u.median()),
        "spearman_error": float(u.corr(error, method="spearman")),
        "spearman_abs_error": float(u.corr(sub["absolute_probability_error"], method="spearman")),
        "error_base_rate": float(error.mean()),
    }
    vals["error_detection_auc"] = (
        float(roc_auc_score(error, u)) if error.nunique() == 2 else np.nan
    )
    vals["error_detection_ap"] = (
        float(average_precision_score(error, u)) if error.nunique() == 2 else np.nan
    )
    return vals


def risk_coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Keep most-certain cases first and calculate retained misclassification risk."""
    sub = df.dropna(subset=["mc_std"]).sort_values("mc_std", ascending=True).reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(columns=["coverage", "error_rate"])
    k = np.arange(1, len(sub) + 1)
    return pd.DataFrame({"coverage": k / len(sub), "error_rate": np.cumsum(sub["error"].to_numpy()) / k})


def clean_reported_metrics(source: pd.DataFrame) -> pd.DataFrame:
    """Validate the optional metrics-table upload without filling missing scores."""
    df = source.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    if "model" not in df:
        raise ValueError("Model summary CSV needs a 'model' column.")
    for metric in METRIC_NAMES:
        if metric not in df:
            df[metric] = np.nan
        else:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
            if (df[metric].dropna().lt(0) | df[metric].dropna().gt(1)).any():
                raise ValueError(f"Metric '{metric}' must be in [0, 1] or blank.")
    df["model"] = df["model"].astype(str)
    if df["model"].duplicated().any():
        raise ValueError("Summary CSV has duplicate model names.")
    return df[["model", *METRIC_NAMES]]
