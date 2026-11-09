import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from dashboard_utils import (
    calibration_bins, clean_reported_metrics, enrich_predictions, evaluation_metrics,
    expected_calibration_error, normalise_predictions, risk_coverage_curve, uncertainty_summary,
)


def observations():
    return pd.DataFrame({
        "firm_id": ["1", "2", "3", "4"],
        "year": [2023, 2024, 2024, 2025],
        "true_label": [0, 1, 0, 1],
        "predicted_probability": [0.1, 0.9, 0.6, 0.4],
        "mc_std": [0.01, 0.03, 0.12, 0.08],
    })


def test_metrics_and_confusion():
    df = normalise_predictions(observations())
    m = evaluation_metrics(df, 0.5)
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert (m["confusion_matrix"] == np.array([[1, 1], [1, 1]])).all()
    assert m["brier"] == pytest.approx(brier_score_loss(df.true_label, df.predicted_probability))
    assert len(calibration_bins(df.true_label, df.predicted_probability)) > 0
    assert 0 <= expected_calibration_error(df.true_label, df.predicted_probability) <= 1


def test_rename_and_variance():
    original = pd.DataFrame({"True Default": [0, 1], "Bayes Prob": [0.2, .8],
                             "Predictive Variance": [.0004, .04]})
    result = normalise_predictions(original)
    assert result["mc_std"].tolist() == pytest.approx([.02, .2])


def test_invalid_data_rejected():
    with pytest.raises(ValueError):
        normalise_predictions(pd.DataFrame({"true_label": [0, 2], "predicted_probability": [.2, .4]}))
    with pytest.raises(ValueError):
        normalise_predictions(pd.DataFrame({"true_label": [0, 1], "predicted_probability": [-.1, 1.1]}))
    with pytest.raises(ValueError):
        normalise_predictions(pd.DataFrame({"true_label": [0], "predicted_probability": [.1], "mc_std": [-1]}))


def test_enrichment_and_coverage():
    df = normalise_predictions(observations())
    rich = enrich_predictions(df, 0.5)
    assert rich.error.sum() == 2
    assert rich.uncertainty_category.isin(["Low", "Medium", "High"]).all()
    assert len(risk_coverage_curve(rich)) == 4
    summ = uncertainty_summary(rich)
    assert "error_detection_auc" in summ


def test_one_class_handling():
    df = normalise_predictions(pd.DataFrame({"true_label": [0, 0], "predicted_probability": [.2, .4]}))
    m = evaluation_metrics(df)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])
    assert m["nll"] >= 0


def test_missing_metrics_preserved():
    summary = clean_reported_metrics(pd.DataFrame({"model": ["A"], "roc_auc": [.9]}))
    assert np.isnan(summary.iloc[0]["nll"])
    assert summary.iloc[0]["roc_auc"] == .9
