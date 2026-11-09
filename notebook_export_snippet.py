"""Copy this into a cell in the notebook after computing aligned final test predictions.

Required notebook objects:
    y_test        Ground-truth 0/1 labels for the held-out test sequences
    bayes_prob    MC-dropout mean predicted default probabilities
    bayes_std     MC-dropout standard deviation per test sequence
    firm_ids_test, years_test   Identifiers aligned to y_test

Output: bayesian_test_predictions.csv
"""
import numpy as np
import pandas as pd


def as_vector(name, values):
    result = np.asarray(values).reshape(-1)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return result


# Requires these five notebook variables to already exist and be aligned:
export_columns = {
    "firm_id": as_vector("firm_ids_test", firm_ids_test),
    "year": as_vector("years_test", years_test),
    "true_label": as_vector("y_test", y_test),
    "predicted_probability": as_vector("bayes_prob", bayes_prob),
    "mc_std": as_vector("bayes_std", bayes_std),
}
lengths = {name: len(values) for name, values in export_columns.items()}
if len(set(lengths.values())) != 1:
    raise ValueError(f"Misaligned arrays: {lengths}. Do NOT export until they match.")

export_columns["model"] = "Bayesian Bidirectional Transformer"
out = pd.DataFrame(export_columns)
if not out["true_label"].isin([0, 1]).all():
    raise ValueError("y_test must be binary [0,1].")
if not out["predicted_probability"].between(0, 1).all():
    raise ValueError("bayes_prob must be a probability between 0 and 1.")
if out["mc_std"].lt(0).any():
    raise ValueError("bayes_std must not contain negative values.")
out.to_csv("bayesian_test_predictions.csv", index=False)
print(f"Saved {len(out):,} aligned test predictions to bayesian_test_predictions.csv")
