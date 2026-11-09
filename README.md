# Credit Default Research Observatory — Streamlit

A multi-page-style interactive dashboard for a firm-level credit-default research dissertation featuring a Bayesian Bidirectional Transformer with Monte Carlo dropout. The application works **immediately** with a *draft, transcribed model-metrics table* and becomes a genuine test-set evaluation dashboard when you upload your real prediction data.

## Features

- Performance overview, model comparison and filters
- Interactive ROC and precision–recall curves (when both classes are present)
- Reliability diagram, Brier score, negative log likelihood, ECE and confusion matrix
- MC-dropout uncertainty analysis: standard deviation versus error, error-detection AUC/AP, uncertainty deciles and risk–coverage curve
- Firm-level explorer with probability threshold, risk-band filtering, firm ID search, and CSV export
- Methodology checklist, data validation and explicit missing-value handling

**Not included:** Model training, inference from raw financial statements, a real-time database, or a deployed/public URL. This app uses model-generated test predictions you supply.

## Run locally

Install Python 3.10 or later. Unzip the project and open a terminal in the folder containing `streamlit_app.py`.

```bash
python -m pip install -rrequirements.txt
python -m streamlit run streamlit_app.py
```

A local web browser page opens automatically (usually `http://localhost:8501`).

## Upload your data

Start with `data/predictions_template.csv`. You must provide one row per model prediction:

```csv
firm_id,year,true_label,predicted_probability,mc_std,model
```

- `true_label`: **0 or 1**, constructed for the correct prediction horizon.
- `predicted_probability`: continuous probability of default in [0, 1], preferably the mean of MC-dropout passes for the Bayesian model.
- `mc_std`: standard deviation of those stochastic probabilities (optional, but required for the uncertainty plots).
- `firm_id`, `year`, `model` are optional for analysis but recommended. If `model` is omitted, the default is `Bayesian Bidirectional Transformer`.
- Accepted aliases include `default_label`, `y_test`, `true_default` for label; `bayes_prob`, `mc_mean` for probability; `bayes_std` for uncertainty. You may also provide nonnegative `predictive_variance` instead of standard deviation.
- **Never upload test data with labels or probabilities from different rows or different models misaligned.**

The dashboard calculates metrics for the selected model and applies your chosen threshold. It does not overwrite the draft scorecard automatically. For new scorecards, upload a CSV with `model` and any of `accuracy,precision,recall,f1,roc_auc,pr_auc,brier,nll,ece`.

## Where the numbers came from

`data/model_metrics.csv` is a **draft, manually transcribed** set of reported results from previous research discussions, **not a verified reproducibility artifact**. Several baseline scores have differed between report versions, so the file must be reconciled against final saved test-set predictions and model logs before publication. Logistic Regression and deterministic Transformer F1 values were left blank because earlier reported F1 values were mathematically inconsistent with their corresponding precision and recall. Missing NLL/ECE baseline values remain blank. The proposed model entry is likewise subject to final verification.

The uncertainty tab shows a small set of previously reported uncertainty statistics if no per-row MC dropout values are supplied. These are displayed as reported values only; they are not recalculated and may refer to a different threshold or test split.

## Export directly from your existing notebook

If your notebook has `bayes_prob`, `bayes_std`, `y_test`, `firm_ids_test` and `years_test`, copy the short snippet in `notebook_export_snippet.py` into the notebook. Run it only after all arrays are aligned to the final test partition. It produces `bayesian_test_predictions.csv` for upload to this dashboard. The helper checks that every array has the same number of records.

## Private deployment

1. Push the app to a **private GitHub repository**; do not commit sensitive raw data, uploaded predictions or secrets. The supplied project includes only a blank prediction CSV schema.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), create an app, and select `streamlit_app.py` as entrypoint.
3. Enable access to private repos if needed; check the app stays **private** and invite only authorized viewers. Community Cloud currently limits private apps on the free tier.
4. Treat user-uploaded data as being processed by your deployed cloud app. For confidential firm-level data, use an approved deployment environment and access controls.

Streamlit documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app

## Files

```text
streamlit_app.py                  Streamlit UI
dashboard_utils.py               Dataset validation and evaluation functions
data/model_metrics.csv           DRAFT results; replace/verify before publication
data/predictions_template.csv    Header-only input schema
requirements.txt                 Dependencies
.streamlit/config.toml           Visual theme
tests/test_utils.py              Automated evaluation unit tests
notebook_export_snippet.py        Optional Jupyter export helper
```

## Scope and methodological caution

Use **firm-group-held-out** test predictions; ensure labels truly refer to the *subsequent 12-month period*, and fit scalers/imputers on training data only. The dashboard does not assert rolling-origin evaluation, because a firm-group split alone is not chronological validation. MC-dropout variability is a measure of predictive variation, not proof of fully identified epistemic uncertainty. Risk bands (0.33 / 0.66 defaults) are UI-only exploratory categories and are not validated regulatory cutoffs.
