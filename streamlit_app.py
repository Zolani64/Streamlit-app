"""Streamlit dashboard for firm-level default modelling and MC-dropout research.

Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from dashboard_utils import (
    METRIC_NAMES,
    calibration_bins,
    clean_reported_metrics,
    enrich_predictions,
    evaluation_metrics,
    normalise_predictions,
    risk_coverage_curve,
    uncertainty_summary,
)

ROOT = Path(__file__).resolve().parent
PROPOSED = "Bayesian Bidirectional Transformer"
LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-score", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
    "brier": "Brier score ↓", "nll": "Log loss ↓", "ece": "ECE ↓",
}
PALETTE = ["#2DD4BF", "#60A5FA", "#FBBF24", "#A78BFA", "#F87171", "#94A3B8"]

st.set_page_config(page_title="Credit Default | Research Observatory", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] {background: linear-gradient(130deg, #0B1220 0%, #0E1829 55%, #101C30 100%);}
  [data-testid="stSidebar"] {background:#111B2C;}
  .block-container {padding-top:2.0rem; max-width:1400px;}
  .eyebrow {color:#2DD4BF;letter-spacing:.17em;text-transform:uppercase;font-size:0.75rem;font-weight:750;}
  .hero {font-size:2.25rem;font-weight:780;line-height:1.16;margin:.30rem 0 .2rem;letter-spacing:-.025em;}
  .subtle {color:#9CAFC2;font-size:.95rem;margin-bottom:1rem;}
  [data-testid="stMetric"] {background:#172439;border:1px solid #2A3A51;padding:16px 18px;border-radius:14px;}
  [data-testid="stMetricLabel"] {color:#AFC1D7;}
  [data-testid="stMetricValue"] {color:#EAFEF9;}
  div[data-testid="stTabs"] button {font-weight:600;}
  .disclaimer {background:#18283A;border:1px solid #32475B;border-radius:12px;padding:13px 18px;color:#BACDDD;font-size:.87rem;}
  .divider {height:1px;background:#26364C;margin:14px 0;}
</style>
""", unsafe_allow_html=True)


def plot(fig, key: str | None = None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1"),
        margin=dict(l=10, r=12, t=48, b=22),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-.19),
        hoverlabel=dict(bgcolor="#152239"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#26374B", zerolinecolor="#26374B")
    fig.update_yaxes(showgrid=True, gridcolor="#26374B", zerolinecolor="#26374B")
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def fmt(value, decimals: int = 3) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{decimals}f}"


def percent(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


def display_metrics(df: pd.DataFrame):
    styled = df.rename(columns=LABELS).copy().set_index("model")
    st.dataframe(styled.style.format({c: "{:.4f}" for c in styled.columns}, na_rep="—"), width="stretch")


# ---- Inputs / state ----
with st.sidebar:
    st.markdown("### Research controls")
    st.caption("Reported results are draft transcriptions. Uploaded predictions give recalculated metrics.")
    threshold = st.slider("Default decision threshold", min_value=0.05, max_value=0.95, value=0.54, step=0.01,
                          help="Threshold for binary classification. ROC-AUC, PR-AUC, Brier, NLL and ECE do not depend on it.")
    n_bins = st.slider("Calibration bins", 5, 20, 10, 1)
    st.divider()
    st.markdown("#### Load your data")
    predictions_upload = st.file_uploader("Prediction-level CSV", type=["csv"], key="prediction_csv",
                                           help="At least true_label and predicted_probability; optional mc_std, firm_id, year, model.")
    metrics_upload = st.file_uploader("Model-metrics CSV (optional)", type=["csv"], key="metrics_csv")
    st.caption("Upload files for this session. The application does not save them to disk.")
    st.download_button("Get prediction CSV template", data=(ROOT / "data/predictions_template.csv").read_bytes(),
                       file_name="predictions_template.csv", mime="text/csv", width="stretch")
    st.download_button("Get metrics CSV template", data=(ROOT / "data/model_metrics.csv").read_bytes(),
                       file_name="model_metrics.csv", mime="text/csv", width="stretch")

metrics_source = "Reported draft summary"
try:
    if metrics_upload is None:
        summary = clean_reported_metrics(pd.read_csv(ROOT / "data/model_metrics.csv"))
    else:
        summary = clean_reported_metrics(pd.read_csv(metrics_upload))
        metrics_source = "Uploaded metrics file"
except Exception as exc:
    st.error(f"Model metrics could not be loaded: {exc}")
    st.stop()

prediction_df = None
if predictions_upload is not None:
    try:
        prediction_df = normalise_predictions(pd.read_csv(predictions_upload, dtype=str))
    except Exception as exc:
        st.error(f"Prediction CSV could not be loaded: {exc}")
        st.info("Use the prediction template. Required: true_label (0/1), predicted_probability (0–1).")

selected = None
calc = None
rich = None
if prediction_df is not None:
    with st.sidebar:
        selected_model = st.selectbox("Uploaded prediction model", sorted(prediction_df["model"].unique().tolist()))
    selected = prediction_df.loc[prediction_df["model"] == selected_model].copy()
    calc = evaluation_metrics(selected, threshold, n_bins)
    rich = enrich_predictions(selected, threshold)

st.markdown('<div class="eyebrow">Research observatory · Firm-level credit risk</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Credit Default Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Bidirectional Transformer · Baseline benchmarking · Calibration · Monte Carlo dropout</div>', unsafe_allow_html=True)

if predictions_upload is None:
    st.info("Showing **reported draft metrics**, not recalculated model outputs. Upload a test-set predictions CSV to unlock calibration curves, confusion matrices, and firm-level uncertainty analysis.")
else:
    st.success(f"Uploaded predictions: {len(prediction_df):,} rows across {prediction_df['model'].nunique()} model(s). Selected: {selected_model} ({len(selected):,} rows).") if prediction_df is not None else None

if metrics_source == "Reported draft summary":
    st.caption("Baseline values have appeared in different draft versions of the report. Reconcile the final model outputs before using these figures in a thesis or presentation.")

pages = st.tabs(["Overview", "Model comparison", "Calibration & diagnostics", "MC-dropout uncertainty", "Firm explorer", "Data & methodology"])

with pages[0]:
    st.subheader("Performance at a glance")
    if calc:
        values = [
            ("Test records", f"{calc['n']:,}"), ("ROC-AUC", fmt(calc["roc_auc"], 4)),
            ("PR-AUC", fmt(calc["pr_auc"], 4)), ("F1-score", fmt(calc["f1"], 4)),
        ]
    else:
        row = summary[summary["model"] == PROPOSED]
        row = row.iloc[0].to_dict() if not row.empty else {}
        values = [("Models", str(len(summary))), ("Reported ROC-AUC", fmt(row.get("roc_auc"), 4)),
                  ("Reported PR-AUC", fmt(row.get("pr_auc"), 4)), ("Reported F1-score", fmt(row.get("f1"), 4))]
    cols = st.columns(4)
    for col, (name, value) in zip(cols, values):
        col.metric(name, value)

    left, right = st.columns([1.45, 1.0])
    with left:
        st.markdown("#### How models compare")
        metric = st.selectbox("Primary comparison metric", ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy", "brier"],
                              format_func=lambda m: LABELS[m], key="home_metric")
        chart_data = summary.dropna(subset=[metric]).sort_values(metric, ascending=(metric == "brier"))
        fig = px.bar(chart_data, x=metric, y="model", orientation="h", color="model",
                     color_discrete_sequence=PALETTE, text=metric)
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(title=LABELS[metric], xaxis_title=LABELS[metric], yaxis_title=None,
                          showlegend=False, height=360)
        fig.update_xaxes(range=[0, max(.15, float(chart_data[metric].max()) * 1.13)] if len(chart_data) else [0, 1])
        plot(fig, "home_bar")
    with right:
        st.markdown("#### Research focus")
        st.markdown("""
        **RQ1 · Temporal modelling**  
        Contrast a Bidirectional Transformer with recurrent and nonsequential baselines.

        **RQ2 · Predictive uncertainty**  
        Test whether MC-dropout variability is associated with incorrect predictions.

        **RQ3 · Calibration**  
        Examine Brier score, log loss and expected calibration error.

        **RQ4 · Baseline comparison**  
        Compare discrimination and classification performance on the same held-out data.
        """)
        st.markdown('<div class="disclaimer">Uncertainty quantification is not itself proof of improved predictive accuracy or an improvement over deterministic models. That requires an appropriate like-for-like comparison.</div>', unsafe_allow_html=True)
    if calc:
        st.markdown("#### Uploaded test-set summary")
        q = st.columns(4)
        q[0].metric("Accuracy", percent(calc["accuracy"]))
        q[1].metric("Recall", percent(calc["recall"]))
        q[2].metric("Brier score", fmt(calc["brier"], 4))
        q[3].metric("ECE", fmt(calc["ece"], 4))

with pages[1]:
    st.subheader("Benchmark models")
    st.caption(f"Source: {metrics_source}. A dash means the metric was not supplied. Lower is better for Brier, log loss, and ECE.")
    display_metrics(summary)
    st.markdown("#### Side-by-side comparison")
    first, second = st.columns([1, 2])
    with first:
        chosen_models = st.multiselect("Models", summary["model"].tolist(),
                                      default=summary["model"].tolist()[:4], key="compare_models")
        chosen_metrics = st.multiselect("Metrics", ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"],
                                       default=["roc_auc", "pr_auc", "f1"], format_func=lambda k: LABELS[k])
    with second:
        if chosen_models and chosen_metrics:
            mdata = summary[summary["model"].isin(chosen_models)].melt(
                id_vars="model", value_vars=chosen_metrics, var_name="metric", value_name="score").dropna()
            mdata["metric"] = mdata["metric"].map(LABELS)
            fig = px.bar(mdata, x="metric", y="score", color="model", barmode="group",
                         color_discrete_sequence=PALETTE, title="Reported metrics by model")
            fig.update_layout(yaxis_range=[0, 1.07], height=380, xaxis_title=None, yaxis_title="Score")
            plot(fig, "benchmark_bar")
        else:
            st.info("Select at least one model and metric.")
    if calc:
        st.markdown("#### Recalculated score for selected uploaded predictions")
        live = {name: calc[name] for name in METRIC_NAMES}
        live["model"] = selected_model + " · uploaded"
        st.dataframe(pd.DataFrame([live]).rename(columns=LABELS).set_index("model").style.format("{:.4f}", na_rep="—"),
                     width="stretch")
        st.caption("Only the selected uploaded model is recalculated. The reported benchmark table is not silently overwritten.")

with pages[2]:
    st.subheader("Calibration & model diagnostics")
    if not calc:
        st.info("Upload prediction-level records to display empirical reliability diagrams, ROC/PR curves and a confusion matrix.")
        st.markdown("#### Available reported calibration measures")
        st.dataframe(summary[["model", "brier", "nll", "ece"]].rename(columns=LABELS).set_index("model").style.format("{:.4f}", na_rep="—"), width="stretch")
    else:
        cards = st.columns(4)
        for col, name, value in zip(cards, ["Brier score", "Log loss", "ECE", "ROC-AUC"],
                                    [calc["brier"], calc["nll"], calc["ece"], calc["roc_auc"]]):
            col.metric(name, fmt(value, 4))
        y = selected["true_label"].to_numpy()
        p = selected["predicted_probability"].to_numpy()
        bins = calibration_bins(y, p, n_bins)
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                     line=dict(color="#64748B", dash="dash")))
            fig.add_trace(go.Scatter(x=bins["mean_probability"], y=bins["observed_frequency"],
                                     mode="lines+markers", name="Uploaded model",
                                     line=dict(color="#2DD4BF", width=3), marker=dict(size=10),
                                     customdata=bins["count"],
                                     hovertemplate="Mean p=%{x:.3f}<br>Default frequency=%{y:.3f}<br>n=%{customdata}<extra></extra>"))
            fig.update_layout(title="Reliability diagram", xaxis_title="Mean predicted probability",
                              yaxis_title="Observed default frequency", xaxis_range=[0, 1], yaxis_range=[0, 1], height=385)
            plot(fig, "reliability")
        with b:
            conf = calc["confusion_matrix"]
            fig = go.Figure(go.Heatmap(z=conf, x=["Predicted 0", "Predicted 1"],
                                       y=["Actual 0", "Actual 1"],
                                       colorscale=[[0, "#14283B"], [1, "#2DD4BF"]],
                                       text=conf, texttemplate="%{text}", textfont=dict(size=22), showscale=False))
            fig.update_layout(title=f"Confusion matrix · threshold {threshold:.2f}", height=385,
                              xaxis_title=None, yaxis_title=None)
            fig.update_yaxes(autorange="reversed")
            plot(fig, "confusion")
        a, b = st.columns(2)
        if len(np.unique(y)) > 1:
            with a:
                fpr, tpr, _ = roc_curve(y, p)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC-AUC: {calc['roc_auc']:.3f}",
                                         line=dict(color="#60A5FA", width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="#64748B")))
                fig.update_layout(title="ROC curve", height=350, xaxis_title="False-positive rate", yaxis_title="True-positive rate")
                plot(fig, "roc")
            with b:
                precisions, recalls, _ = precision_recall_curve(y, p)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recalls, y=precisions, name=f"AP: {calc['pr_auc']:.3f}",
                                         line=dict(color="#FBBF24", width=3)))
                fig.add_hline(y=y.mean(), line=dict(color="#64748B", dash="dash"))
                fig.update_layout(title="Precision–recall curve", height=350,
                                  xaxis_title="Recall", yaxis_title="Precision")
                plot(fig, "pr")
        else:
            st.warning("ROC and PR curves require both default and non-default cases; this file contains only one class.")
        st.caption("ECE is the frequency-weighted absolute difference between predicted probability and observed default frequency in equal-width bins. It changes with the number of bins.")

with pages[3]:
    st.subheader("MC-dropout uncertainty")
    if rich is None or rich["mc_std"].notna().sum() < 2:
        st.info("Upload prediction-level records containing `mc_std`, `bayes_std`, or `predictive_variance` to explore MC-dropout uncertainty. No synthetic uncertainty values are created.")
        st.markdown("#### Previously reported uncertainty statistics · not recomputed")
        st.dataframe(pd.DataFrame([
            {"Measure": "Spearman(uncertainty, misclassification)", "Reported value": "0.2708"},
            {"Measure": "Spearman(uncertainty, absolute probability error)", "Reported value": "0.7257"},
            {"Measure": "AUROC of uncertainty for detecting errors", "Reported value": "0.6900"},
            {"Measure": "Average precision for detecting errors", "Reported value": "0.2940"},
            {"Measure": "Misclassification prevalence in uncertainty analysis", "Reported value": "0.2160"},
        ]).set_index("Measure"), width="stretch")
        st.caption("These numbers are transcribed from earlier analysis; confirm that they refer to the same model, threshold and test split used in your final thesis.")
    else:
        uncertainty = uncertainty_summary(rich)
        cards = st.columns(4)
        for col, name, value in zip(cards,
                                    ["Mean MC std", "Spearman: std vs error", "Error-detection AUROC", "Error-detection AP"],
                                    [uncertainty.get("mean_std"), uncertainty.get("spearman_error"),
                                     uncertainty.get("error_detection_auc"), uncertainty.get("error_detection_ap")]):
            col.metric(name, fmt(value, 4))
        st.caption("Error-detection AUROC treats MC standard deviation as the score and incorrect classification as the outcome. This is not credit-default ROC-AUC.")
        available = rich.dropna(subset=["mc_std"]).copy()
        available["prediction_result"] = np.where(available["error"] == 1, "Incorrect", "Correct")
        a, b = st.columns(2)
        with a:
            fig = px.scatter(available, x="predicted_probability", y="mc_std", color="prediction_result",
                             color_discrete_map={"Correct": "#2DD4BF", "Incorrect": "#FB7185"},
                             labels={"predicted_probability": "Predicted default probability", "mc_std": "MC standard deviation", "prediction_result": "Prediction result"},
                             opacity=.65, title="Uncertainty against default probability",
                             hover_data=[c for c in ["firm_id", "year"] if c in available])
            fig.update_layout(height=390)
            plot(fig, "uncertainty_scatter")
        with b:
            coverage = risk_coverage_curve(rich)
            fig = px.line(coverage, x="coverage", y="error_rate",
                          title="Risk–coverage: most-certain cases retained first",
                          labels={"coverage": "Fraction of predictions retained", "error_rate": "Error rate among retained"})
            fig.update_traces(line=dict(color="#60A5FA", width=3))
            fig.update_layout(height=390)
            fig.update_xaxes(range=[0, 1])
            plot(fig, "riskcoverage")
        st.markdown("#### Error rate by uncertainty decile")
        available["uncertainty_decile"] = pd.qcut(available["mc_std"].rank(method="first"),
                                                   q=min(10, len(available)), labels=False) + 1
        groups = available.groupby("uncertainty_decile", as_index=False).agg(error_rate=("error", "mean"), count=("error", "size"))
        fig = px.bar(groups, x="uncertainty_decile", y="error_rate", text="count",
                     title="D1 = lowest MC variability · bar labels = sample count",
                     labels={"uncertainty_decile": "Uncertainty decile", "error_rate": "Misclassification rate"})
        fig.update_traces(marker_color="#2DD4BF", textposition="outside")
        fig.update_layout(height=340)
        plot(fig, "uncertainty_deciles")
        st.caption("MC-dropout standard deviation measures variability across stochastic predictions. Its association with errors does not, by itself, establish purely epistemic uncertainty or calibrated uncertainty.")

with pages[4]:
    st.subheader("Firm-level prediction explorer")
    if rich is None:
        st.info("Upload firm-level test predictions to filter cases and download the resulting error and risk table.")
        st.code("firm_id,year,true_label,predicted_probability,mc_std,model", language="text")
    else:
        a, b, c = st.columns([1, 1, 1])
        with a:
            low_risk = st.slider("Low risk / medium risk boundary", .05, .45, .33, .01,
                                 help="Illustrative probability band, not a validated clinical or regulatory risk rule.")
        with b:
            high_risk = st.slider("Medium / high risk boundary", .55, .95, .66, .01)
        with c:
            error_choice = st.selectbox("Prediction result", ["All", "Correct", "Incorrect"])
        explorer = enrich_predictions(selected, threshold, low_risk, high_risk)
        ca, cb, cc = st.columns(3)
        with ca:
            risk_filter = st.multiselect("Default risk", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        with cb:
            uncertainty_filter = st.multiselect("MC uncertainty", ["Low", "Medium", "High", "Not available"],
                                                default=["Low", "Medium", "High", "Not available"])
        with cc:
            if "firm_id" in explorer:
                firm_search = st.text_input("Find firm ID (partial match)")
            else:
                firm_search = ""
                st.caption("Firm ID not provided in CSV")
        visible = explorer[
            explorer["risk_category"].isin(risk_filter) & explorer["uncertainty_category"].isin(uncertainty_filter)
        ]
        if error_choice == "Correct":
            visible = visible[visible["error"] == 0]
        elif error_choice == "Incorrect":
            visible = visible[visible["error"] == 1]
        if firm_search:
            visible = visible[visible["firm_id"].str.contains(firm_search, case=False, regex=False, na=False)]
        if "year" in visible.columns and visible["year"].notna().any():
            years = sorted(visible["year"].dropna().unique().tolist())
            chosen_years = st.multiselect("Observation year", years, default=years)
            visible = visible[visible["year"].isin(chosen_years)]
        st.caption(f"Showing {len(visible):,} of {len(explorer):,} selected-model prediction rows.")
        priority = ["firm_id", "year", "true_label", "predicted_probability", "predicted_label", "mc_std",
                    "risk_category", "uncertainty_category", "error", "absolute_probability_error"]
        ordered = [col for col in priority if col in visible.columns]
        display = visible[ordered].sort_values("predicted_probability", ascending=False)
        st.dataframe(display, width="stretch", hide_index=True)
        st.download_button("Download filtered firm predictions", data=display.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_firm_predictions.csv", mime="text/csv")
        st.caption("Risk bands are configurable probability categories. Uncertainty categories are relative within the uploaded model cohort (tercile ranks), not calibrated absolute thresholds.")

with pages[5]:
    st.subheader("Data contract & methodological checks")
    st.markdown("""
    **Prediction CSV (required columns)**: `true_label`, `predicted_probability`.
    Supported alternatives: `default_label`, `y_test`, `true_default` for the label and
    `bayes_prob`, `mc_mean` for probability.

    **Optional columns**: `firm_id`, `year`, `model`, `mc_std` (or `bayes_std`), or `predictive_variance`.
    If variance is supplied, the dashboard computes its square root to obtain standard deviation.

    **Model summary CSV**: `model` plus any of `accuracy`, `precision`, `recall`, `f1`, `roc_auc`,
    `pr_auc`, `brier`, `nll`, `ece`. Empty cells are preserved as missing.
    """)
    st.markdown("#### Suggested validation checklist")
    checks = [
        "Confirm that the default label reflects the NEXT 12 months after the final input observation, not the same-period outcome.",
        "Confirm the final prediction-year convention, observation frequency, and three-period window alignment.",
        "Ensure firm IDs do not overlap across train, validation and test groups.",
        "Fit imputation, encoding and scaling only on the training partition.",
        "Select the binary classification threshold on validation data, never on the test set.",
        "Evaluate all baselines against the same held-out labels and time horizon.",
        "Reconcile the baseline table and observed confusion matrix with the final saved predictions.",
        "Report MC-dropout pass count and ensure dropout is enabled at inference without enabling training-time batch-norm updates.",
        "Do not claim forward-chaining or rolling-origin validation unless it was actually conducted.",
    ]
    for index, item in enumerate(checks):
        st.checkbox(item, value=False, key=f"method_{index}")
    st.divider()
    st.markdown("#### Interpretation notes")
    st.markdown("""
    - The dashboard **evaluates existing predictions**; it does not train or serve the Transformer.
    - The selected threshold affects accuracy, precision, recall, F1, confusion matrix and error indicators.
    - Calibration, AUC and average precision use continuous default probabilities, not hard class labels.
    - ECE depends on the binning convention; Brier and log loss are proper scoring rules.
    - Draft model metrics should not be presented as validated final experimental results.
    - If deploying with sensitive firm data, assess hosting, access, privacy, and data-retention requirements before uploading.
    """)

st.divider()
st.caption("Credit Default Research Observatory · Interactive evaluation of uploaded test-set predictions · No model retraining")
