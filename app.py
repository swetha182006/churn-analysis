"""Streamlit app: customer churn prediction with SHAP explanations and business impact."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.business import campaign_profit, profit_curve
from src.data import clean_features
from src.explain import explain, get_explainer
from src.features import add_features

ART = Path("artifacts")
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_bundle():
    bundle = joblib.load(ART / "model.joblib")
    bundle["explainer"] = get_explainer(bundle["model"])
    return bundle


def to_model_input(raw_df, bundle):
    """raw customer rows -> engineered, ordered frame -> preprocessed features."""
    df = add_features(clean_features(raw_df))
    return bundle["preprocessor"].transform(df[bundle["model_columns"]])


def render_input_form(schema):
    values = {}
    cols = st.columns(3)
    visible = [k for k in schema if k != "TotalCharges"]  # derived: MonthlyCharges x tenure
    for i, name in enumerate(visible):
        spec, label = schema[name], name.replace("_", " ").title()
        with cols[i % 3]:
            if spec["type"] == "cat":
                opts = spec["options"]
                values[name] = st.selectbox(label, opts, index=opts.index(spec["default"]))
            elif spec["is_int"]:
                values[name] = st.number_input(
                    label, min_value=int(spec["min"]), max_value=int(spec["max"] * 2), value=int(spec["median"]), step=1
                )
            else:
                values[name] = st.number_input(
                    label, min_value=float(spec["min"]), max_value=float(spec["max"] * 2), value=float(spec["median"])
                )
    if "TotalCharges" in schema:
        values["TotalCharges"] = values.get("MonthlyCharges", 0.0) * values.get("tenure", 0)
    return values


st.title("📉 Customer Churn Predictor")
st.caption("Predict who is likely to leave, understand why with SHAP, and pick the retention strategy that makes money.")

if not (ART / "model.joblib").exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

bundle = load_bundle()
tabs = st.tabs(["🔮 Predict", "📂 Batch scoring", "💰 Business impact", "📊 Model insights", "ℹ️ About"])

# ---------------------------------------------------------------- single prediction
with tabs[0]:
    with st.form("customer"):
        values = render_input_form(bundle["schema"])
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        row = pd.DataFrame([values])[bundle["raw_columns"]]
        Xt = to_model_input(row, bundle)
        prob = float(bundle["model"].predict_proba(Xt)[0, 1])
        high = prob >= bundle["threshold"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn probability", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            if high:
                st.error(f"AT RISK: above decision threshold ({bundle['threshold']:.2f}). Consider a retention offer.")
            else:
                st.success(f"LIKELY TO STAY: below decision threshold ({bundle['threshold']:.2f}).")
        with c2:
            import shap

            exp = explain(bundle["explainer"], Xt)
            st.subheader("Why this prediction?")
            plt.figure()
            shap.plots.waterfall(exp[0], max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            st.caption("Red bars push toward CHURN, blue bars toward STAYING.")

        contrib = pd.Series(exp.values[0], index=Xt.columns).sort_values(key=abs, ascending=False).head(3)
        st.markdown(
            "**Top 3 drivers:** "
            + ", ".join(f"`{k}` ({'↑ churn risk' if v > 0 else '↓ churn risk'})" for k, v in contrib.items())
        )

# ---------------------------------------------------------------- batch scoring
with tabs[1]:
    st.write("Upload a CSV of customers (same columns as the training data) to score them all at once.")
    up = st.file_uploader("Customer CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        cleaned = clean_features(raw)
        missing = [c for c in bundle["raw_columns"] if c not in cleaned.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            Xt = to_model_input(cleaned[bundle["raw_columns"]], bundle)
            probs = bundle["model"].predict_proba(Xt)[:, 1]
            out = raw.copy()
            out["churn_probability"] = probs.round(4)
            out["at_risk"] = probs >= bundle["threshold"]
            out = out.sort_values("churn_probability", ascending=False)
            st.metric("Customers flagged at risk", f"{int(out['at_risk'].sum())} of {len(out)}")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_customers.csv", "text/csv")

# ---------------------------------------------------------------- business impact
with tabs[2]:
    st.write(
        "A model only matters if it drives a decision. Suppose we send a retention offer to every customer flagged "
        "as churn. Set your assumptions to find the best threshold in money terms."
    )
    tp_path = ART / "test_predictions.csv"
    if tp_path.exists():
        tp = pd.read_csv(tp_path)
        a, b, c = st.columns(3)
        clv = a.slider("Customer lifetime value saved ($)", 100, 3000, 500, 50)
        cost = b.slider("Cost of a retention offer ($)", 5, 300, 50, 5)
        succ = c.slider("Offer success rate", 0.05, 1.0, 0.30, 0.05)

        grid, profits = profit_curve(tp.y_true, tp.proba, clv, cost, succ)
        scale = 1000 / len(tp)
        best_t = float(grid[profits.argmax()])
        rows = {
            "Default 0.50": 0.5,
            f"F1-optimal ({bundle['threshold']:.2f})": bundle["threshold"],
            f"Profit-optimal ({best_t:.2f})": best_t,
        }
        comp = pd.DataFrame(
            {"Threshold": list(rows.values()),
             "Profit per 1,000 customers ($)": [round(campaign_profit(tp.y_true, tp.proba, t, clv, cost, succ) * scale) for t in rows.values()]},
            index=list(rows.keys()),
        )
        st.dataframe(comp)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(grid, profits * scale)
        ax.axvline(best_t, color="green", ls="--", label=f"best threshold {best_t:.2f}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Profit per 1,000 customers ($)")
        ax.legend()
        st.pyplot(fig)
        st.caption("Computed on the held-out test set. Profit is relative to doing nothing.")
    else:
        st.info("Train the model to generate test predictions.")

# ---------------------------------------------------------------- insights
with tabs[3]:
    mpath = ART / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.subheader(f"Selected model: {m['selected_model']}")
        items = list(m["test"].items())
        cols = st.columns(len(items))
        for col, (k, v) in zip(cols, items):
            col.metric(k.replace("_", " ").upper(), v)
        st.markdown("**Cross-validated ROC-AUC (training set)**")
        st.dataframe(pd.Series(m["cv_roc_auc"], name="ROC-AUC").to_frame())
    for pair in [
        [("shap_bar.png", "Global feature importance (SHAP)"), ("shap_beeswarm.png", "Feature impact (SHAP)")],
        [("roc_curve.png", "ROC curve"), ("confusion_matrix.png", "Confusion matrix")],
    ]:
        for col, (img, title) in zip(st.columns(2), pair):
            if (ART / img).exists():
                col.subheader(title)
                col.image(str(ART / img))

with tabs[4]:
    st.markdown(
        """
**Pipeline:** raw customer data → cleaning → engineered features (`num_services`, `avg_monthly_spend`, `charge_vs_avg`)
→ imputation + one-hot encoding → tuned tree ensemble (compared with a Logistic Regression baseline)
→ F1-optimised threshold → SHAP explanations → profit-based threshold analysis.

**Data:** Telco Customer Churn (~7,000 customers, ~27% churn).

**Caveat:** public dataset for demonstration; business assumptions in the profit tab are adjustable guesses.
"""
    )
