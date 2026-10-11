# 📉 Customer Churn Predictor (Explainable ML + Business Impact)

End-to-end ML project: data cleaning → feature engineering → model comparison & tuning → evaluation →
**SHAP explainability** → **profit-based threshold optimisation** → **Streamlit app** (single prediction + batch scoring).

Live demo: _add your Streamlit link here_

## Why this project
Predicting churn is only useful if the business can act on it. This project goes beyond a classifier:
it explains *why* each customer is at risk (SHAP), and converts model scores into a retention-campaign decision
by choosing the threshold that maximises profit, not just F1.

## Project structure
```
app.py              Streamlit app (Predict, Batch scoring, Business impact, Model insights)
src/data.py         Loading + cleaning (blank TotalCharges, SeniorCitizen, ID columns)
src/features.py     Feature engineering, preprocessing pipeline, input schema
src/train.py        CV, tuning, threshold selection, evaluation, SHAP plots
src/business.py     Retention-campaign profit curve, top-20% churner capture
src/explain.py      SHAP helpers
artifacts/          Saved model, metrics, test predictions, plots (commit for deployment)
data/               Put telco_churn.csv here (optional, see data/README.md)
```

## Run locally
```bash
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train            # writes artifacts/
python -m streamlit run app.py
```
Dataset: Kaggle "Telco Customer Churn" saved as `data/telco_churn.csv` (otherwise auto-downloaded from OpenML).

## Method
- **Cleaning:** blank `TotalCharges` for new customers set to 0; `SeniorCitizen` converted to Yes/No; ID column dropped.
- **Feature engineering:** `num_services` (count of subscribed services), `avg_monthly_spend` (TotalCharges / tenure), `charge_vs_avg` (current bill minus historical average).
- **Split:** 80/20 stratified. All tuning uses 5-fold stratified CV on the training set only; preprocessing sits inside the CV pipeline (no leakage).
- **Models:** Logistic Regression baseline vs tuned Random Forest and Gradient Boosting (RandomizedSearchCV, ROC-AUC).
- **Imbalance (~27% churn):** stratification, class weights, PR-AUC, and threshold tuning from out-of-fold predictions.
- **Explainability:** SHAP TreeExplainer: global (bar, beeswarm) and per-customer (waterfall).
- **Business layer:** profit curve under adjustable assumptions (offer cost, lifetime value saved, success rate).


## Deploy (Streamlit Community Cloud, free)
1. Run `python -m src.train`, then push the whole folder (including `artifacts/`) to a public GitHub repo.
2. share.streamlit.io → New app → pick repo, branch `main`, file `app.py` → Deploy.
3. Put the live URL at the top of this README and on your resume.

## Limitations & next steps
- Single public dataset; no time dimension, so no true out-of-time validation.
- Business assumptions are illustrative; real values would come from finance/marketing.
- Next: XGBoost/LightGBM, probability calibration, uplift modelling (who responds to an offer), FastAPI + Docker.
