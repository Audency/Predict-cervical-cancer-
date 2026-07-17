#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Additional model-evaluation analyses (Reviewer 5, BMC Cancer — Revision 4)
#
#  Paper : Machine learning for hospital-level risk stratification of in-hospital
#          mortality in cervical cancer hospitalizations (Mato Grosso, Brazil).
#
#  This script reproduces the published modelling pipeline and adds the analyses
#  requested by Reviewer 5:
#
#     1. AUROC and Brier score, each with a bootstrap 95% confidence interval.
#     2. Calibration curves (reliability diagrams) — before AND after post-hoc
#        isotonic recalibration.
#     3. A multi-threshold performance table (Supplementary Table S1),
#        thresholds 0.10–0.90.
#     4. Decision curve analysis (net benefit) vs. treat-all / treat-none.
#
#  The pipeline is identical to the notebook (feature encoding, 80/20 stratified
#  split with random_state=42, SMOTE on the training fold, GridSearchCV over the
#  same hyper-parameter grids). Analyses are run on the *analytic sample*
#  (n = 3,493) obtained by excluding records with unknown race.
#
#  Usage
#  -----
#      python reviewer5_additional_analyses.py [path/to/Banco_Internacao.csv]
#
#  Outputs are written to ./outputs (figures at 300 dpi + CSV tables).
# =============================================================================

import os
import sys
import glob
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")               # headless backend — write figures to disk only
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, brier_score_loss,
    accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef,
)
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_BOOTSTRAP  = 2000                 # resamples for the bootstrap confidence intervals
THRESHOLDS   = np.round(np.arange(0.10, 0.91, 0.10), 2)

# Display order and styling, shared by every figure/table
MODEL_ORDER = ["XGBoost", "RandomForest", "CatBoost", "LGBM", "LogisticRegression"]
NICE_NAME   = {"LGBM": "LightGBM", "LogisticRegression": "Logistic Regression"}
COLOR       = {"XGBoost": "#0d5c63", "RandomForest": "#c1666b", "CatBoost": "#e08e0b",
               "LGBM": "#5b8c5a", "LogisticRegression": "#7d7d7d"}
label = lambda name: NICE_NAME.get(name, name)

# Columns (raw dataset)
NUMERIC_COLS = ["DiasDePermanencia", "NumeroInternacoes", "ValorTotal", "Idade"]
CATEG_COLS   = ["AnoInternacao", "CaraterInternacao", "Complexidade", "DiagnosticoPrincipal",
                "Especialidade", "FoiAObito", "Gestao", "HospitalNome", "MunicipioResidencia",
                "ProcedimentoGrupo", "RacaCor", "Regime", "TeveDiariasUTI"]
# English feature names (order must match NUMERIC_COLS + CATEG_COLS minus the target)
FEATURE_NAMES = ["Length of Stay", "Number of Admissions", "Total Cost", "Age",
                 "Year of Admission", "Admission Type", "Complexity", "Main Diagnosis",
                 "Medical Specialty", "Management", "Hospital", "City of Residence",
                 "Procedure Group", "Race Color", "Regimen", "Intensive Care Unit"]


# -----------------------------------------------------------------------------
#  Data loading and preprocessing
# -----------------------------------------------------------------------------
def load_analytic_sample(csv_path):
    """Load the SIH/SUS data and restrict to the analytic sample used in the paper.

    The analytic sample (n = 3,493) excludes records with unknown race
    (RacaCor == '*Nao informado'), mirroring the R data-cleaning step.
    """
    df = pd.read_csv(csv_path, sep=";").drop("Sexo", axis=1)
    df = df[df["RacaCor"] != "*Nao informado"].reset_index(drop=True)
    print(f"[data] {csv_path}")
    print(f"[data] analytic sample: n={len(df)}  deaths={(df['FoiAObito'] == 'Sim').sum()}")
    return df


def build_features(df):
    """Encode predictors and return the model matrix X and target y.

    - numeric 'ValorTotal' uses a comma decimal separator -> convert to float;
      any missing numeric values are mean-imputed.
    - categorical predictors are label-encoded; high-cardinality columns
      (hospital, city, main diagnosis) are additionally target-encoded.
    """
    data = df.loc[:, NUMERIC_COLS + CATEG_COLS].copy()
    data["ValorTotal"] = data["ValorTotal"].astype(str).str.replace(",", ".").astype(float)
    for col in NUMERIC_COLS:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mean())

    label_cols  = ["AnoInternacao", "CaraterInternacao", "Complexidade", "DiagnosticoPrincipal",
                   "Especialidade", "Gestao", "ProcedimentoGrupo", "RacaCor", "Regime",
                   "TeveDiariasUTI", "FoiAObito"]
    target_cols = ["HospitalNome", "MunicipioResidencia", "DiagnosticoPrincipal"]

    encoder = LabelEncoder()
    for col in label_cols:
        data[col] = encoder.fit_transform(data[col])
    data[target_cols] = TargetEncoder(cols=target_cols).fit_transform(data[target_cols], data["FoiAObito"])

    X = data.drop(columns=["FoiAObito"])
    y = data["FoiAObito"]
    X.columns = FEATURE_NAMES
    return X, y


# -----------------------------------------------------------------------------
#  Model definitions and training
# -----------------------------------------------------------------------------
def model_grid():
    """Return {name: (estimator, param_grid)} — identical to the notebook."""
    return {
        "LogisticRegression": (LogisticRegression(max_iter=1000),
                               {"C": [0.1, 1, 10]}),
        "RandomForest":       (RandomForestClassifier(random_state=RANDOM_STATE),
                               {"n_estimators": [100, 200], "max_depth": [10, 20, None],
                                "min_samples_split": [2, 5]}),
        "CatBoost":           (CatBoostClassifier(random_state=RANDOM_STATE, silent=True),
                               {"depth": [4, 6, 8], "learning_rate": [0.01, 0.1, 0.3],
                                "iterations": [100, 200]}),
        "LGBM":               (LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                               {"num_leaves": [31, 50], "learning_rate": [0.01, 0.1, 0.3],
                                "n_estimators": [100, 200]}),
        "XGBoost":            (XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False,
                                             eval_metric="logloss"),
                               {"max_depth": [4, 6, 8], "learning_rate": [0.01, 0.1, 0.3],
                                "n_estimators": [100, 200]}),
    }


def tune_models(X_train_smote, y_train_smote):
    """Tune each model with 5-fold cross-validated grid search (scoring = ROC-AUC)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best = {}
    for name, (estimator, grid) in model_grid().items():
        print(f"[fit] {name} ...", flush=True)
        search = GridSearchCV(estimator, param_grid=grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train_smote, y_train_smote)
        best[name] = search.best_estimator_
        print(f"      best params: {search.best_params_}")
    return best


# -----------------------------------------------------------------------------
#  Statistics helpers
# -----------------------------------------------------------------------------
def _bootstrap_ci(metric_fn, y_true, y_prob, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Percentile bootstrap 95% CI for a probabilistic metric (AUROC or Brier)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng, idx, scores = np.random.default_rng(seed), np.arange(len(y_true)), []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:      # AUROC undefined for a single-class resample
            continue
        scores.append(metric_fn(y_true[b], y_prob[b]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return metric_fn(y_true, y_prob), lo, hi


auc_score = lambda yt, yp: auc(*roc_curve(yt, yp)[:2])
auc_ci    = lambda yt, yp: _bootstrap_ci(auc_score, yt, yp)
brier_ci  = lambda yt, yp: _bootstrap_ci(brier_score_loss, yt, yp)


def specificity_from(y_true, y_pred):
    """Specificity = TN / (TN + FP)."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


# -----------------------------------------------------------------------------
#  (1) Discrimination + calibration summary (AUROC & Brier with 95% CI)
# -----------------------------------------------------------------------------
def discrimination_table(y_test, proba, out_dir):
    rows = []
    for name in MODEL_ORDER:
        a, a_lo, a_hi = auc_ci(y_test.values, proba[name])
        b, b_lo, b_hi = brier_ci(y_test.values, proba[name])
        rows.append({"Model": label(name),
                     "AUC-ROC": round(a, 3), "AUC 95% CI": f"{a_lo:.3f}-{a_hi:.3f}",
                     "Brier": round(b, 3),  "Brier 95% CI": f"{b_lo:.3f}-{b_hi:.3f}"})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "discrimination_calibration.csv"), index=False)
    print("\n[AUROC + Brier, 95% CI]\n", table.to_string(index=False))
    return table


# -----------------------------------------------------------------------------
#  (2) Calibration curves — before and after isotonic recalibration
# -----------------------------------------------------------------------------
def recalibrate(best_models, X_train, y_train, X_test):
    """Post-hoc isotonic recalibration, done correctly.

    A calibration subset is held out from the training data *at the real class
    prevalence* (never SMOTE-resampled). For each model a clone with the same
    tuned hyper-parameters is refit on SMOTE(fit-part), and the isotonic map is
    learned on the untouched calibration part. The published models are unchanged.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train)
    X_fit_s, y_fit_s = SMOTE(random_state=RANDOM_STATE).fit_resample(X_fit, y_fit)

    proba_cal = {}
    for name in MODEL_ORDER:
        base = clone(best_models[name]).fit(X_fit_s, y_fit_s)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit").fit(X_cal, y_cal)
        proba_cal[name] = calibrated.predict_proba(X_test)[:, 1]
    return proba_cal


def plot_calibration(y_test, proba, proba_cal, out_dir):
    """Two-panel reliability diagram: raw probabilities vs. recalibrated ones."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), sharey=True)
    for ax, (probs, title) in zip(axes, [(proba, "Before recalibration"),
                                          (proba_cal, "After isotonic recalibration")]):
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color="#333", label="Perfectly calibrated")
        for name in MODEL_ORDER:
            frac_pos, mean_pred = calibration_curve(y_test, probs[name], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", ms=5, lw=1.9, color=COLOR[name],
                    label=f"{label(name)} (Brier={brier_score_loss(y_test, probs[name]):.3f})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Observed fraction of deaths")
    fig.suptitle("Calibration curves (reliability diagram) — test set", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_curves_300dpi.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "calibration_curves.tiff"), dpi=300)
    plt.close(fig)

    # Brier score after recalibration (with 95% CI) for the manuscript/letter
    rows = []
    for name in MODEL_ORDER:
        b, lo, hi = brier_ci(y_test.values, proba_cal[name])
        rows.append({"Model": label(name), "Brier (recalibrated)": round(b, 3),
                     "Brier 95% CI": f"{lo:.3f}-{hi:.3f}"})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "brier_recalibrated.csv"), index=False)
    print("[fig] calibration_curves_300dpi.png (before vs after isotonic recalibration)")
    print("[recalibrated Brier]\n", pd.DataFrame(rows).to_string(index=False))


# -----------------------------------------------------------------------------
#  (3) Multi-threshold performance table (Supplementary Table S1)
# -----------------------------------------------------------------------------
def threshold_table(y_test, proba, out_dir):
    """Threshold-dependent metrics for every model across THRESHOLDS."""
    auc_cache = {name: auc_ci(y_test.values, proba[name]) for name in MODEL_ORDER}
    rows = []
    for name in MODEL_ORDER:
        a, a_lo, a_hi = auc_cache[name]                   # AUROC is threshold-independent
        for t in THRESHOLDS:
            y_pred = (proba[name] >= t).astype(int)
            rows.append({
                "Model": label(name), "Threshold": t,
                "AUC-ROC": round(a, 3), "AUC 95% CI": f"{a_lo:.3f}-{a_hi:.3f}",
                "Accuracy":    round(accuracy_score(y_test, y_pred), 2),
                "Sensitivity": round(recall_score(y_test, y_pred, zero_division=0), 2),
                "Specificity": round(specificity_from(y_test, y_pred), 2),
                "Precision":   round(precision_score(y_test, y_pred, zero_division=0), 2),
                "F1":          round(f1_score(y_test, y_pred, zero_division=0), 2),
                "MCC":         round(matthews_corrcoef(y_test, y_pred), 2),
            })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "TableS1_multithreshold.csv"), index=False)
    print(f"[tab] TableS1_multithreshold.csv "
          f"({len(table)} rows = {len(THRESHOLDS)} thresholds x {len(MODEL_ORDER)} models)")
    return table


# -----------------------------------------------------------------------------
#  (4) Decision curve analysis (net benefit)
# -----------------------------------------------------------------------------
def _net_benefit(y_true, y_prob, thresholds):
    """Net benefit = TP/n - (FP/n) * (p_t / (1 - p_t)) for each threshold p_t."""
    y_true, n, out = np.asarray(y_true), len(y_true), []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        out.append(tp / n - (fp / n) * (t / (1 - t)))
    return np.array(out)


def plot_decision_curve(y_test, proba, out_dir):
    """Decision curve analysis vs. the treat-all / treat-none reference strategies."""
    grid = np.linspace(0.01, 0.60, 120)
    prevalence = float(np.mean(y_test.values))
    nb_treat_all = prevalence - (1 - prevalence) * (grid / (1 - grid))

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot(grid, np.zeros_like(grid), ls=":",  lw=1.3, color="#000", label="Treat none")
    ax.plot(grid, nb_treat_all,        ls="--", lw=1.4, color="#888", label="Treat all")
    for name in MODEL_ORDER:
        ax.plot(grid, _net_benefit(y_test.values, proba[name], grid),
                lw=1.9, color=COLOR[name], label=label(name))
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis — in-hospital mortality")
    ax.set_xlim(0, 0.60)
    ax.set_ylim(-0.02, prevalence + 0.03)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "decision_curve_analysis_300dpi.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "decision_curve_analysis.tiff"), dpi=300)
    plt.close(fig)

    # Net benefit at a few representative thresholds, for the text
    rows = []
    for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
        row = {"Threshold": t,
               "Treat all": round(float(prevalence - (1 - prevalence) * (t / (1 - t))), 4),
               "Treat none": 0.0}
        for name in MODEL_ORDER:
            row[label(name)] = round(float(_net_benefit(y_test.values, proba[name], [t])[0]), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "dca_net_benefit_summary.csv"), index=False)
    print("[fig] decision_curve_analysis_300dpi.png")
    print("\n[DCA net benefit]\n", summary.to_string(index=False))
    return summary, prevalence


# -----------------------------------------------------------------------------
#  Orchestration
# -----------------------------------------------------------------------------
def main():
    # Resolve the dataset path (CLI argument, else search the working tree)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = sys.argv[1:2] or glob.glob(os.path.join(here, "**", "Banco_Internacao.csv"),
                                             recursive=True)
    assert candidates, "Banco_Internacao.csv not found"
    csv_path = candidates[0]

    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- Data ---------------------------------------------------------------
    df = load_analytic_sample(csv_path)
    X, y = build_features(df)
    print(f"[y] positive (death) rate = {y.mean():.4f}")

    # --- Split + SMOTE (as in the notebook) ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train_smote, y_train_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    # --- Train + predicted probabilities on the test set --------------------
    best_models = tune_models(X_train_smote, y_train_smote)
    proba = {name: model.predict_proba(X_test)[:, 1] for name, model in best_models.items()}

    # --- Reviewer-5 analyses ------------------------------------------------
    discrimination_table(y_test, proba, out_dir)                       # (1)
    proba_cal = recalibrate(best_models, X_train, y_train, X_test)     # (2) recalibration
    plot_calibration(y_test, proba, proba_cal, out_dir)               #     figure + Brier
    threshold_table(y_test, proba, out_dir)                           # (3)
    _, prevalence = plot_decision_curve(y_test, proba, out_dir)       # (4)

    # --- Machine-readable digest -------------------------------------------
    with open(os.path.join(out_dir, "digest.json"), "w") as fh:
        json.dump({"csv_path": csv_path, "n_test": int(len(y_test)),
                   "test_prevalence": round(prevalence, 4),
                   "thresholds": THRESHOLDS.tolist()}, fh, indent=2)
    print("\n[done] all outputs in", out_dir)


if __name__ == "__main__":
    main()
