#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer 5 additional analyses for the cervical-cancer in-hospital mortality paper.
Faithfully reproduces the existing pipeline (encoding, 80/20 stratified split seed=42,
SMOTE on train, GridSearchCV over 5 models, StratifiedKFold(5), scoring='roc_auc'),
then ADDS ONLY what is missing:
  (1) Calibration curves (reliability diagrams) for all 5 models  -> Reviewer 5, comment 1
  (2) Brier score + 95% CI (bootstrap) for all models             -> supports comment 1
  (3) Multi-threshold performance table (Table S1 rebuilt)        -> Reviewer 5, comment 2b
  (4) Decision Curve Analysis (net benefit) for all models        -> Reviewer 5, comment 2c
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_curve, auc, confusion_matrix, recall_score,
                             precision_score, f1_score, accuracy_score,
                             matthews_corrcoef, brier_score_loss)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RNG = 42
np.random.seed(RNG)

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = sys.argv[1:2] or glob.glob(os.path.join(HERE, "**", "Banco_Internacao.csv"), recursive=True)
assert CSV_CANDIDATES, "Banco_Internacao.csv not found"
CSV_PATH = CSV_CANDIDATES[0]
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
print(f"[data] {CSV_PATH}")
print(f"[out ] {OUT}")

# ---------------------------------------------------------------- load (as in notebook)
df = pd.read_csv(CSV_PATH, sep=";").drop("Sexo", axis=1)

cols_num = ['DiasDePermanencia', 'NumeroInternacoes', 'ValorTotal', 'Idade']
cols_cat = ['AnoInternacao', 'CaraterInternacao', 'Complexidade', 'DiagnosticoPrincipal',
            'Especialidade', 'FoiAObito', 'Gestao', 'HospitalNome',
            'MunicipioResidencia', 'ProcedimentoGrupo', 'RacaCor', 'Regime', 'TeveDiariasUTI']

data = df.loc[:, cols_num + cols_cat].copy()
data['ValorTotal'] = data['ValorTotal'].astype(str).str.replace(',', '.').astype(float)

# mean imputation for numeric (as described in the manuscript), if any missing
for c in cols_num:
    if data[c].isna().any():
        data[c] = data[c].fillna(data[c].mean())

print(f"[data] shape={data.shape}  deaths={int((data['FoiAObito']=='Sim').sum()) if data['FoiAObito'].dtype==object else data['FoiAObito'].sum()}")

# ---------------------------------------------------------------- encoding (as in notebook)
label_encoding_cols = ['AnoInternacao', 'CaraterInternacao', 'Complexidade', 'DiagnosticoPrincipal',
                       'Especialidade', 'Gestao', 'ProcedimentoGrupo', 'RacaCor', 'Regime',
                       'TeveDiariasUTI', 'FoiAObito']
target_encoding_cols = ['HospitalNome', 'MunicipioResidencia', 'DiagnosticoPrincipal']

le = LabelEncoder()
for col in label_encoding_cols:
    data[col] = le.fit_transform(data[col])

te = TargetEncoder(cols=target_encoding_cols)
data[target_encoding_cols] = te.fit_transform(data[target_encoding_cols], data['FoiAObito'])

X = data.drop(columns=['FoiAObito'])
y = data['FoiAObito']

X.columns = ['Length of Stay', 'Number of Admissions', 'Total Cost', 'Age',
             'Year of Admission', 'Admission Type', 'Complexity', 'Main Diagnosis',
             'Medical Specialty', 'Management', 'Hospital', 'City of Residence',
             'Procedure Group', 'Race Color', 'Regimen', 'Intensive Care Unit']

print(f"[y] positive rate = {y.mean():.4f}")

# ---------------------------------------------------------------- split + SMOTE (as in notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=y)
X_train_s, y_train_s = SMOTE(random_state=RNG).fit_resample(X_train, y_train)

# ---------------------------------------------------------------- models + tuning (as in notebook)
models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    'RandomForest': (RandomForestClassifier(random_state=RNG),
                     {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}),
    'CatBoost': (CatBoostClassifier(random_state=RNG, silent=True),
                 {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.3], 'iterations': [100, 200]}),
    'LGBM': (LGBMClassifier(random_state=RNG, verbose=-1),
             {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [100, 200]}),
    'XGBoost': (XGBClassifier(random_state=RNG, use_label_encoder=False, eval_metric='logloss'),
                {'max_depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [100, 200]}),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

best_models, proba = {}, {}
for name, (mdl, params) in models.items():
    print(f"[fit] {name} ...", flush=True)
    gs = GridSearchCV(mdl, param_grid=params, cv=cv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train_s, y_train_s)
    best_models[name] = gs.best_estimator_
    proba[name] = gs.best_estimator_.predict_proba(X_test)[:, 1]
    print(f"      best: {gs.best_params_}")

# nice display order / colors
ORDER = ['XGBoost', 'RandomForest', 'CatBoost', 'LGBM', 'LogisticRegression']
LABELS = {'LGBM': 'LightGBM', 'LogisticRegression': 'Logistic Regression'}
COLORS = {'XGBoost': '#0d5c63', 'RandomForest': '#c1666b', 'CatBoost': '#e08e0b',
          'LGBM': '#5b8c5a', 'LogisticRegression': '#7d7d7d'}
lbl = lambda n: LABELS.get(n, n)

# ================================================================ (2) Brier + 95% CI
def brier_ci(yt, p, n_boot=2000, seed=RNG):
    yt = np.asarray(yt); p = np.asarray(p)
    base = brier_score_loss(yt, p)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yt))
    bs = [brier_score_loss(yt[b], p[b]) for b in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return base, lo, hi

brier_rows = []
for name in ORDER:
    b, lo, hi = brier_ci(y_test.values, proba[name])
    fpr, tpr, _ = roc_curve(y_test, proba[name]); a = auc(fpr, tpr)
    brier_rows.append({'Model': lbl(name), 'AUC-ROC': round(a, 3),
                       'Brier': round(b, 3), 'Brier 95% CI': f"{lo:.3f}-{hi:.3f}"})
brier_df = pd.DataFrame(brier_rows)
brier_df.to_csv(os.path.join(OUT, "brier_scores.csv"), index=False)
print("\n[Brier]\n", brier_df.to_string(index=False))

# ================================================================ (1) Calibration curves
fig, ax = plt.subplots(figsize=(7.2, 6.4))
ax.plot([0, 1], [0, 1], ls='--', lw=1.4, color='#333', label='Perfectly calibrated')
for name in ORDER:
    frac_pos, mean_pred = calibration_curve(y_test, proba[name], n_bins=10, strategy='quantile')
    ax.plot(mean_pred, frac_pos, marker='o', ms=5, lw=1.9, color=COLORS[name],
            label=f"{lbl(name)} (Brier={brier_score_loss(y_test, proba[name]):.3f})")
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed fraction of deaths')
ax.set_title('Calibration curves (reliability diagram) — test set')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "calibration_curves_300dpi.png"), dpi=300)
fig.savefig(os.path.join(OUT, "calibration_curves.tiff"), dpi=300)
plt.close(fig)
print("[fig] calibration_curves_300dpi.png")

# ================================================================ (3) Multi-threshold table (Table S1)
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.10), 2)
rows = []
for name in ORDER:
    p = proba[name]
    for t in THRESHOLDS:
        yp = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append({
            'Model': lbl(name), 'Threshold': t,
            'Accuracy': round(accuracy_score(y_test, yp), 2),
            'Sensitivity': round(recall_score(y_test, yp, zero_division=0), 2),
            'Specificity': round(spec, 2),
            'Precision': round(precision_score(y_test, yp, zero_division=0), 2),
            'F1': round(f1_score(y_test, yp, zero_division=0), 2),
            'MCC': round(matthews_corrcoef(y_test, yp), 2),
        })
tableS1 = pd.DataFrame(rows)
tableS1.to_csv(os.path.join(OUT, "TableS1_multithreshold.csv"), index=False)
print(f"[tab] TableS1_multithreshold.csv  ({len(tableS1)} rows, {len(THRESHOLDS)} thresholds x {len(ORDER)} models)")

# ================================================================ (4) Decision Curve Analysis
def net_benefit_model(yt, p, thr):
    yt = np.asarray(yt); n = len(yt)
    out = []
    for t in thr:
        pred = (p >= t).astype(int)
        tp = np.sum((pred == 1) & (yt == 1))
        fp = np.sum((pred == 1) & (yt == 0))
        w = t / (1 - t)
        out.append(tp / n - (fp / n) * w)
    return np.array(out)

pt = np.linspace(0.01, 0.60, 120)          # threshold-probability grid
prev = np.mean(y_test.values)
nb_all = prev - (1 - prev) * (pt / (1 - pt))  # treat-all
nb_none = np.zeros_like(pt)                   # treat-none

fig, ax = plt.subplots(figsize=(7.6, 6.0))
ax.plot(pt, nb_none, color='#000', lw=1.3, ls=':', label='Treat none')
ax.plot(pt, nb_all, color='#888', lw=1.4, ls='--', label='Treat all')
for name in ORDER:
    ax.plot(pt, net_benefit_model(y_test.values, proba[name], pt),
            lw=1.9, color=COLORS[name], label=lbl(name))
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_title('Decision curve analysis — in-hospital mortality')
ax.set_ylim(-0.02, prev + 0.03); ax.set_xlim(0, 0.60)
ax.legend(loc='upper right', fontsize=9, frameon=False); ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "decision_curve_analysis_300dpi.png"), dpi=300)
fig.savefig(os.path.join(OUT, "decision_curve_analysis.tiff"), dpi=300)
plt.close(fig)

# net-benefit summary at representative thresholds for the text
dca_rows = []
for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
    row = {'Threshold': t,
           'Treat all': round(float(prev - (1 - prev) * (t / (1 - t))), 4),
           'Treat none': 0.0}
    for name in ORDER:
        row[lbl(name)] = round(float(net_benefit_model(y_test.values, proba[name], [t])[0]), 4)
    dca_rows.append(row)
dca_df = pd.DataFrame(dca_rows)
dca_df.to_csv(os.path.join(OUT, "dca_net_benefit_summary.csv"), index=False)
print("[fig] decision_curve_analysis_300dpi.png")
print("\n[DCA net benefit]\n", dca_df.to_string(index=False))

# ---------------------------------------------------------------- machine-readable digest for letter
digest = {
    'test_prevalence': round(float(prev), 4),
    'n_test': int(len(y_test)),
    'brier': brier_df.to_dict(orient='records'),
    'dca_summary': dca_df.to_dict(orient='records'),
    'thresholds': THRESHOLDS.tolist(),
}
with open(os.path.join(OUT, "digest.json"), "w") as f:
    json.dump(digest, f, indent=2)
print("\n[done] all outputs in", OUT)
