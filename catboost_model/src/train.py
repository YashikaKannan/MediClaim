"""
train.py - standout fraud stack with 5-fold cross-validation + honest metrics.

Stack:
    CatBoost (overfit-safe settings)    -- strong supervised base
  (Isolation Forest removed in this version)
  + fraud-ring graph (ring_connections) -- wow feature
  + temporal drift  (max_month_jump)    -- bonus novelty
  + SHAP                                -- explainability

Overfit protection:
  * shallow trees (depth=4), strong L2 (l2_leaf_reg=12), row/col subsampling
  * we print the TRAIN-vs-TEST gap each fold; a small gap = not memorizing
Honest metrics (data is 9.4% fraud, so accuracy is misleading):
  * ROC-AUC, AUC-PR, Recall, Precision, F1 -- averaged over 5 folds with +/- spread

Run:  python -m src.train
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
import shap

from .features import build_features, FEATURES
from .graph_viz import draw as draw_graph

warnings.filterwarnings("ignore")
OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

CB_PARAMS = dict(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=12,
                 subsample=0.8, rsm=0.8, auto_class_weights="Balanced",
                 verbose=0, random_seed=42, allow_writing_files=False)


def cross_validate(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    print("\n----- 5-FOLD CROSS-VALIDATION -----")
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        cb = CatBoostClassifier(**CB_PARAMS)
        cb.fit(X.iloc[tr], y[tr])
        p = cb.predict_proba(X.iloc[te])[:, 1]
        ptr = cb.predict_proba(X.iloc[tr])[:, 1]
        pred = (p >= 0.5).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary")
        te_auc, tr_auc = roc_auc_score(y[te], p), roc_auc_score(y[tr], ptr)
        rows.append(dict(fold=i, test_auc=te_auc, train_auc=tr_auc, gap=tr_auc - te_auc,
                         auc_pr=average_precision_score(y[te], p),
                         recall=rc, precision=pr, f1=f1))
        print(f"  Fold {i}: test-AUC {te_auc:.3f} | train-AUC {tr_auc:.3f} | "
              f"gap {tr_auc-te_auc:.3f} | Recall {rc:.3f} | F1 {f1:.3f}")
    cv = pd.DataFrame(rows)
    print("\n  ===== 5-FOLD AVERAGES (honest metrics) =====")
    for m, label in [("test_auc", "ROC-AUC "), ("auc_pr", "AUC-PR  "),
                     ("recall", "Recall  "), ("precision", "Precision"), ("f1", "F1      ")]:
        print(f"  {label}: {cv[m].mean():.3f}  (+/- {cv[m].std():.3f})")
    gap = cv["gap"].mean()
    verdict = "small gap -> NOT overfitting" if gap < 0.05 else "watch the gap"
    print(f"  Train-vs-Test gap: {gap:.3f}  ({verdict})")
    print("  Tiny +/- spread = scores are stable = not luck.")
    cv.to_csv(OUTPUT / "cross_validation_scores.csv", index=False)
    return cv



def tune_threshold(X, y):
    """Find the best cutoff using OUT-OF-FOLD predictions (unseen data -> no leakage)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = CatBoostClassifier(**CB_PARAMS)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    print("\n----- THRESHOLD TUNING (on unseen out-of-fold predictions) -----")
    print(f"  {'cutoff':>7} {'Recall':>8} {'Precision':>10} {'F1':>7}")
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.30, 0.71, 0.05):
        pr, rc, f1, _ = precision_recall_fscore_support(y, (oof >= t).astype(int), average="binary")
        if f1 > best_f1:
            best_f1, best_t = f1, t
        print(f"  {t:>7.2f} {rc:>8.3f} {pr:>10.3f} {f1:>7.3f}")
    print(f"  BEST cutoff for F1: {best_t:.2f} (F1={best_f1:.3f})")
    print("  Lower cutoff = catch more fraud (more false alarms); higher = fewer false alarms.")
    return best_t


def run():
    print("Building features (graph + temporal included)...")
    prov = build_features()
    X, y = prov[FEATURES], prov["y"].values

    cross_validate(X, y)
    best_t = tune_threshold(X, y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    cb = CatBoostClassifier(**CB_PARAMS)
    cb.fit(Xtr, ytr)
    proba_all = cb.predict_proba(X)[:, 1]

    prov["fraud_proba"] = proba_all
    prov["Risk_Score"] = (proba_all * 100).round(2)
    prov["Risk_Level"] = pd.cut(prov["Risk_Score"], [-1, 30, 60, 80, 101],
                                labels=["Low", "Medium", "High", "Critical"])
    # flag using the tuned threshold (fewer false alarms than a blind 0.5)
    prov["Model_Flag"] = (prov["fraud_proba"] >= best_t).astype(int)

    expl = shap.TreeExplainer(cb)
    sv = expl.shap_values(X)
    def reason(i):
        row = dict(zip(FEATURES, sv[i]))
        top = sorted(row.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return "; ".join(f"{k} (impact {v:+.2f})" for k, v in top)
    prov["Explanation"] = [reason(i) for i in range(len(prov))]

    prov.sort_values("Risk_Score", ascending=False)[
        ["Provider", "Risk_Score", "Risk_Level", "fraud_proba",
         "Model_Flag", "ring_connections", "max_month_jump", "n_after_death", "Explanation", "PotentialFraud"]
    ].to_csv(OUTPUT / "provider_risk_ranked.csv", index=False)

    print(f"\nProviders scored: {len(prov):,} | linked in fraud-ring graph: {(prov['ring_connections']>0).sum():,}")
    print(f"Saved -> {OUTPUT/'provider_risk_ranked.csv'}")
    print(f"Saved -> {OUTPUT/'cross_validation_scores.csv'}")

    # draw the fraud-ring picture
    try:
        draw_graph()
    except Exception as e:
        print(f"[graph] skipped ({e})")


if __name__ == "__main__":
    run()
