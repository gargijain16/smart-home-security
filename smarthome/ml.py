"""Random Forest (attack classification) + Isolation Forest (anomaly detection), risk scoring."""
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from . import config
from .data import FEATURES, featurize

SEVERITY = {"normal": 0.0, "port_scan": 0.5, "brute_force": 0.8, "dos": 0.9, "suspicious_outbound": 1.0}


def _dir(model_dir):
    p = Path(model_dir or config.MODEL_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train(df, model_dir=None, seed=42):
    """Train both models on a stratified 75% split, evaluate on the held-out 25%, persist everything."""
    d = _dir(model_dir)
    X = featurize(df).to_numpy(dtype=np.float64)
    y = df["label"].astype(str).to_numpy(dtype=str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    rf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=2, class_weight="balanced",
                                n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed).fit(Xtr[ytr == "normal"])

    raw_tr = -iso.score_samples(Xtr)
    calib = {"lo": float(np.percentile(raw_tr[ytr == "normal"], 50)), "hi": float(np.percentile(raw_tr, 99))}

    pred = rf.predict(Xte)
    labels = list(rf.classes_)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    rep = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)

    is_attack = yte != "normal"
    flagged = iso.predict(Xte) == -1
    ip, ir, if1, _ = precision_recall_fscore_support(is_attack, flagged, average="binary", zero_division=0)
    fpr = float(flagged[~is_attack].mean()) if (~is_attack).any() else 0.0

    metrics = {
        "data_note": "Evaluated on a held-out 25% split of SYNTHETIC data. Not a real-world validation.",
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "n_rows": int(len(df)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "random_forest": {
            "accuracy": float(accuracy_score(yte, pred)), "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f), "labels": labels, "per_class": {k: rep[k] for k in labels},
            "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
            "feature_importance": dict(sorted(zip(FEATURES, map(float, rf.feature_importances_)), key=lambda t: -t[1])),
        },
        "isolation_forest": {"precision_vs_attack": float(ip), "recall_vs_attack": float(ir), "f1_vs_attack": float(if1),
                             "false_positive_rate_on_normal": fpr, "contamination": 0.05, "trained_on": "normal rows only"},
    }
    joblib.dump(rf, d / "random_forest.joblib")
    joblib.dump({"model": iso, "calib": calib}, d / "isolation_forest.joblib")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def models_exist(model_dir=None):
    d = Path(model_dir or config.MODEL_DIR)
    return all((d / f).exists() for f in ["random_forest.joblib", "isolation_forest.joblib", "metrics.json"])


def load_models(model_dir=None):
    """Return {'rf','iso','calib','metrics'} or None if not trained yet."""
    if not models_exist(model_dir):
        return None
    d = Path(model_dir or config.MODEL_DIR)
    iso = joblib.load(d / "isolation_forest.joblib")
    return {"rf": joblib.load(d / "random_forest.joblib"), "iso": iso["model"], "calib": iso["calib"],
            "metrics": json.loads((d / "metrics.json").read_text())}


def risk_level(score):
    return pd.cut(score, bins=config.RISK_BINS, labels=config.RISK_LABELS, right=False, include_lowest=True).astype(str)


def score(df, bundle):
    """Add prediction, confidence, anomaly score, risk score (0-100) and level to a flow table."""
    X = featurize(df)
    proba = bundle["rf"].predict_proba(X)
    classes = list(bundle["rf"].classes_)
    out = df.copy()
    out["predicted"] = np.array(classes)[proba.argmax(1)]
    out["confidence"] = proba.max(1).round(4)
    p_attack = 1.0 - proba[:, classes.index("normal")] if "normal" in classes else np.zeros(len(df))
    raw = -bundle["iso"].score_samples(X)
    c = bundle["calib"]
    anom = np.clip((raw - c["lo"]) / max(c["hi"] - c["lo"], 1e-9), 0, 1)
    sev = out["predicted"].map(SEVERITY).fillna(0.5).values
    out["p_attack"] = p_attack.round(4)
    out["anomaly_score"] = anom.round(4)
    out["is_anomaly"] = (bundle["iso"].predict(X) == -1).astype(int)
    # Heuristic blend (documented in README): 50% attack probability, 30% anomaly, 20% class severity
    out["risk_score"] = (100 * (0.5 * p_attack + 0.3 * anom + 0.2 * sev * p_attack)).round(1)
    out["risk_level"] = risk_level(out["risk_score"])
    return out
