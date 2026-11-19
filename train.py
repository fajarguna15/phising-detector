"""Training + evaluasi model deteksi URL phishing.

Contoh:
  python train.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --legit-label 1
  python train.py --data data/sample_synthetic.csv          # hanya smoke test

Label internal: 1 = phishing, 0 = legitimate.
Catatan: pada dataset PhiUSIIL, label 1 = LEGITIMATE dan 0 = PHISHING,
sehingga gunakan --legit-label 1 (default).
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURE_NAMES, extract_features, get_hostname

SEED = 42


def load_data(path, url_col, label_col, legit_label, sample):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    url_col = url_col or cols.get("url")
    label_col = label_col or cols.get("label")
    if url_col is None or label_col is None:
        raise SystemExit(f"Kolom URL/label tidak ditemukan. Kolom tersedia: {list(df.columns)[:10]}")
    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]
    # y: 1 = phishing, 0 = legit
    df["y"] = (df["label"] != legit_label).astype(int)
    before = len(df)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    print(f"Data: {before} baris -> {len(df)} setelah hapus URL duplikat")
    if sample and sample < len(df):
        df = df.sample(sample, random_state=SEED).reset_index(drop=True)
        print(f"Disampling menjadi {len(df)} baris")
    print("Distribusi kelas (1=phishing):", df["y"].value_counts().to_dict())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--url-col")
    ap.add_argument("--label-col")
    ap.add_argument("--legit-label", type=int, default=1, help="nilai label untuk URL legitimate")
    ap.add_argument("--sample", type=int, default=0, help="batasi jumlah baris (0 = semua)")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.data, args.url_col, args.label_col, args.legit_label, args.sample)

    t = time.time()
    X = pd.DataFrame([extract_features(u) for u in df["url"]], columns=FEATURE_NAMES)
    y = df["y"].values
    print(f"Ekstraksi fitur selesai ({time.time() - t:.1f}s), {X.shape[1]} fitur")

    # Split per hostname agar domain yang sama tidak muncul di train DAN test (cegah data leakage)
    groups = df["url"].map(get_hostname)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
    print(f"Train: {len(tr)} | Test: {len(te)} (split berdasarkan hostname)")

    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss", n_jobs=-1, random_state=SEED),
    }

    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        results[name] = {
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
        }
        fitted[name] = (m, pred)
        joblib.dump({"model": m, "features": FEATURE_NAMES, "name": name}, f"models/{name}.joblib")
        print(f"\n=== {name} ===")
        print(classification_report(y_te, pred, target_names=["legit", "phishing"], digits=4))

    best = max(results, key=lambda k: results[k]["f1"])
    model, pred = fitted[best]
    print(f"\nModel terbaik (F1): {best}")

    # Simpan model & metrik
    joblib.dump({"model": model, "features": FEATURE_NAMES, "name": best}, "models/model.joblib")
    with open("reports/metrics.json", "w") as f:
        json.dump({"best": best, "results": results}, f, indent=2)
    pd.DataFrame(results).T.to_csv("reports/model_comparison.csv")

    # Confusion matrix model terbaik
    cm = confusion_matrix(y_te, pred)
    ConfusionMatrixDisplay(cm, display_labels=["legit", "phishing"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best}")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        imp = np.abs(clf.coef_[0])
    order = np.argsort(imp)[-15:]
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(FEATURE_NAMES)[order], imp[order])
    plt.title(f"Top 15 Feature Importance - {best}")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()

    # Analisis False Positive / False Negative
    te_df = df.iloc[te].copy()
    te_df["pred"] = pred
    te_df["proba_phishing"] = model.predict_proba(X_te)[:, 1]
    fp = te_df[(te_df.y == 0) & (te_df.pred == 1)]
    fn = te_df[(te_df.y == 1) & (te_df.pred == 0)]
    fp[["url", "proba_phishing"]].to_csv("reports/false_positives.csv", index=False)
    fn[["url", "proba_phishing"]].to_csv("reports/false_negatives.csv", index=False)
    print(f"\nFalse Positive (legit dikira phishing): {len(fp)}")
    print(fp[["url", "proba_phishing"]].head(5).to_string(index=False))
    print(f"\nFalse Negative (phishing lolos): {len(fn)}")
    print(fn[["url", "proba_phishing"]].head(5).to_string(index=False))
    print("\nOutput tersimpan di models/ dan reports/")


if __name__ == "__main__":
    main()
