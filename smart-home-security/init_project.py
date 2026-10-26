"""Run once before launching the dashboard:  python init_project.py [--rows 6000] [--seed 42]"""
import argparse

from smarthome import config, pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=config.DEFAULT_ROWS)
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    a = ap.parse_args()
    print(config.SYNTHETIC_NOTICE)
    res = pipeline.initialize(a.rows, a.seed)
    m = res["metrics"]
    rf, iso = m["random_forest"], m["isolation_forest"]
    print(f"Rows: {m['n_rows']} (train {m['n_train']} / held-out test {m['n_test']})")
    print(f"RandomForest  accuracy={rf['accuracy']:.3f}  macro-F1={rf['macro_f1']:.3f}  (held-out SYNTHETIC test set)")
    print(f"IsolationFor. precision={iso['precision_vs_attack']:.3f} recall={iso['recall_vs_attack']:.3f} "
          f"FPR={iso['false_positive_rate_on_normal']:.3f}")
    print(f"Saved models -> {config.MODEL_DIR}, dataset -> {config.DATASET_PATH}, DB -> {config.DB_PATH} "
          f"({res['n_events']} security events)")
    print("Next: streamlit run app.py")
