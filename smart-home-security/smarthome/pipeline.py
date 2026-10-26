"""End-to-end initialisation: generate data -> train -> score -> persist."""
from . import config, data, db, ml


def initialize(n=config.DEFAULT_ROWS, seed=config.DEFAULT_SEED, data_dir=None, model_dir=None, db_path=None):
    from pathlib import Path
    data_dir = Path(data_dir or config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    df = data.generate_dataset(n, seed)
    df.to_csv(data_dir / "synthetic_iot_traffic.csv", index=False)
    metrics = ml.train(df, model_dir=model_dir, seed=seed)
    bundle = ml.load_models(model_dir)
    scored = ml.score(df, bundle)
    db.init_db(db_path, reset=True)
    batch_id, n_events = db.save_scored(scored, f"synthetic_seed{seed}_n{n}", path=db_path)
    return {"metrics": metrics, "batch_id": batch_id, "n_events": n_events}
