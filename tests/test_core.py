import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from smarthome import config, data, db, ml, pipeline  # noqa: E402


class TestData(unittest.TestCase):
    def test_reproducible(self):
        self.assertTrue(data.generate_dataset(300, 7).equals(data.generate_dataset(300, 7)))
        self.assertFalse(data.generate_dataset(300, 7).equals(data.generate_dataset(300, 8)))

    def test_all_classes_present_and_schema(self):
        df = data.generate_dataset(1500, 1)
        self.assertEqual(set(df["label"]), set(data.CLASSES))
        self.assertEqual(data.validate_upload(df), [])
        self.assertFalse(df[data.FEATURES[:-1]].isna().any().any())

    def test_upload_validation_and_normalisation(self):
        df = data.generate_dataset(50, 1)[data.REQUIRED_UPLOAD]
        self.assertEqual(data.validate_upload(df.drop(columns=["syn_ratio"])), ["syn_ratio"])
        norm = data.normalize_upload(df)
        self.assertIn("device_id", norm.columns)
        self.assertEqual(len(norm), 50)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.paths = dict(data_dir=t / "d", model_dir=t / "m", db_path=t / "d" / "x.db")
        cls.res = pipeline.initialize(n=2500, seed=3, **cls.paths)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_models_saved_and_reload(self):
        self.assertTrue(ml.models_exist(self.paths["model_dir"]))
        self.assertIsNotNone(ml.load_models(self.paths["model_dir"]))
        self.assertIsNone(ml.load_models(Path(self.tmp.name) / "empty"))

    def test_metrics_are_computed_not_hardcoded(self):
        rf = self.res["metrics"]["random_forest"]
        self.assertGreater(rf["accuracy"], 0.5)  # sanity floor only
        cm = rf["confusion_matrix"]
        self.assertEqual(sum(map(sum, cm)), self.res["metrics"]["n_test"])

    def test_scoring_ranges_and_ordering(self):
        b = ml.load_models(self.paths["model_dir"])
        s = ml.score(data.generate_dataset(800, 99), b)
        self.assertTrue(s["risk_score"].between(0, 100).all())
        self.assertTrue(s["anomaly_score"].between(0, 1).all())
        self.assertTrue(set(s["risk_level"]) <= set(config.RISK_LABELS))
        self.assertGreater(s[s.label != "normal"]["risk_score"].mean(), s[s.label == "normal"]["risk_score"].mean() + 20)

    def test_db_persistence_and_status_update(self):
        p = self.paths["db_path"]
        self.assertEqual(len(db.list_batches(p)), 1)
        ev = db.load_events(path=p)
        self.assertEqual(len(ev), self.res["n_events"])
        self.assertTrue((ev["risk_score"] >= config.EVENT_THRESHOLD).all())
        self.assertEqual(db.update_event_status(ev["id"].head(3).tolist(), "resolved", p), 3)
        self.assertEqual((db.load_events(path=p)["status"] == "resolved").sum(), 3)
        with self.assertRaises(ValueError):
            db.update_event_status([1], "bogus", p)
        self.assertEqual(len(db.load_predictions(path=p)), 2500)

    def test_save_scored_second_batch(self):
        b = ml.load_models(self.paths["model_dir"])
        s = ml.score(data.normalize_upload(data.generate_dataset(100, 5)[data.REQUIRED_UPLOAD]), b)  # no label column
        db.save_scored(s, "upload:test.csv", path=self.paths["db_path"])
        self.assertEqual(len(db.list_batches(self.paths["db_path"])), 2)


if __name__ == "__main__":
    unittest.main()
