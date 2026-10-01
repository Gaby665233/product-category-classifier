"""Teste rapide pentru curățare, features și modelul serializat."""
import pickle
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from features import FEATURE_COLUMNS, clean_dataset, prepare_product


class ProductClassifierTests(unittest.TestCase):
    def test_cleanup_removes_whitespace_and_conflicts(self):
        raw = pd.DataFrame({
            "Product Title": ["Test phone", "test phone", "Other CPU", "Other CPU", "Microwave"],
            " Category Label": ["Mobile Phone", "Mobile Phones", "CPU", "Fridges", "Microwaves"],
        })
        clean = clean_dataset(raw)
        self.assertEqual(set(clean["Category Label"]), {"Mobile Phones", "Microwaves"})
        self.assertEqual(len(clean), 2)
        self.assertFalse(any(col.startswith(" ") for col in clean.columns))

    def test_preparing_new_title(self):
        row = prepare_product("SAMSUNG TV 55\" OLED 2026")
        self.assertEqual(row.columns.tolist(), FEATURE_COLUMNS)
        self.assertGreater(int(row.iloc[0]["digit_count"]), 0)

    def test_model_roundtrip_prediction(self):
        path = ROOT / "models" / "product_classifier.pkl"
        self.assertTrue(path.is_file(), "Antrenează modelul dacă fișierul .pkl lipsește")
        with path.open("rb") as stream:
            model = pickle.load(stream)
        self.assertEqual(model.predict(prepare_product("kenwood k20mss15 solo"))[0], "Microwaves")


if __name__ == "__main__":
    unittest.main()
