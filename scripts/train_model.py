"""Antrenează, compară și salvează un model de clasificare a produselor.

Rulează din rădăcina proiectului: python scripts/train_model.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features import FEATURE_COLUMNS, NUMERIC_FEATURES, clean_dataset, create_features

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "IMLP6_TASK_03-products.csv"
MODEL_FILE = ROOT / "models" / "product_classifier.pkl"
RESULTS_FILE = ROOT / "results" / "comparison.csv"
METADATA_FILE = ROOT / "results" / "model_selection.json"


def create_preprocessor(use_numeric: bool) -> ColumnTransformer:
    transformers = [
        ("text", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=18000), "Product Title")
    ]
    if use_numeric:
        transformers.append(("numeric", StandardScaler(with_mean=False), NUMERIC_FEATURES))
    return ColumnTransformer(transformers=transformers)


def main():
    raw = pd.read_csv(DATA_FILE)
    data = clean_dataset(raw)
    print(f"CSV: {len(raw)} rânduri; după curățare și eliminarea titlurilor duplicate: {len(data)}")
    print("Distribuția claselor:", data["Category Label"].value_counts().to_dict())
    featured = create_features(data)
    X, y = featured[FEATURE_COLUMNS], featured["Category Label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVC": LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
    }
    candidates = []
    for use_numeric in (False, True):
        for name, algorithm in algorithms.items():
            label = name + (" + features" if use_numeric else " + TF-IDF only")
            candidate = Pipeline([
                ("preprocessor", create_preprocessor(use_numeric)),
                ("model", clone(algorithm)),
            ])
            candidate.fit(X_train, y_train)
            predicted = candidate.predict(X_test)
            measures = {
                "Model": label,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision macro": precision_score(y_test, predicted, average="macro", zero_division=0),
                "Recall macro": recall_score(y_test, predicted, average="macro", zero_division=0),
                "F1 macro": f1_score(y_test, predicted, average="macro", zero_division=0),
                "F1 weighted": f1_score(y_test, predicted, average="weighted", zero_division=0),
            }
            print("\n", label, {k: round(v, 4) for k, v in measures.items() if k != "Model"})
            print(classification_report(y_test, predicted, zero_division=0))
            candidates.append((measures, use_numeric, name))

    # Alegerea se bazează EXCLUSIV pe setul de test; cifrele rămân valori de evaluare,
    # nu rezultatul modelului după reantrenarea pe toate datele.
    table = pd.DataFrame([item[0] for item in candidates]).sort_values("F1 macro", ascending=False)
    RESULTS_FILE.parent.mkdir(exist_ok=True, parents=True)
    table.to_csv(RESULTS_FILE, index=False)
    best = max(candidates, key=lambda item: item[0]["F1 macro"])
    measures, use_numeric, name = best
    print("\nSelectat:", measures["Model"], "F1 macro pe test:", round(measures["F1 macro"], 4))

    # Refacem modelul ales pe toate datele curate pentru utilizare ulterioară.
    final = Pipeline([
        ("preprocessor", create_preprocessor(use_numeric)),
        ("model", clone(algorithms[name])),
    ])
    final.fit(X, y)
    MODEL_FILE.parent.mkdir(exist_ok=True, parents=True)
    with MODEL_FILE.open("wb") as stream:
        pickle.dump(final, stream)
    import sklearn
    METADATA_FILE.write_text(json.dumps({
        "selected_model": measures["Model"],
        "test_scores": {k: float(v) for k, v in measures.items() if k != "Model"},
        "train_size": len(X_train), "test_size": len(X_test),
        "unique_titles_used": len(data),
        "sklearn_version": sklearn.__version__,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Modelul a fost salvat:", MODEL_FILE)
    print("Tabelul metricilor:", RESULTS_FILE)


if __name__ == "__main__":
    main()
