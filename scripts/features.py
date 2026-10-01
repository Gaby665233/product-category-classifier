"""Prelucrarea folosită identic de antrenare și predicția interactivă."""

import pandas as pd

NUMERIC_FEATURES = [
    "title_length", "word_count", "digit_count",
    "special_char_count", "uppercase_count", "longest_word_length",
]
FEATURE_COLUMNS = ["Product Title", *NUMERIC_FEATURES]

# Etichete care desemnează aceeași categorie, dar apar scrise diferit în CSV.
CATEGORY_RENAMES = {
    "cpu": "CPUs", "cpus": "CPUs",
    "mobile phone": "Mobile Phones", "mobile phones": "Mobile Phones",
    "fridge": "Fridges", "fridges": "Fridges",
}


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Curăță etichetele și elimină duplicatele care ar denatura testarea."""
    data = raw.copy()
    # Fișierul primit are, de exemplu, numele brut ' Category Label'.
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["Product Title", "Category Label"]).copy()
    data["Product Title"] = data["Product Title"].astype(str).str.strip()
    data["Category Label"] = data["Category Label"].astype(str).str.strip()
    data = data[(data["Product Title"] != "") & (data["Category Label"] != "")].copy()
    standard = data["Category Label"].str.casefold().map(CATEGORY_RENAMES)
    data["Category Label"] = standard.fillna(data["Category Label"])

    # Aceeași denumire apare uneori de mai multe ori în fișier; dacă apare
    # și cu categorii contradictorii, o eliminăm integral (nu ghicim eticheta).
    data["normalized_title"] = data["Product Title"].str.casefold()
    categories_per_title = data.groupby("normalized_title")["Category Label"].transform("nunique")
    data = data[categories_per_title == 1].copy()
    data = data.drop_duplicates(subset=["normalized_title"], keep="first")
    return data.drop(columns="normalized_title").reset_index(drop=True)


def create_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Extrage exclusiv informații disponibile și pentru titluri noi."""
    data = raw.copy()
    title = data["Product Title"].fillna("").astype(str)
    data["title_length"] = title.str.len()
    data["word_count"] = title.str.split().str.len()
    data["digit_count"] = title.apply(lambda s: sum(ch.isdigit() for ch in s))
    data["special_char_count"] = title.apply(
        lambda s: sum(not ch.isalnum() and not ch.isspace() for ch in s)
    )
    data["uppercase_count"] = title.apply(lambda s: sum(ch.isupper() for ch in s))
    data["longest_word_length"] = title.apply(
        lambda s: max((len(word) for word in s.split()), default=0)
    )
    return data


def prepare_product(title: str) -> pd.DataFrame:
    """Pregătește un titlu nou în același format ca datele de antrenare."""
    if not title or not title.strip():
        raise ValueError("Titlul produsului nu poate fi gol.")
    return create_features(pd.DataFrame({"Product Title": [title.strip()]}))[FEATURE_COLUMNS]
