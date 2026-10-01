"""Încarcă modelul salvat și clasifică interactiv titlurile introduse."""
from pathlib import Path
import pickle
from features import prepare_product

MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "product_classifier.pkl"


def main():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            "Nu găsesc modelul .pkl. Mai întâi rulează: python scripts/train_model.py"
        )
    # Un fișier pickle poate executa cod la încărcare; folosim doar modelul
    # creat de propriul script, nu fișiere .pkl din surse necunoscute.
    with MODEL_FILE.open("rb") as stream:
        model = pickle.load(stream)
    print("Clasificator de produse. Scrie 'exit' pentru a închide programul.")
    while True:
        try:
            title = input("\nTitlul produsului: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nProgram închis.")
            break
        if title.lower() in {"exit", "quit"}:
            print("Program închis.")
            break
        if not title:
            print("Introdu un titlu valid.")
            continue
        print("Categoria prezisă:", model.predict(prepare_product(title))[0])


if __name__ == "__main__":
    main()
