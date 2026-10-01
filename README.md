# Clasificarea automată a produselor după titlu

Proiect de machine learning pentru un catalog de comerț online. Modelul citește **titlul unui produs** și sugerează una dintre cele 10 categorii curate din setul de date. Include un notebook cu analiza pas cu pas, testare comparativă a algoritmilor și un script de predicție interactivă.

## Structura proiectului

```text
product-category-classifier/
├── data/
│   └── IMLP6_TASK_03-products.csv
├── notebooks/
│   └── product_classification.ipynb
├── scripts/
│   ├── features.py
│   ├── train_model.py
│   └── predict_category.py
├── models/
│   └── product_classifier.pkl
├── results/
│   ├── comparison.csv
│   └── model_selection.json
├── tests/
│   └── test_project.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Date și curățare

Fișierul original are **35.311 rânduri** și coloanele `product ID`, `Product Title`, `Merchant ID`, ` Category Label`, `_Product Code`, `Number_of_Views`, `Merchant Rating`, ` Listing Date  `. Unele antete conțin spații accidentale. Codul aplică `df.columns.str.strip()` înainte de accesarea coloanelor, rezolvând eroarea `KeyError: 'Category Label'`.

Sunt eliminate înregistrările fără titlu ori categorie. Variantele `CPU` / `CPUs`, `Mobile Phone` / `Mobile Phones` și `fridge` / `Fridges` sunt standardizate. Deoarece apar titluri repetate, excludem titlurile cu etichete contradictorii și păstrăm **o singură înregistrare per titlu**. Rezultă **30.822 titluri unice**, cu **10 categorii**. Această curățare are loc **înainte de împărțirea train/test**, pentru a evita apariția aceluiași produs în ambele seturi.

**Caracteristici:** TF-IDF pe unigram-uri și bigram-uri, cu sau fără șase caracteristici extrase exclusiv din titlu: lungimea titlului, numărul de cuvinte, de cifre, de caractere speciale și de majuscule, respectiv lungimea celui mai lung cuvânt. Testăm utilitatea caracteristicilor în loc să o presupunem.

## Rezultatele experimentului

Împărțire stratificată 80/20 (`random_state=42`): **24.657** exemple pentru antrenare și **6.165** pentru test. TF-IDF și scalarea sunt ajustate numai pe datele de antrenament, cu `Pipeline`. Se compară două algoritmi, fiecare cu două variante de caracteristici:

| Algoritm și caracteristici | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Linear SVC + TF-IDF + caracteristici | 0,9633 | 0,9648 | 0,9645 | **0,9645** |
| Linear SVC + TF-IDF | 0,9612 | 0,9637 | 0,9623 | 0,9628 |
| Logistic Regression + TF-IDF + caracteristici | 0,9513 | 0,9515 | 0,9542 | 0,9526 |
| Logistic Regression + TF-IDF | 0,9509 | 0,9515 | 0,9540 | 0,9522 |

Selectăm **Linear SVC + caracteristici** pe baza **F1 macro**, pentru a trata echilibrat categoriile. Scorurile din tabel provin din setul de test, nu din reantrenarea finală. După comparație, antrenăm o nouă instanță a variantei selectate pe toate datele curate și o salvăm în `models/product_classifier.pkl`. Fișierele de rezultate detaliate se află în `results/`, iar notebook-ul prezintă și raportul pe categorii, graficele și matricea de confuzie completă.

Aceasta este evaluarea unei singure împărțiri. Selectarea modelului pe același test este potrivită pentru comparația demonstrativă, dar poate face estimarea finală optimistă; pentru producție ar fi recomandat un set suplimentar de validare ori validare încrucișată și un set final de test complet separat.

## Instalare și rulare locală

Instalează **Python 3.11+**, apoi deschide un terminal în rădăcina proiectului (folderul care conține `README.md`):

```bash
python -m pip install -r requirements.txt
```

Antrenează din nou și salvează modelul:

```bash
python scripts/train_model.py
```

Încearcă predicția interactivă:

```bash
python scripts/predict_category.py
```

Exemplu:

```text
Titlul produsului: olympus e m10 mark iii geh use silber
Categoria prezisă: Digital Cameras
```

Tastează `exit` ca să închizi scriptul. Pentru rularea testelor automate:

```bash
python -m unittest discover -s tests -v
```

### Notebook în Google Colab

1. Deschide `notebooks/product_classification.ipynb` în Google Colab.
2. În panoul **Fișiere**, încarcă fișierul `data/IMLP6_TASK_03-products.csv` (poți încărca doar CSV-ul, în directorul principal al sesiunii Colab).
3. Alege **Runtime → Run all / Execuție → Executați-le pe toate**. Notebook-ul găsește CSV-ul fie în `data/`, fie lângă notebook, fie în `/content/`.
4. După rulare, modelul salvat apare în `models/product_classifier.pkl` în sesiunea Colab. **Descarcă notebook-ul `.ipynb`, nu doar exportul `.py`.**

Modelul `.pkl` inclus a fost creat cu **scikit-learn 1.8.0**. Formatul `pickle` nu este garantat compatibil între toate versiunile: dacă mediul tău folosește altă versiune, recreează modelul rulând `train_model.py` acolo. Nu deschide niciodată modele `.pkl` obținute din surse necunoscute.

## Testarea pe exemplele din temă

Cinci dintre cele șase exemple de verificare au fost prezise conform așteptărilor. Singura nepotrivire observată a fost `smeg sbs8004po`: modelul a returnat `Fridges` în loc de `Fridge Freezers`. Aceasta este o limitare reală, nu am modificat manual predicția.

## Idei de îmbunătățire

- Validare încrucișată și un set de test separat după selecția hiperparametrilor.
- Analiza produselor care nu au suficiente cuvinte relevante în titlu.
- Analiza cazurilor în care categoriile apropiate, ca `Fridges` și `Fridge Freezers`, sunt confundate.
- Testarea unor metode de vectorizare pe secvențe de caractere pentru codurile de produs.
- Colectarea mai multor exemple verificate manual pentru cazurile dificil de clasificat.

## Predare

Acest proiect trebuie publicat într-un **repository GitHub nou, public, dedicat doar acestei teme**, cu structura de foldere de mai sus. Nu îl include ca subfolder în repository-ul vechi cu alte exerciții. Se predă instructorului **linkul repository-ului public**.
