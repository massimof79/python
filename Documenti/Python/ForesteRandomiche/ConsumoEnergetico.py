# ==========================================
# SOLUZIONE: Consumo energetico aule (RF)
# File atteso: consumo_aule.csv
# Target: consumo_energetico (basso/medio/alto)
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1) CARICAMENTO DATI
df = pd.read_csv("consumo_aule.csv")

print("Prime righe:")
print(df.head())
print("\nValori mancanti per colonna:")
print(df.isnull().sum())


# 2) PULIZIA MINIMA (semplice e robusta)
# - Rimuovo righe con target mancante (non ha senso tenerle)
df = df.dropna(subset=["consumo_energetico"]).copy()

# - Imputazione molto semplice per eventuali mancanti:
#   numeriche -> mediana, categoriche -> moda
categorical_cols = ["uso_luci", "uso_climatizzazione", "giorno_settimana", "attivita"]
numeric_cols = [
    "numero_studenti",
    "ore_utilizzo_aula",
    "numero_dispositivi_elettronici",
    "temperatura_esterna",
]

# (se qualche colonna non esiste, errore esplicito)
required_cols = set(categorical_cols + numeric_cols + ["consumo_energetico"])
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"Mancano nel CSV queste colonne richieste: {sorted(missing)}")

# Imputazione numeriche
for c in numeric_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

# Imputazione categoriche
for c in categorical_cols:
    if df[c].isnull().any():
        mode_val = df[c].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) > 0 else "sconosciuto"
        df[c] = df[c].fillna(mode_val)

print("\nDistribuzione target (consumo_energetico):")
print(df["consumo_energetico"].value_counts(normalize=True))


# 3) FEATURES / TARGET
X = df[categorical_cols + numeric_cols]
y = df["consumo_energetico"]


# 4) TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # utile se le classi sono sbilanciate
)


# 5) PREPROCESSING: OneHot sulle categoriche, passthrough sulle numeriche
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# 6) MODELLO: Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

# Pipeline completa: preprocessing + modello
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", rf)
])


# 7) TRAINING
model.fit(X_train, y_train)


# 8) VALUTAZIONE
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

print("\nMatrice di confusione:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))


# 9) FEATURE IMPORTANCE (con nomi delle feature dopo OneHot)
# Recupero i nomi delle feature
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_cols)

all_feature_names = list(cat_feature_names) + numeric_cols

importances = model.named_steps["rf"].feature_importances_

importance_df = (
    pd.DataFrame({"feature": all_feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 15 feature per importanza:")
print(importance_df.head(15))


# 10) PREDIZIONE SU UN CASO NUOVO (ESEMPIO)
nuovo_caso = pd.DataFrame([{
    "uso_luci": "si",
    "uso_climatizzazione": "no",
    "giorno_settimana": "feriale",
    "attivita": "laboratorio",
    "numero_studenti": 22,
    "ore_utilizzo_aula": 5,
    "numero_dispositivi_elettronici": 18,
    "temperatura_esterna": 12.0
}])

pred_nuovo = model.predict(nuovo_caso)[0]
proba_nuovo = model.predict_proba(nuovo_caso)[0]
classi = model.named_steps["rf"].classes_

print("\nEsempio previsione nuovo caso:", pred_nuovo)
print("Probabilità per classe:", dict(zip(classi, proba_nuovo)))
