import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Colonne
CAT = ["uso_luci", "uso_climatizzazione", "giorno_settimana", "attivita"]
NUM = ["numero_studenti", "ore_utilizzo_aula", "numero_dispositivi_elettronici", "temperatura_esterna"]
TARGET = "consumo_energetico"

# 1) Carico i dati
df = pd.read_csv("consumo_aule.csv").dropna(subset=[TARGET])

# 2) X e y
X = df[CAT + NUM]
y = df[TARGET]

# 3) Split (stessa divisione per entrambi i modelli)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Preprocessing unico (uguale per entrambi)
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", "passthrough", NUM),
])

X_train_prep = preprocess.fit_transform(X_train)
X_test_prep = preprocess.transform(X_test)

# 5) Modello 1: Albero decisionale
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train_prep, y_train)
pred_dt = dt.predict(X_test_prep)
acc_dt = accuracy_score(y_test, pred_dt)

# 6) Modello 2: Foresta randomica
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train_prep, y_train)
pred_rf = rf.predict(X_test_prep)
acc_rf = accuracy_score(y_test, pred_rf)

# 7) Confronto
print("\n=== CONFRONTO MODELLI ===")
print("Accuracy Albero decisionale:", round(acc_dt, 2))
print("Accuracy Foresta randomica :", round(acc_rf, 2))

print("\n--- Report Albero decisionale ---")
print(classification_report(y_test, pred_dt))

print("\n--- Report Foresta randomica ---")
print(classification_report(y_test, pred_rf))

# 8) Stessa previsione su un caso nuovo (per vedere differenza pratica)
nuovo = pd.DataFrame([{
    "uso_luci": "si",
    "uso_climatizzazione": "no",
    "giorno_settimana": "feriale",
    "attivita": "laboratorio",
    "numero_studenti": 22,
    "ore_utilizzo_aula": 5,
    "numero_dispositivi_elettronici": 18,
    "temperatura_esterna": 12.0
}])

nuovo_prep = preprocess.transform(nuovo)

pred_nuovo_dt = dt.predict(nuovo_prep)[0]
pred_nuovo_rf = rf.predict(nuovo_prep)[0]

proba_dt = dt.predict_proba(nuovo_prep)[0]
proba_rf = rf.predict_proba(nuovo_prep)[0]

print("\n=== PREVISIONE SU CASO NUOVO ===")
print("Albero decisionale:", pred_nuovo_dt, " | Probabilità:", dict(zip(dt.classes_, proba_dt)))
print("Foresta randomica :", pred_nuovo_rf, " | Probabilità:", dict(zip(rf.classes_, proba_rf)))
