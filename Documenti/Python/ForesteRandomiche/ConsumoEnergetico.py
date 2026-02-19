import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Colonne
CAT = ["uso_luci", "uso_climatizzazione", "giorno_settimana", "attivita"]
NUM = ["numero_studenti", "ore_utilizzo_aula", "numero_dispositivi_elettronici", "temperatura_esterna"]
TARGET = "consumo_energetico"

# 1) Carico i dati
df = pd.read_csv("consumo_aule.csv")

# 2) Rimuovo righe senza target
df = df.dropna(subset=[TARGET])

# 3) Separo X e y
X = df[CAT + NUM]
y = df[TARGET]

print(X)
""" Il parametro stratify serve a mantenere la stessa distribuzione delle classi tra training set e test set.

Nel tuo caso la variabile target è consumo_energetico, che ha tre possibili valori: basso, medio, alto.

Quando dividi il dataset in:

80% training
20% test

senza stratify la divisione è casuale. Questo può creare squilibri.

Esempio.

Immagina che nel dataset completo tu abbia:

50% medio
30% basso
20% alto

Se non usi stratify, può succedere che nel test set capiti, per puro caso:

70% medio
25% basso
5% alto """

# 4) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Preprocessing (trasformazione dati)
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", "passthrough", NUM),
])



# Applico la trasformazione ai dati di training
X_train_prep = preprocess.fit_transform(X_train)

print(CAT + NUM)
print(X_train_prep)
# Applico la stessa trasformazione ai dati di test
X_test_prep = preprocess.transform(X_test)

# 6) Modello
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

# Addestramento
model.fit(X_train_prep, y_train)

# 7) Valutazione
pred = model.predict(X_test_prep)
print("Accuracy:", round(accuracy_score(y_test, pred), 2))

# 8) Nuovo caso
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

# Trasformo anche il nuovo caso
nuovo_prep = preprocess.transform(nuovo)

print("Previsione nuovo caso:", model.predict(nuovo_prep)[0])
