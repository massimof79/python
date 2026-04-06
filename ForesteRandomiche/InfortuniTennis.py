import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Colonne
CAT = ["livello", "riscaldamento", "dolore_pre_sessione"]
NUM = [
    "eta", "ore_settimanali_allenamento", "durata_sessione_min", "carico_percepito",
    "giorni_recupero_ultima_sessione", "infortuni_precedenti", "sonno_medio_ore"
]
TARGET = "rischio_infortunio"

# 1) Carico dati e pulizia minima
df = pd.read_csv("injury_risk_tennis.csv").dropna(subset=[TARGET])

# 2) X e y
X = df[CAT + NUM]
y = df[TARGET]

# 3) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Preprocess + modello (pipeline)
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", "passthrough", NUM),
])

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
])

# 5) Training + valutazione
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 2))
