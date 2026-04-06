import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Colonne
CAT = ["livello", "riscaldamento", "dolore_pre_sessione"]
NUM = [
    "eta", "ore_settimanali_allenamento", "durata_sessione_min",
    "carico_percepito", "giorni_recupero_ultima_sessione",
    "infortuni_precedenti", "sonno_medio_ore"
]
TARGET = "rischio_infortunio"

# 1) Carico dati
df = pd.read_csv("injury_risk_tennis.csv").dropna(subset=[TARGET])

X = df[CAT + NUM]
y = df[TARGET]

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Preprocessing comune
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", "passthrough", NUM),
])

# 4) Albero decisionale
tree_model = Pipeline([
    ("prep", preprocess),
    ("tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
])

tree_model.fit(X_train, y_train)
pred_tree = tree_model.predict(X_test)
acc_tree = accuracy_score(y_test, pred_tree)

# 5) Foresta randomica
rf_model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
])

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)

# 6) Confronto
print("\n=== CONFRONTO MODELLI ===")
print("Accuracy Albero decisionale:", round(acc_tree, 2))
print("Accuracy Foresta randomica :", round(acc_rf, 2))
