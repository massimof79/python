# ==============================
# PREVISIONE RISCHIO INFORTUNIO
# Random Forest - Tennis
# ==============================

# Import delle librerie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==============================
# 1. CARICAMENTO DATI
# ==============================

# Carica il dataset
df = pd.read_csv("injury_risk_tennis.csv")

# Mostra le prime righe
print("Prime righe del dataset:")
print(df.head())

# Controllo valori mancanti
print("\nValori mancanti:")
print(df.isnull().sum())

# ==============================
# 2. SEPARAZIONE FEATURES / TARGET
# ==============================

X = df.drop("rischio_infortunio", axis=1)  # variabili di input
y = df["rischio_infortunio"]              # target

# ==============================
# 3. INDIVIDUAZIONE VARIABILI CATEGORICHE
# ==============================

categorical_features = [
    "livello",
    "riscaldamento",
    "dolore_pre_sessione"
]

numeric_features = [
    "eta",
    "ore_settimanali_allenamento",
    "durata_sessione_min",
    "carico_percepito",
    "giorni_recupero_ultima_sessione",
    "infortuni_precedenti",
    "sonno_medio_ore"
]

# ==============================
# 4. PREPROCESSING
# ==============================

# OneHotEncoder trasforma le variabili categoriche in numeri
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features)
    ],
    remainder="passthrough"  # lascia invariate le numeriche
)

# ==============================
# 5. DIVISIONE TRAIN / TEST
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 6. CREAZIONE MODELLO
# ==============================

rf = RandomForestClassifier(
    n_estimators=100,   # numero di alberi
    max_depth=5,        # profondità massima
    random_state=42
)

# Pipeline: preprocessing + modello
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", rf)
])

# ==============================
# 7. TRAINING
# ==============================

model.fit(X_train, y_train)

# ==============================
# 8. VALUTAZIONE
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nMatrice di confusione:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 9. FEATURE IMPORTANCE
# ==============================

# Recuperiamo il modello addestrato
rf_model = model.named_steps["classifier"]

importances = rf_model.feature_importances_

# Recuperiamo i nomi delle colonne dopo OneHotEncoding
encoded_cols = model.named_steps["preprocessing"]\
    .transformers_[0][1]\
    .get_feature_names_out(categorical_features)

all_features = list(encoded_cols) + numeric_features

# Creiamo tabella importanza
importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importanza": importances
}).sort_values(by="Importanza", ascending=False)

print("\nImportanza delle variabili:")
print(importance_df)

# ==============================
# 10. TUNING SEMPLICE
# ==============================

rf2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    random_state=42
)

model2 = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", rf2)
])

model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred2)
print("\nAccuracy dopo tuning:", accuracy2)
