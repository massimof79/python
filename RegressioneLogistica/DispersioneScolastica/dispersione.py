import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. CARICAMENTO DEL DATASET
# ============================================================

# Esempio di dataset online (abbandono scolastico simulato)
url = "https://raw.githubusercontent.com/massimo-datasets/student-dropout.csv"

colonne = [
    "age", "gender", "absences", "avg_grade",
    "late_days", "activities", "family_support", "dropout"
]

df = pd.read_csv(url, names=colonne, na_values="?")

print("Dataset originale")
print(df.head())

# ============================================================
# 2. PULIZIA DEI DATI
# ============================================================

# Elimina le righe con valori mancanti
df = df.dropna()

# Trasforma il target in binario
# 0 = non a rischio, 1 = a rischio
df["dropout"] = df["dropout"].astype(int)

print("\nNumero di studenti:", len(df))

# ============================================================
# 3. PREPARAZIONE DEI DATI
# ============================================================

X = df.drop("dropout", axis=1)
y = df["dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. STANDARDIZZAZIONE
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 5. ADDESTRAMENTO DEL MODELLO
# ============================================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================================================
# 6. VALUTAZIONE
# ============================================================

y_pred = model.predict(X_test)

print("\nAccuratezza:", accuracy_score(y_test, y_pred))
print("\nReport di classificazione")
print(classification_report(y_test, y_pred))

# ============================================================
# 7. ESEMPIO DI PREDIZIONE
# ============================================================

studente = {
    "age": 17,
    "gender": 1,
    "absences": 18,
    "avg_grade": 5.4,
    "late_days": 7,
    "activities": 0,
    "family_support": 1
}

studente_df = pd.DataFrame([studente])
studente_scaled = scaler.transform(studente_df)

predizione = model.predict(studente_scaled)[0]
prob = model.predict_proba(studente_scaled)[0][1]

print("\nESEMPIO STUDENTE")
print("Predizione:", "A RISCHIO" if predizione == 1 else "NON A RISCHIO")
print(f"Probabilità di abbandono: {prob:.1%}")
