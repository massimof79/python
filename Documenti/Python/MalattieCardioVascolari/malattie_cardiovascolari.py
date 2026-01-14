"""
Sistema di Predizione del Rischio Cardiovascolare
Versione didattica ESSENZIALE
Algoritmo: Regressione Logistica
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. CARICAMENTO DEL DATASET
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

colonne = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]


#Legge il csv e carica il dataframe
df = pd.read_csv(url, names=colonne, na_values='?')

#Nel dataset originale, il carattere ? viene usato per indicare valori mancanti.
#Questo parametro dice a pandas:
#“ogni volta che trovi ?, trattalo come un valore mancante (NaN)”.

print("Dataframe: ")

print(df)

df = df.dropna()

# Trasformo il target in binario: 0 = sano, 1 = malattia
df['target'] = (df['target'] > 0).astype(int)

#(df['target'] > 0) restituisce true o false a seconda che il valore sia maggiore di 0 o 0
#Trasforma true in 1 e false in 0
#quindi se il valore originale è zero il paziente è sano se è maggiore di zero il paziente è malato.

print("Dataset caricato:", len(df), "pazienti")

# ============================================================
# 2. PREPARAZIONE DEI DATI
# ============================================================


#Suddivide il dataset in due parti: X è il 
X = df.drop('target', axis=1)   #Elimina la colonna Target sulla base dei valori presenti nella riga 1
y = df['target'] #Prende solo la colonna target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#rende confrontabili variabili con scale molto diverse (es. età vs colesterolo);
#evita che una feature domini le altre solo per l’ordine di grandezza;

#Questo oggetto implementa la standardizzazione statistica
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 3. ADDESTRAMENTO DEL MODELLO
# ============================================================

model = LogisticRegression(max_iter=1000)

#Addestra il modello
model.fit(X_train, y_train)

# ============================================================
# 4. VALUTAZIONE
# ============================================================

#Effettua una previsione
y_pred = model.predict(X_test)


print("\nAccuratezza:", accuracy_score(y_test, y_pred) , "/1")

# ============================================================
# 5. ESEMPIO DI PREDIZIONE
# ============================================================

# Paziente di esempio
paziente = {
    'age': 60, 'sex': 1, 'cp': 0, 'trestbps': 150, 'chol': 250,
    'fbs': 1, 'restecg': 1, 'thalach': 130, 'exang': 1,
    'oldpeak': 2.3, 'slope': 1, 'ca': 1, 'thal': 3
}

paziente_df = pd.DataFrame([paziente])
paziente_scaled = scaler.transform(paziente_df)

predizione = model.predict(paziente_scaled)[0]

print(predizione)

prob = model.predict_proba(paziente_scaled)[0][1]

print("\nESEMPIO PAZIENTE")
print("Predizione:", "A RISCHIO" if predizione == 1 else "SANO")
print(f"Probabilità di rischio: {prob:.1%}")
