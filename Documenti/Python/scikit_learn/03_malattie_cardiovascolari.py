from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Simuliamo un dataset cardiologico
np.random.seed(42)

# Carica dataset relativo ai dati del cancro al seno del Winsconsin
data = load_breast_cancer()

print("Data: ")
print(data)

exit


X = data.data[:, :10]  # Prendiamo solo 10 features
y_original = data.target

# Invertiamo le classi per simulare: 1 = rischio, 0 = no rischio
y = 1 - y_original

# Informazioni sul dataset
n_pazienti = X.shape[0]
n_features = X.shape[1]
n_rischio = np.sum(y == 1)
n_sani = np.sum(y == 0)
percentuale_rischio = (n_rischio / n_pazienti) * 100

print("=" * 60)
print("SISTEMA DI SCREENING CARDIOVASCOLARE")
print("=" * 60)
print(f"Numero totale pazienti analizzati: {n_pazienti}")
print(f"Numero di parametri clinici: {n_features}")
print(f"Pazienti a RISCHIO cardiovascolare: {n_rischio}")
print(f"Pazienti NON a rischio: {n_sani}")
print(f"Percentuale pazienti a rischio: {percentuale_rischio:.1f}%")

# Dividi i dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crea e addestra il modello LogisticRegression
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

# Fai predizioni sul test set
y_pred = model.predict(X_test_scaled)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"PERFORMANCE DEL MODELLO")
print(f"{'='*60}")
print(f"Accuratezza del sistema: {accuracy:.2%}")

# Visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n📊 Matrice di Confusione:")
print("                Predetto")
print("              No Risk  |  Risk")
print(f"Reale  No    [[  {conf_matrix[0,0]:3d}   |  {conf_matrix[0,1]:3d}  ]]")
print(f"       Risk  [[  {conf_matrix[1,0]:3d}   |  {conf_matrix[1,1]:3d}  ]]")

# Calcola metriche specifiche per contesto medico
TN, FP, FN, TP = conf_matrix.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"\n🏥 METRICHE CLINICHE:")
print(f"Sensibilità (Recall): {sensitivity:.2%} - capacità di identificare pazienti a rischio")
print(f"Specificità: {specificity:.2%} - capacità di identificare pazienti sani")
print(f"Falsi Negativi (CRITICO): {FN} pazienti a rischio NON identificati ⚠️")
print(f"Falsi Positivi: {FP} pazienti sani identificati erroneamente come a rischio")

# Report di classificazione
print(f"\n{'='*60}")
print("REPORT DETTAGLIATO")
print(f"{'='*60}")
print(classification_report(y_test, y_pred, 
                          target_names=['No Rischio', 'Rischio Cardiovascolare']))

# Testa con 3 nuovi pazienti
print(f"\n{'='*60}")
print("TEST SU NUOVI PAZIENTI")
print(f"{'='*60}")

paziente_1 = [[0.2, -0.5, -0.3, -0.2, -0.4, -0.5, 0.1, 0.3, -0.6, -0.2]]
paziente_2 = [[0.8, 0.6, 0.4, 0.5, 0.7, 0.8, 0.3, -0.2, 0.5, 0.6]]
paziente_3 = [[1.2, 1.1, 0.9, 1.0, 1.3, 1.2, 0.8, -0.5, 1.1, 1.0]]

pazienti_test = [paziente_1, paziente_2, paziente_3]
nomi_pazienti = ["Mario Rossi (45 anni)", "Lucia Verdi (58 anni)", "Giovanni Bianchi (67 anni)"]

for i, (paziente, nome) in enumerate(zip(pazienti_test, nomi_pazienti), 1):
    # Predizione (usiamo i dati già "mock-scaled" forniti nell'esempio)
    predizione = model.predict(paziente)
    probabilita = model.predict_proba(paziente)[0]
    
    print(f"\n👤 Paziente {i}: {nome}")
    print(f"   Predizione: {predizione[0]}")
    print(f"   Probabilità rischio: {probabilita[1]:.1%}")
    
    if predizione[0] == 1:
        if probabilita[1] > 0.8:
            print("   ⚠️  ALTO RISCHIO - Consulto cardiologico urgente")
        else:
            print("   ⚠️  RISCHIO MODERATO - Approfondimenti consigliati")
    else:
        print("   ✓ BASSO RISCHIO - Controlli di routine")

print(f"\n{'='*60}")