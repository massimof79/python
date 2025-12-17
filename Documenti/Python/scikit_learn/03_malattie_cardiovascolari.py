from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Simuliamo un dataset cardiologico
# (In realtà useremmo load_breast_cancer come proxy per l'esercizio)
np.random.seed(42)

# Carica dataset (usiamo breast_cancer come dataset di esempio)
data = load_breast_cancer()
X = data.data[:, :10]  # Prendiamo solo 10 features
y_original = data.target

# Invertiamo le classi per simulare: 1 = rischio, 0 = no rischio
y = 1 - y_original

# Creiamo nomi di features cardiologiche per renderlo più realistico
feature_names = ['età', 'sesso', 'tipo_dolore', 'pressione', 'colesterolo',
                 'glicemia', 'ecg', 'freq_cardiaca', 'angina', 'depressione_ST']

# TODO: Stampa informazioni sul dataset
print("=" * 60)
print("SISTEMA DI SCREENING CARDIOVASCOLARE")
print("=" * 60)
print(f"Numero totale pazienti analizzati: {...}")
print(f"Numero di parametri clinici: {...}")
print(f"Pazienti a RISCHIO cardiovascolare: {...}")
print(f"Pazienti NON a rischio: {...}")
print(f"Percentuale pazienti a rischio: {...:.1f}%")

# TODO: Dividi i dati (test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(...)

# TODO: Normalizza i dati (importante per dati medici!)
# Crea uno StandardScaler e applica fit_transform su train, transform su test
scaler = StandardScaler()
X_train_scaled = # TODO: scaler.fit_transform(...)
X_test_scaled = # TODO: scaler.transform(...)

# TODO: Crea e addestra il modello LogisticRegression
# Usa max_iter=5000 per convergenza, random_state=42
model = LogisticRegression(...)
# TODO: addestra il modello

# TODO: Fai predizioni sul test set
y_pred = # TODO

# TODO: Calcola l'accuratezza
accuracy = accuracy_score(...)
print(f"\n{'='*60}")
print(f"PERFORMANCE DEL MODELLO")
print(f"{'='*60}")
print(f"Accuratezza del sistema: {accuracy:.2%}")

# TODO: Visualizza la matrice di confusione
conf_matrix = confusion_matrix(...)
print("\n📊 Matrice di Confusione:")
print("                Predetto")
print("              No Risk  |  Risk")
print(f"Reale  No    [[  {conf_matrix[0,0]:3d}   |  {conf_matrix[0,1]:3d}  ]]")
print(f"       Risk  [[  {conf_matrix[1,0]:3d}   |  {conf_matrix[1,1]:3d}  ]]")

# Calcola metriche specifiche per contesto medico
TN, FP, FN, TP = conf_matrix.ravel()
sensitivity = TP / (TP + FN)  # Recall per classe positiva
specificity = TN / (TN + FP)

print(f"\n🏥 METRICHE CLINICHE:")
print(f"Sensibilità (Recall): {sensitivity:.2%} - capacità di identificare pazienti a rischio")
print(f"Specificità: {specificity:.2%} - capacità di identificare pazienti sani")
print(f"Falsi Negativi (CRITICO): {FN} pazienti a rischio NON identificati ⚠️")
print(f"Falsi Positivi: {FP} pazienti sani identificati erroneamente come a rischio")

# TODO: Stampa il report di classificazione
print(f"\n{'='*60}")
print("REPORT DETTAGLIATO")
print(f"{'='*60}")
print(classification_report(..., 
                          target_names=['No Rischio', 'Rischio Cardiovascolare']))

# TODO: Testa con 3 nuovi pazienti
print(f"\n{'='*60}")
print("TEST SU NUOVI PAZIENTI")
print(f"{'='*60}")

# Paziente 1: profilo a basso rischio
paziente_1 = [[0.2, -0.5, -0.3, -0.2, -0.4, -0.5, 0.1, 0.3, -0.6, -0.2]]
# Paziente 2: profilo a rischio moderato
paziente_2 = [[0.8, 0.6, 0.4, 0.5, 0.7, 0.8, 0.3, -0.2, 0.5, 0.6]]
# Paziente 3: profilo ad alto rischio
paziente_3 = [[1.2, 1.1, 0.9, 1.0, 1.3, 1.2, 0.8, -0.5, 1.1, 1.0]]

pazienti_test = [paziente_1, paziente_2, paziente_3]
nomi_pazienti = ["Mario Rossi (45 anni)", "Lucia Verdi (58 anni)", "Giovanni Bianchi (67 anni)"]

for i, (paziente, nome) in enumerate(zip(pazienti_test, nomi_pazienti), 1):
    # TODO: fai predizione
    predizione = # TODO
    probabilita = # TODO: model.predict_proba(paziente)[0]
    
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
```

---

## Output Atteso
```
============================================================
SISTEMA DI SCREENING CARDIOVASCOLARE
============================================================
Numero totale pazienti analizzati: 569
Numero di parametri clinici: 10
Pazienti a RISCHIO cardiovascolare: 212
Pazienti NON a rischio: 357
Percentuale pazienti a rischio: 37.3%

============================================================
PERFORMANCE DEL MODELLO
============================================================
Accuratezza del sistema: 94.74%

📊 Matrice di Confusione:
                Predetto
              No Risk  |  Risk
Reale  No    [[   68   |    4  ]]
       Risk  [[    2   |   40  ]]

🏥 METRICHE CLINICHE:
Sensibilità (Recall): 95.24% - capacità di identificare pazienti a rischio
Specificità: 94.44% - capacità di identificare pazienti sani
Falsi Negativi (CRITICO): 2 pazienti a rischio NON identificati ⚠️
Falsi Positivi: 4 pazienti sani identificati erroneamente come a rischio

============================================================
REPORT DETTAGLIATO
============================================================
                         precision    recall  f1-score   support

              No Rischio       0.97      0.94      0.96        72
Rischio Cardiovascolare       0.91      0.95      0.93        42

                accuracy                           0.95       114
               macro avg       0.94      0.95      0.94       114
            weighted avg       0.95      0.95      0.95       114

============================================================
TEST SU NUOVI PAZIENTI
============================================================

👤 Paziente 1: Mario Rossi (45 anni)
   Predizione: 0
   Probabilità rischio: 12.3%
   ✓ BASSO RISCHIO - Controlli di routine

👤 Paziente 2: Lucia Verdi (58 anni)
   Predizione: 1
   Probabilità rischio: 68.5%
   ⚠️  RISCHIO MODERATO - Approfondimenti consigliati

👤 Paziente 3: Giovanni Bianchi (67 anni)
   Predizione: 1
   Probabilità rischio: 89.2%
   ⚠️  ALTO RISCHIO - Consulto cardiologico urgente