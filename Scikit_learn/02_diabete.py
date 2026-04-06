from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# NOTA: Il dataset load_diabetes() è per regressione
# Useremo invece un dataset binario creato da noi
# basato sui valori di glucosio

# Carica il dataset
diabetes = load_diabetes()
X = diabetes.data
y_continuous = diabetes.target

# Crea una classificazione binaria
# Considera diabete se target > 140
y = (y_continuous > 140).astype(int)

# Stampa informazioni sul dataset
print(f"Numero di campioni: {len(y)}")
print(f"Numero di features: {X.shape[1]}")
print(f"Persone con diabete: {np.sum(y == 1)}")
print(f"Persone senza diabete: {np.sum(y == 0)}")

# Dividi i dati (test_size=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Crea e addestra il modello LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)



""" LogisticRegression
È la classe che implementa la regressione logistica, un modello di classificazione (tipicamente binaria, ma anche multiclasse) che stima la probabilità che un’istanza appartenga a una certa classe usando la funzione logistica (sigmoide).

model = ...
Qui non stai ancora addestrando il modello: stai solo creando l’oggetto e salvandolo nella variabile model.
L’addestramento avverrà più avanti con model.fit(X_train, y_train).

max_iter=1000
La regressione logistica viene addestrata tramite un algoritmo iterativo di ottimizzazione.
Questo parametro indica il numero massimo di iterazioni consentite all’algoritmo per convergere.
 """

model.fit(X_train, y_train)

# Fai predizioni sul test set
y_pred = model.predict(X_test)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuratezza del modello: {accuracy:.2%}")

# Visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice di confusione:")
print(conf_matrix)


""" Cosa fa il codice
pythonconf_matrix = confusion_matrix(y_test, y_pred)
```

La funzione `confusion_matrix()` confronta:
- **y_test**: le etichette reali (ground truth)
- **y_pred**: le predizioni fatte dal modello

E restituisce una matrice 2x2 (per classificazione binaria) che mostra tutti i possibili risultati.

### Struttura della Matrice di Confusione
```
                    PREDETTO
                 No (0)    Sì (1)
REALE    No (0)  [[  TN  |  FP  ]
         Sì (1)   [  FN  |  TP  ]]
```

Dove:
- **TN** (True Negative) = Veri Negativi: predetto NO, realtà NO ✓
- **FP** (False Positive) = Falsi Positivi: predetto SÌ, realtà NO ✗ (Errore Tipo I)
- **FN** (False Negative) = Falsi Negativi: predetto NO, realtà SÌ ✗ (Errore Tipo II)
- **TP** (True Positive) = Veri Positivi: predetto SÌ, realtà SÌ ✓

### Esempio dal tuo codice
```
[[45 10]
 [17 39]]
Interpretazione:

45 = TN: 45 persone senza diabete correttamente identificate
10 = FP: 10 persone senza diabete erroneamente classificate come diabetiche
17 = FN: 17 persone diabetiche NON identificate (errore grave!)
39 = TP: 39 persone diabetiche correttamente identificate

Metriche derivate
Da questa matrice si calcolano:

Accuracy = (TP + TN) / Totale = (39 + 45) / 111 = 75.45%
Precision = TP / (TP + FP) = 39 / (39 + 10) = 79.6%

"Delle persone che ho classificato diabetiche, quante lo sono davvero?"


Recall (Sensibilità) = TP / (TP + FN) = 39 / (39 + 17) = 69.6%

"Di tutte le persone diabetiche, quante ne ho identificate?"


Specificità = TN / (TN + FP) = 45 / (45 + 10) = 81.8%

"Di tutte le persone sane, quante ne ho identificate correttamente?"

 """
# Stampa il report di classificazione
print("\nReport di classificazione:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Diabete', 'Diabete']))

# Testa con un nuovo paziente
nuovo_paziente = [[ 0.05, -0.04, 0.06, -0.04, -0.01, 
                    -0.03, 0.04, 0.00, 0.09, 0.03]]
predizione = model.predict(nuovo_paziente)
print(f"\nPredizione per nuovo paziente: {predizione[0]}")
if predizione[0] == 1:
    print("⚠️  Rischio diabete")
else:
    print("✓ Nessun rischio diabete")