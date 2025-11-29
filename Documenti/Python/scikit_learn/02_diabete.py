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

# TODO: Crea una classificazione binaria
# Considera diabete se target > 140
y = # TODO: (y_continuous > 140).astype(int)

# TODO: Stampa informazioni sul dataset
print(f"Numero di campioni: {...}")
print(f"Numero di features: {...}")
print(f"Persone con diabete: {...}")
print(f"Persone senza diabete: {...}")

# TODO: Dividi i dati (test_size=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(...)

# TODO: Crea e addestra il modello LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
# TODO: usa fit()

# TODO: Fai predizioni sul test set
y_pred = # TODO

# TODO: Calcola l'accuratezza
accuracy = accuracy_score(...)
print(f"\nAccuratezza del modello: {accuracy:.2%}")

# TODO: Visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice di confusione:")
print(conf_matrix)

# TODO: Stampa il report di classificazione
print("\nReport di classificazione:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Diabete', 'Diabete']))

# TODO: Testa con un nuovo paziente
nuovo_paziente = [[ 0.05, -0.04, 0.06, -0.04, -0.01, 
                    -0.03, 0.04, 0.00, 0.09, 0.03]]
predizione = # TODO
print(f"\nPredizione per nuovo paziente: {predizione[0]}")
if predizione[0] == 1:
    print("⚠️  Rischio diabete")
else:
    print("✓ Nessun rischio diabete")


## Output atteso
""" ```
Numero di campioni: 442
Numero di features: 10
Persone con diabete: 220
Persone senza diabete: 222

Accuratezza del modello: 75.45%

Matrice di confusione:
[[45 10]
 [17 39]]

Report di classificazione:
              precision    recall  f1-score   support

  No Diabete       0.73      0.82      0.77        55
     Diabete       0.80      0.70      0.74        56

    accuracy                           0.75       111
   macro avg       0.76      0.76      0.76       111
weighted avg       0.76      0.75      0.75       111

Predizione per nuovo paziente: 1
 """