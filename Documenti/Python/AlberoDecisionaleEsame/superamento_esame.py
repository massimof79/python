# Importiamo le librerie necessarie
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Dataset di esempio (20 studenti)
# Caratteristiche: [ore_studio, presenza_%, voto_medio]
X = np.array([
    [2, 60, 55],   # Studente 1
    [5, 85, 70],   # Studente 2
    [3, 70, 58],   # Studente 3
    [8, 95, 85],   # Studente 4
    [4, 75, 65],   # Studente 5
    [1, 50, 45],   # Studente 6
    [7, 90, 80],   # Studente 7
    [3, 65, 52],   # Studente 8
    [6, 88, 75],   # Studente 9
    [2, 55, 48],   # Studente 10
    [9, 98, 90],   # Studente 11
    [4, 80, 68],   # Studente 12
    [5, 82, 72],   # Studente 13
    [1, 45, 42],   # Studente 14
    [7, 92, 82],   # Studente 15
    [3, 68, 56],   # Studente 16
    [6, 86, 76],   # Studente 17
    [2, 58, 50],   # Studente 18
    [8, 94, 88],   # Studente 19
    [4, 78, 66]    # Studente 20
])

# Etichette (risultati): 0 = Bocciato, 1 = Promosso
y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1])

# Dividiamo i dati in training set (80%) e test set (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== CLASSIFICAZIONE: SUPERAMENTO ESAME ===\n")
print(f"Dati di training: {len(X_train)} studenti")
print(f"Dati di test: {len(X_test)} studenti\n")

# Creiamo e addestriamo il modello (Albero Decisionale)
modello = DecisionTreeClassifier(max_depth=3, random_state=42)
modello.fit(X_train, y_train)

# Facciamo previsioni sui dati di test
previsioni = modello.predict(X_test)

# Valutiamo le prestazioni
accuratezza = accuracy_score(y_test, previsioni)
print(f"Accuratezza del modello: {accuratezza * 100:.1f}%\n")

# Mostriamo i risultati dettagliati
print("=== RISULTATI SUI DATI DI TEST ===")
for i in range(len(X_test)):
    ore, presenza, voto_medio = X_test[i]
    risultato_reale = "Promosso" if y_test[i] == 1 else "Bocciato"
    previsione = "Promosso" if previsioni[i] == 1 else "Bocciato"
    corretto = "✓" if y_test[i] == previsioni[i] else "✗"
    
    print(f"Studente {i+1}: {ore}h studio, {presenza}% presenza, "
          f"voto medio {voto_medio}")
    print(f"  Reale: {risultato_reale} | Previsto: {previsione} {corretto}\n")

# Report dettagliato
print("\n=== REPORT CLASSIFICAZIONE ===")
print(classification_report(y_test, previsioni, 
                           target_names=['Bocciato', 'Promosso']))

# Proviamo a fare una previsione per un nuovo studente
print("\n=== PREVISIONE PER NUOVO STUDENTE ===")
nuovo_studente = np.array([[6, 87, 74]])  # 6 ore, 87% presenza, voto medio 74
previsione_nuovo = modello.predict(nuovo_studente)
risultato = "PROMOSSO" if previsione_nuovo[0] == 1 else "BOCCIATO"

print(f"Nuovo studente: 6 ore studio, 87% presenza, voto medio 74")
print(f"Previsione: {risultato}")