# =============================================================================
# DOMANDE DI RIFLESSIONE
#
# 1. Perché è importante normalizzare le feature prima di applicare KNN?
#    KNN misura le distanze tra punti. Se una feature ha valori molto grandi
#    (es. "total sulfur dioxide" fino a 300) e un'altra valori piccoli
#    (es. "pH" intorno a 3), la prima dominerebbe il calcolo della distanza
#    rendendo le altre quasi irrilevanti. La normalizzazione mette tutte
#    le feature sulla stessa scala (0-1) così da trattarle equamente.
#
# 2. Cosa succede all'accuratezza quando K è molto piccolo? E molto grande?
#    Con K piccolo (es. K=1) il modello è molto sensibile al singolo punto
#    più vicino: funziona bene sui dati di addestramento ma tende a sbagliare
#    sui dati nuovi (overfitting). Con K grande il modello considera troppi
#    vicini e tende a prevedere sempre la classe più comune, perdendo
#    precisione (underfitting). Il valore ottimale di K sta nel mezzo.
#
# 3. Perché i parametri di normalizzazione si calcolano solo su X_train?
#    Perché il test set deve simulare dati "nuovi" che il modello non ha mai
#    visto. Se usassimo anche il test set per calcolare min e max,
#    staremmo "spiando" i dati di test durante l'addestramento, ottenendo
#    una stima dell'accuratezza falsamente ottimistica (data leakage).
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- 1. Carica i dati ---

def carica_dati(percorso):
    df = pd.read_csv(percorso, sep=";")
    X = df.drop(columns="quality").values
    y = df["quality"].values
    return X, y


# --- 2. Normalizzazione min-max ---

def normalizza(X_train, X_test):
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train_norm = (X_train - X_min) / (X_max - X_min)
    X_test_norm  = (X_test  - X_min) / (X_max - X_min)
    return X_train_norm, X_test_norm


# --- 3. Distanza euclidea ---

def distanza_euclidea(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# --- 4. Previsione KNN per un singolo punto ---

def knn_predict(X_train, y_train, x_new, k):
    distanze = [distanza_euclidea(x_new, x) for x in X_train]
    indici_vicini = np.argsort(distanze)[:k]
    etichette_vicini = y_train[indici_vicini]
    valori, conteggi = np.unique(etichette_vicini, return_counts=True)
    return valori[np.argmax(conteggi)]


# --- 5. Valutazione del modello ---

def valuta_modello(X_train, y_train, X_test, y_test, k):
    previsioni = [knn_predict(X_train, y_train, x, k) for x in X_test]
    corretti = sum(p == v for p, v in zip(previsioni, y_test))
    return corretti / len(y_test)


# --- Programma principale ---

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Caricamento
X, y = carica_dati(URL)

# Divisione train/test 80-20 (senza shuffle per semplicità)
n_train = int(len(X) * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Normalizzazione
X_train, X_test = normalizza(X_train, X_test)

# Valutazione per diversi valori di K
valori_k = [1, 3, 5, 7, 9]
accuratezze = []

print(f"{'K':>2} | Accuratezza")
print("-" * 14)

for k in valori_k:
    acc = valuta_modello(X_train, y_train, X_test, y_test, k)
    accuratezze.append(acc)
    print(f"{k:>2} | {acc:.4f}")

# Grafico
plt.figure(figsize=(7, 4))
plt.plot(valori_k, accuratezze, marker="o", color="steelblue", linewidth=2)
plt.title("Accuratezza del modello KNN al variare di K")
plt.xlabel("Valore di K")
plt.ylabel("Accuratezza")
plt.xticks(valori_k)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("risultati_knn.png")
plt.show()
print("\nGrafico salvato in: risultati_knn.png")

# --- Previsioni su nuovi campioni ---

# Scegliamo il K migliore (quello con accuratezza massima)
k_migliore = valori_k[np.argmax(accuratezze)]
print(f"\nK migliore: {k_migliore} (accuratezza: {max(accuratezze):.4f})")

# Definiamo alcuni campioni inventati con caratteristiche plausibili.
# L'ordine delle colonne è:
# fixed acidity, volatile acidity, citric acid, residual sugar,
# chlorides, free sulfur dioxide, total sulfur dioxide,
# density, pH, sulphates, alcohol

nuovi_campioni = np.array([
    [7.4, 0.70, 0.00, 1.9, 0.076, 11.0,  34.0, 0.9978, 3.51, 0.56,  9.4],  # vino "medio"
    [5.0, 0.20, 0.40, 2.0, 0.040, 30.0,  90.0, 0.9900, 3.60, 0.80, 13.0],  # vino potenzialmente buono
    [9.5, 0.90, 0.10, 2.5, 0.120,  5.0,  20.0, 1.0010, 3.10, 0.40,  8.5],  # vino potenzialmente scarso
])

nomi_campioni = [
    "Campione 1 (caratteristiche medie)",
    "Campione 2 (bassa acidità, alto alcol)",
    "Campione 3 (alta acidità, basso alcol)",
]

# Normalizziamo i nuovi campioni usando i parametri di X_train
X_min = X[:n_train].min(axis=0)
X_max = X[:n_train].max(axis=0)
nuovi_norm = (nuovi_campioni - X_min) / (X_max - X_min)

print("\n--- Previsioni su nuovi campioni ---")
print(f"{'Campione':<40} | Qualità prevista")
print("-" * 55)
for nome, campione in zip(nomi_campioni, nuovi_norm):
    qualita = knn_predict(X_train, y_train, campione, k_migliore)
    print(f"{nome:<40} | {qualita}/10")

# Previsione su un campione preso dal test set e confronto con il valore reale
print("\n--- Verifica su 5 campioni del test set ---")
print(f"{'#':<4} | {'Qualità reale':>13} | {'Qualità prevista':>16}")
print("-" * 40)
for i in range(5):
    previsto = knn_predict(X_train, y_train, X_test[i], k_migliore)
    reale    = y_test[i]
    esito    = "✓" if previsto == reale else "✗"
    print(f"{i+1:<4} | {reale:>13} | {previsto:>16} {esito}")
