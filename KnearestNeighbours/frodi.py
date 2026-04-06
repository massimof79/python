# Nome: ____________ Cognome: ____________ Classe: _____ Data: __________

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Generazione dati (frodi)
# -------------------------------
def genera_dati(n=300):
    np.random.seed(42)

    importo = np.random.uniform(1, 1000, n)
    ora = np.random.randint(0, 24, n)
    distanza = np.random.uniform(0, 500, n)
    frequenza = np.random.randint(1, 20, n)

    X = np.column_stack((importo, ora, distanza, frequenza))

    y = []
    for i in range(n):
        score = importo[i]*0.5 + distanza[i]*0.3 + frequenza[i]*10

        if score > 600 or (ora[i] < 5 and importo[i] > 500):
            y.append(1)  # frode
        else:
            y.append(0)  # normale

    return X, np.array(y)

# -------------------------------
# 2. Normalizzazione
# -------------------------------
def normalizza(X_train, X_test):
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)

    X_train_norm = (X_train - min_val) / (max_val - min_val)
    X_test_norm = (X_test - min_val) / (max_val - min_val)

    return X_train_norm, X_test_norm

# -------------------------------
# 3. Distanza
# -------------------------------
def distanza_euclidea(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# -------------------------------
# 4. KNN
# -------------------------------
def knn_predict(X_train, y_train, x_new, k):
    distanze = []

    for i in range(len(X_train)):
        d = distanza_euclidea(X_train[i], x_new)
        distanze.append((d, y_train[i]))

    distanze.sort(key=lambda x: x[0])
    vicini = distanze[:k]

    classi = [v[1] for v in vicini]

    return max(set(classi), key=classi.count)

# -------------------------------
# 5. Valutazione
# -------------------------------
def valuta_modello(X_train, y_train, X_test, y_test, k):
    corrette = 0

    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test[i], k)
        if pred == y_test[i]:
            corrette += 1

    return corrette / len(X_test)

# -------------------------------
# MAIN 
# -------------------------------
X, y = genera_dati()

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Normalizzazione
X_train, X_test = normalizza(X_train, X_test)

K_values = [1, 3, 5, 7, 9]
accuratezze = []

print("K | Accuratezza")
print("----------------")

for k in K_values:
    acc = valuta_modello(X_train, y_train, X_test, y_test, k)
    accuratezze.append(acc)
    print(f"{k} | {acc:.4f}")

# Grafico
plt.plot(K_values, accuratezze, marker='o')
plt.xlabel("Valore di K")
plt.ylabel("Accuratezza")
plt.title("KNN - Rilevamento Frodi")
plt.grid()
plt.savefig("accuratezza_knn.png")
plt.show()

# Miglior K
best_k = K_values[np.argmax(accuratezze)]
print("\nMiglior K:", best_k)

# -------------------------------
# Nuove transazioni
# -------------------------------
nuove = np.array([
    [800, 2, 300, 15],
    [50, 14, 10, 2],
    [600, 23, 400, 10]
])

min_val = X_train.min(axis=0)
max_val = X_train.max(axis=0)
nuove = (nuove - min_val) / (max_val - min_val)

for i, t in enumerate(nuove):
    pred = knn_predict(X_train, y_train, t, best_k)
    tipo = "FRODE" if pred == 1 else "NORMALE"
    print(f"Transazione {i+1}: {tipo}")