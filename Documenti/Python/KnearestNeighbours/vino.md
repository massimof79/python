# Esercizio — Previsione della qualità del vino con K-Nearest Neighbors

## Contesto

Un'azienda vinicola della Toscana vuole automatizzare la valutazione preliminare dei propri vini rossi. Ogni anno, i tecnici analizzano campioni di vino misurando alcune proprietà chimiche e assegnano un punteggio di qualità da 1 a 10. L'azienda vorrebbe usare questi dati storici per prevedere automaticamente la qualità di nuovi campioni, riducendo i tempi e i costi di analisi.

Il dataset utilizzato è il celebre *Wine Quality Dataset* (Cortez et al., 2009), disponibile pubblicamente sul repository UCI Machine Learning Repository all'indirizzo:

```
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

Il file è in formato CSV con separatore `;` e contiene 1599 campioni di vino rosso, descritti da 11 variabili chimiche e da un punteggio di qualità.

Le variabili sono:

| Variabile | Descrizione |
|---|---|
| `fixed acidity` | Acidità fissa (g/dm³) |
| `volatile acidity` | Acidità volatile (g/dm³) |
| `citric acid` | Acido citrico (g/dm³) |
| `residual sugar` | Zucchero residuo (g/dm³) |
| `chlorides` | Cloruri (g/dm³) |
| `free sulfur dioxide` | Diossido di zolfo libero (mg/dm³) |
| `total sulfur dioxide` | Diossido di zolfo totale (mg/dm³) |
| `density` | Densità (g/cm³) |
| `pH` | pH |
| `sulphates` | Solfati (g/dm³) |
| `alcohol` | Gradazione alcolica (% vol) |
| `quality` | **Punteggio di qualità (variabile target)** |

---

## Obiettivo

Scrivere un programma Python che:

1. Carichi il dataset
2. Divida i dati in insieme di addestramento (80%) e insieme di test (20%)
3. Implementi **da zero** un classificatore K-Nearest Neighbors — senza usare la classe `KNeighborsClassifier` di scikit-learn — che preveda la qualità del vino
4. Valuti le prestazioni del modello sull'insieme di test
5. Ripeta la valutazione per K ∈ {1, 3, 5, 7, 9} e individui il valore di K migliore

---

## Requisiti tecnici

Il programma deve essere strutturato nelle seguenti funzioni:

```python
def carica_dati(percorso: str):
    """
    Carica il dataset dal file CSV.
    Restituisce X (features) e y (target) come array NumPy.
    """

def normalizza(X_train, X_test):
    """
    Applica la normalizzazione min-max sulle features.
    I parametri di normalizzazione si calcolano SOLO su X_train
    e si applicano sia a X_train che a X_test.
    Restituisce X_train_norm, X_test_norm.
    """

def distanza_euclidea(a, b):
    """
    Calcola la distanza euclidea tra due vettori a e b.
    """

def knn_predict(X_train, y_train, x_new, k: int):
    """
    Dato un singolo punto x_new, restituisce la previsione
    ottenuta con il metodo KNN usando k vicini.
    """

def valuta_modello(X_train, y_train, X_test, y_test, k: int):
    """
    Calcola l'accuratezza del modello KNN sull'insieme di test
    per un dato valore di k.
    Restituisce un valore tra 0 e 1.
    """
```

> **Nota:** è consentito l'uso di `numpy`, `pandas` e `matplotlib`. Non è consentito importare modelli già pronti da `scikit-learn` o librerie equivalenti.

---

## Output atteso

Il programma deve stampare una tabella nel seguente formato:

```
K | Accuratezza
--------------
1 | 0.5134
3 | 0.5512
5 | 0.5678
7 | 0.5601
9 | 0.5489
```

e produrre un grafico (salvato come `risultati_knn.png`) che mostri l'accuratezza al variare di K.

---

## Domande di riflessione

*Da rispondere in forma scritta in testa al file, come commento Python.*

1. Perché è importante normalizzare le feature prima di applicare KNN?
2. Cosa succede all'accuratezza quando K è molto piccolo? E quando è molto grande?
3. Perché i parametri di normalizzazione si calcolano solo sull'insieme di addestramento e non sull'intero dataset?

---

## Consegna

Un unico file `knn_vino.py` contenente tutto il codice e le risposte alle domande di riflessione come commento iniziale.
