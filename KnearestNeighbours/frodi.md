# Compito di Realtà

## Classificazione delle Frodi nelle Transazioni con KNN

**Materia:** Intelligenza Artificiale e Machine Learning
**Tipo di prova:** Compito individuale — Programmazione Python
**Docente:** Prof. Fedeli Massimo

---

## 1. Scenario

Una banca desidera sviluppare un sistema automatico per individuare possibili frodi nelle transazioni elettroniche.

Ogni transazione è descritta da alcune caratteristiche:

* Importo della transazione
* Ora del giorno
* Distanza geografica dal luogo abituale
* Frequenza delle transazioni recenti

Ogni operazione viene classificata come:

* **0 → Transazione legittima**
* **1 → Transazione fraudolenta**

L'obiettivo è utilizzare i dati storici per costruire un modello in grado di prevedere automaticamente se una nuova transazione è sospetta.

---

## 2. Obiettivo

Realizzare un programma Python che:

* Generi o carichi un dataset di transazioni
* Suddivida i dati in training (80%) e test (20%)
* Implementi da zero un classificatore **K-Nearest Neighbors (KNN)**
* Calcoli l'accuratezza del modello
* Valuti le prestazioni per K ∈ {1, 3, 5, 7, 9}
* Identifichi il valore di K migliore
* Produca un grafico dell'accuratezza
* Effettui previsioni su nuove transazioni

---

## 3. Struttura del codice

Il programma deve includere le seguenti funzioni:

* `genera_dati(n)` → genera il dataset
* `normalizza(X_train, X_test)` → normalizzazione min-max
* `distanza_euclidea(a, b)` → distanza tra due punti
* `knn_predict(...)` → predizione KNN
* `valuta_modello(...)` → calcolo accuratezza

**Librerie consentite:**

* numpy
* matplotlib

**Non è consentito usare modelli già pronti (es. scikit-learn)**

---

## 4. Output atteso

### 4.1 Tabella

Il programma deve stampare:

## K | Accuratezza

1 | ...
3 | ...
5 | ...
7 | ...
9 | ...

### 4.2 Grafico

Generare un grafico salvato come:

accuratezza_knn.png

### 4.3 Previsioni

Eseguire previsioni su almeno 3 nuove transazioni, ad esempio:

Transazione 1 → FRODE
Transazione 2 → NORMALE

---

## 5. Domande di riflessione

Rispondere come commento nel codice:

1. Perché KNN è sensibile alla scala delle feature?
2. Cosa succede quando K è molto piccolo o molto grande?
3. Perché è necessario separare training e test?

---

## 6. Consegna

* Nome file: **knn_frodi.py**
* Inserire intestazione con nome, cognome, classe e data
* Caricare il file sul registro elettronico

---

## 7. Criteri di valutazione

| Criterio                   | Punti |
| -------------------------- | ----- |
| Correttezza del codice     | 4     |
| Normalizzazione            | 2     |
| Uso di diversi valori di K | 2     |
| Previsioni                 | 2     |
| Domande di riflessione     | 3     |
| Qualità del codice         | 2     |

**Totale: 15 pu
