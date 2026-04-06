Stima del prezzo di una casa in funzione delle sue caratteristiche.

Usiamo il dataset “Housing” disponibile tramite la libreria scikit-learn, derivato dal classico dataset di Boston Housing, largamente usato in ambito didattico.

Obiettivo costruire un modello di regressione lineare che preveda un valore numerico (prezzo) e integrarlo in una web app.

PARTE 1 – CONTESTO DEL PROBLEMA

Un’agenzia immobiliare vuole uno strumento che fornisca una stima automatica del prezzo di una casa a partire da alcune caratteristiche strutturali e ambientali.

Ogni abitazione è descritta da variabili numeriche come:

RM: numero medio di stanze

LSTAT: percentuale di popolazione a basso reddito nel quartiere

PTRATIO: rapporto studenti/docenti nelle scuole della zona

TAX: imposta immobiliare locale

AGE: percentuale di abitazioni costruite prima del 1940

Variabile target:

MEDV: valore medio delle abitazioni (in migliaia di dollari)
