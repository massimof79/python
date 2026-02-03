Titolo
Sistema di predizione del rischio di abbandono scolastico mediante regressione logistica.

Un istituto scolastico secondario intende prevenire il fenomeno dell’abbandono scolastico individuando in anticipo gli studenti potenzialmente a rischio.
Per questo motivo viene richiesto di progettare e realizzare un semplice sistema di supporto alle decisioni basato su tecniche di Machine Learning supervisionato.

Il sistema dovrà analizzare dati storici relativi a studenti degli anni precedenti e stimare, per ogni nuovo studente, la probabilità di abbandono scolastico.

Obiettivo del compito
Realizzare, utilizzando il linguaggio Python e la libreria scikit-learn, un modello di classificazione binaria in grado di prevedere se uno studente è:

    0 – non a rischio di abbandono
    1 – a rischio di abbandono

Dati a disposizione
Il dataset fornito contiene informazioni anonime sugli studenti, tra cui:

    -età
    -genere
    -numero di assenze
    -media dei voti
    -numero di ritardi
    -partecipazione ad attività extracurricolari
    -livello di supporto familiare
    -esito finale (abbandono / non abbandono)

Nel dataset sono presenti valori mancanti, indicati con un simbolo speciale.

Attività richieste

-Caricare il dataset in un DataFrame utilizzando la libreria pandas.
-Gestire correttamente i valori mancanti, eliminando le osservazioni incomplete.
-Trasformare la variabile di output in una forma binaria adatta alla classificazione.
-Separare le variabili di input dalla variabile target.
-Suddividere il dataset in training set e test set, utilizzando una suddivisione stratificata.
-Standardizzare le variabili numeriche tramite StandardScaler.
-Addestrare un modello di regressione logistica sui dati di training.
Valutare le prestazioni del modello sui dati di test utilizzando:
    accuratezza
    report di classificazione

Simulare la predizione del rischio di abbandono per un nuovo studente, mostrando:

la classe predetta

la probabilità associata alla previsione.
