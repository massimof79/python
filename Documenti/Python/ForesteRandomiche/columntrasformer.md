ColumnTransformer serve proprio a questo: applicare trasformazioni diverse a gruppi diversi di colonne.

Qui stai dicendo:

alle colonne categoriali applica una trasformazione
alle colonne numeriche non fare nulla

Vediamo i due blocchi.

("cat", OneHotEncoder(handle_unknown="ignore"), CAT)

Questo significa:

prendi le colonne contenute nella lista CAT
applica su di esse il One-Hot Encoding

Il OneHotEncoder trasforma variabili testuali in variabili numeriche binarie.

Esempio:

giorno_settimana = feriale

diventa:

giorno_settimana_feriale = 1
giorno_settimana_festivo = 0
giorno_settimana_prefestivo = 0

Il parametro handle_unknown="ignore" evita errori quando, nei dati nuovi, compare un valore mai visto durante l’addestramento.

Esempio: se in fase di training non esisteva "straordinario" come attività, il modello non va in errore.

Secondo blocco:

("num", "passthrough", NUM)

Qui stai dicendo:

prendi le colonne numeriche contenute in NUM
non trasformarle

passthrough significa “lascia passare così come sono”.

Quindi:

numero_studenti resta numero_studenti
ore_utilizzo_aula resta ore_utilizzo_aula

In sintesi:

le variabili categoriali vengono convertite in numeri
le variabili numeriche restano invariate

Il risultato finale è una nuova matrice completamente numerica che il modello Random Forest può usare per apprendere.