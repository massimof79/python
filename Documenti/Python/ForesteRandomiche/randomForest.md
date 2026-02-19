Nel contesto di RandomForestClassifier, il parametro n_estimators indica quanti estimator compongono la foresta.

Ma prima: cos’è un estimator?

In scikit-learn un estimator è un modello che apprende dai dati. Nel caso della Random Forest, l’estimator elementare è un albero decisionale.

Quindi:

una Random Forest non è un singolo modello
è un insieme di molti modelli (alberi)

Ogni albero:

viene addestrato su una versione leggermente diversa dei dati
impara regole proprie
produce una propria previsione

La previsione finale della foresta è una votazione tra tutti gli alberi.

Ora il parametro:

n_estimators = 200

significa:

la foresta sarà composta da 200 alberi decisionali.

Quindi:

200 modelli semplici → una decisione finale più robusta

Più alberi:

riduce la variabilità del modello
aumenta la stabilità delle previsioni
migliora in genere l’accuratezza

ma:

aumenta il tempo di calcolo

In sintesi:

un estimator = un albero decisionale
n_estimators = quanti alberi compongono la foresta

La Random Forest prende la decisione finale facendo votare tutti questi estimator.