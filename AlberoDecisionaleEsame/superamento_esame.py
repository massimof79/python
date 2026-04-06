""" L’algoritmo implementa un semplice sistema di classificazione supervisionata finalizzato 
a prevedere l’esito di uno studente (promosso o bocciato) 
sulla base di tre variabili osservabili: ore di studio settimanali, percentuale di presenza alle lezioni e voto medio conseguito durante l’anno.
Ogni studente è rappresentato come un vettore numerico di tre componenti, mentre l’etichetta associata è binaria: 1 indica la promozione, 0 la bocciatura.

Il flusso logico è il seguente.
In primo luogo, il dataset viene suddiviso in un training set (80%) e in un test set (20%). 
Questa separazione consente di addestrare il modello su una parte dei dati e di valutarne le prestazioni su esempi mai visti prima, evitando valutazioni eccessivamente ottimistiche.
Successivamente viene creato e addestrato un DecisionTreeClassifier. 
Durante la fase di addestramento, l’albero analizza i dati di training e costruisce una sequenza di regole del tipo “se… allora…”, 
scegliendo a ogni nodo la variabile e la soglia che meglio separano studenti promossi e bocciati. 
Il criterio di scelta è basato sulla riduzione dell’impurità cioè sulla capacità di 
rendere i gruppi risultanti il più possibile omogenei rispetto all’esito finale.

Una volta addestrato, il modello viene utilizzato per:
- effettuare previsioni sugli studenti del test set;
- calcolare l’accuratezza complessiva;
- confrontare, caso per caso, il risultato reale con quello previsto;
- stimare l’esito di un nuovo studente non presente nel dataset iniziale.

Il codice, oltre a produrre una misura quantitativa delle prestazioni, è strutturato per rendere leggibile e interpretabile ogni previsione, aspetto cruciale in un contesto educativo.

Per quanto riguarda la scelta dell’albero decisionale, essa è particolarmente appropriata per questo problema per diversi motivi.
In primo luogo, il fenomeno modellato è naturalmente decisionale: nella pratica scolastica si ragiona spesso in termini di soglie 
(“se studi poco e hai molte assenze, allora…”). L’albero replica esattamente questo schema logico, 
rendendo il modello intuitivo anche per chi non ha competenze avanzate di machine learning.
In secondo luogo, l’albero decisionale è intrinsecamente interpretabile. 
Ogni previsione può essere spiegata seguendo il percorso dai nodi radice alle foglie, 
mostrando quali condizioni hanno portato alla decisione finale. Questo è un vantaggio decisivo rispetto 
a modelli più complessi ma opachi, soprattutto in ambito didattico o valutativo.

Infine, l’albero non richiede normalizzazione delle variabili, gestisce bene dati eterogenei e funziona 
correttamente anche con dataset di dimensioni ridotte, come in questo esempio.

La scelta di limitare la profondità dell’albero a 3 livelli non è casuale.
Un albero troppo profondo tenderebbe a memorizzare i dati di training, 
adattandosi anche al rumore e perdendo capacità di generalizzazione: il classico fenomeno di overfitting. Con soli 20 esempi, questo rischio è particolarmente elevato.

Una profondità pari a 3 rappresenta un compromesso efficace tra:

semplicità del modello, che rimane comprensibile e spiegabile;

capacità espressiva, sufficiente a catturare relazioni non banali tra studio, presenza e rendimento;

robustezza, perché il modello evita di costruire regole eccessivamente specifiche su pochi casi.

In termini concettuali, un albero di profondità 3 corrisponde a una decisione basata su al massimo tre 
condizioni successive, una complessità coerente con il modo in cui un docente o un consiglio di classe ragiona nella realtà. 
È, quindi, una scelta tecnicamente sensata e pedagogicamente elegante.

 """
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


# Proviamo a fare una previsione per un nuovo studente
print("\n=== PREVISIONE PER NUOVO STUDENTE ===")
nuovo_studente = np.array([[6, 87, 74]])  # 6 ore, 87% presenza, voto medio 74
previsione_nuovo = modello.predict(nuovo_studente)
risultato = "PROMOSSO" if previsione_nuovo[0] == 1 else "BOCCIATO"

print(f"Nuovo studente: 6 ore studio, 87% presenza, voto medio 74")
print(f"Previsione: {risultato}")