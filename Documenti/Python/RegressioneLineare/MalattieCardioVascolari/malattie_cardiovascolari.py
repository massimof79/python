"""
Sistema di Predizione del Rischio Cardiovascolare
Versione didattica ESSENZIALE
Algoritmo: Regressione Logistica

L’algoritmo realizza un sistema essenziale di predizione del rischio cardiovascolare basato su tecniche di Machine Learning supervisionato. 
L’obiettivo è classificare ciascun paziente come sano oppure a rischio di malattia cardiaca, utilizzando un modello di regressione logistica 
addestrato su dati clinici reali.
Il punto di partenza è il caricamento del dataset, che contiene misurazioni cliniche e anagrafiche di pazienti, 
come età, sesso, pressione sanguigna, colesterolo e risultati di test diagnostici. 
I dati vengono letti direttamente da una sorgente online e organizzati in un DataFrame. 
Alcuni valori nel dataset originale sono mancanti e indicati con un simbolo speciale; 
Questi vengono convertiti in valori nulli e successivamente eliminati per garantire che il modello lavori solo su osservazioni complete. 
Questa scelta, pur semplice, è funzionale a una versione didattica dell’algoritmo.
Successivamente, la variabile di interesse, cioè il target, viene trasformata in una forma binaria. 

Nel dataset originale il grado di malattia è espresso con più valori interi; 
l’algoritmo li semplifica distinguendo solo tra assenza di malattia e presenza di malattia. 
In questo modo il problema viene ricondotto a una classificazione binaria, più adatta all’uso della regressione logistica e più semplice da interpretare per chi studia.
La fase di preparazione dei dati prevede la separazione tra le variabili di input, che descrivono le caratteristiche del paziente, e l’output, che rappresenta lo stato di salute. Il dataset viene poi suddiviso in un insieme di addestramento e uno di test. La suddivisione è effettuata in modo stratificato, così da mantenere proporzioni simili di pazienti sani e malati in entrambe le parti, evitando distorsioni nella valutazione del modello.
Prima dell’addestramento, i dati vengono standardizzati. 
Le variabili cliniche hanno infatti scale molto diverse tra loro: alcune sono espresse in anni, altre in milligrammi o in valori discreti. La standardizzazione riporta tutte le feature su una scala comparabile, centrata sulla media e con deviazione standard unitaria. Questo passaggio è particolarmente importante per la regressione logistica, che è sensibile alle differenze di scala tra le variabili.
L’addestramento del modello avviene tramite regressione logistica, un algoritmo di classificazione che stima la probabilità che un’osservazione appartenga alla classe “a rischio”. Il modello apprende, a partire dai dati di training, un insieme di pesi che quantificano l’influenza di ciascuna variabile clinica sulla probabilità finale di malattia.
Una volta addestrato, il modello viene valutato sui dati di test, cioè su pazienti mai visti durante l’apprendimento. 
Le previsioni ottenute vengono confrontate con i valori reali per calcolare l’accuratezza, che rappresenta la percentuale di classificazioni corrette. Questa metrica fornisce una prima indicazione dell’efficacia del sistema, pur non esaurendo tutte le possibili valutazioni clinicamente rilevanti.


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


""" 1. `import pandas as pd`  
   Carica la libreria **pandas** e la abbrevia con l’alias `pd`.  
   Serve per leggere, esplorare e manipolare i dati (file CSV, Excel, SQL, ecc.).

2. `from sklearn.model_selection import train_test_split`  
   Importa la funzione `train_test_split` da **scikit-learn**.  Serve a dividere il dataset in due insiemi:  
   - **training set** (dati su cui il modello “impara”)  
   - **test set** (dati su cui verifichiamo le prestazioni).

3. `from sklearn.preprocessing import StandardScaler`  
   Importa la classe `StandardScaler`. Serve a **standardizzare** (media = 0, deviazione standard = 1) le feature numeriche.  
   È utile quando gli algoritmi (come la regressione logistica) sono sensibili alla scala dei dati.

4. `from sklearn.linear_model import LogisticRegression`. - Importa il modello **LogisticRegression**.  
   Nonostante il nome, è un classificatore (non un regressore) che stima la probabilità che un campione appartenga a una classe.

5. `from sklearn.metrics import accuracy_score, classification_report`  
   Importa le metriche per valutare il classificatore:  
   - `accuracy_score`: percentuale di predizioni corrette.  
   - `classification_report`: tabella con precision, recall, F1-score per ogni classe.

Lo script prepara il terreno per un tipico flusso di *machine learning supervisionato* con regressione logistica: caricamento dati → split → scaling → training → valutazione. """

# ============================================================
# 1. CARICAMENTO DEL DATASET
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

colonne = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]


#Legge il csv e carica il dataframe
df = pd.read_csv(url, names=colonne, na_values='?')

#Nel dataset originale, il carattere ? viene usato per indicare valori mancanti.
#Questo parametro dice a pandas:
#“ogni volta che trovi ?, trattalo come un valore mancante (NaN)”.

print("Dataframe: ")
print(df)

df = df.dropna() # Restituisce una nuova copia del DataFrame in cui tutte le righe che contengono almeno un valore mancante (NaN) sono state eliminate.

# Trasformo il target in binario: 0 = sano, 1 = malattia
df['target'] = (df['target'] > 0).astype(int)

#(df['target'] > 0) restituisce true o false a seconda che il valore sia maggiore di 0 o 0
#Trasforma true in 1 e false in 0
#quindi se il valore originale è zero il paziente è sano se è maggiore di zero il paziente è malato.

print("Dataset caricato:", len(df), "pazienti")

# ============================================================
# 2. PREPARAZIONE DEI DATI
# ============================================================


#Suddivide il dataset in due parti: X è il 
X = df.drop('target', axis=1)   #Elimina la colonna Target sulla base dei valori presenti nella riga 1
y = df['target'] #Prende solo la colonna target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


""" print("X Train prima della trasformazione")
print(X_train)
print("X Test prima della trasformazione")
print(X_test) """

#rende confrontabili variabili con scale molto diverse (es. età vs colesterolo);
#evita che una feature domini le altre solo per l’ordine di grandezza;

#Questo oggetto implementa la standardizzazione statistica

""" Queste tre righe **standardizzano** (portano media ≈ 0 e varianza ≈ 1) le colonne numeriche del tuo dataset, facendo attenzione a non “barare” con il test set.

1. `scaler = StandardScaler()`  
   Crea un oggetto `StandardScaler` di scikit-learn.  
   È uno “strumento” che calcolerà media e deviazione standard di ogni colonna.

2. `X_train = scaler.fit_transform(X_train)`  
   - `fit`: osserva i valori di `X_train` e calcola per ogni feature la media e la deviazione standard.  
   - `transform`: sottrae la media e divide per la deviazione standard, così ogni colonna avrà media 0 e varianza 1.  
   - Il risultato è un **nuovo array NumPy** (sempre chiamato `X_train`) che sostituisce il vecchio.

3. `X_test = scaler.transform(X_test)`  
   Applica **la stessa trasformazione** (media e dev.std. appena calcolate sul training) al test set.  
   Non si fa mai `fit` sul test, altrimenti si introdurrebbe **data leakage** (informazioni del test entrerebbero nel modello).

Dopo queste righe sia il training che il test hanno le stesse scale, ma il modello non ha mai “visto” i dati di test durante la fase di addestramento. """

""" Perché la **scala** (l’ordine di grandezza) delle feature influenza molti algoritmi di machine learning, **regressione logistica compresa**. Se non standardizzi:

1. **Pesi (coefficienti) distorti**  
   La regressione logistica minimizza una funzione di perdita che usa la **distanza euclidea** tra punto e iper-piano.  
   Una feature con range 0-100 000 domina su una 0-1: il modello abbassa il suo coefficiente di molto, facendo sembrare la seconda feature “inutile” anche se predittiva.

2. **Convergenza lenta o instabile**  
   I metodi di ottimizzazione (gradient descent, liblinear, l-bfgs) convergono più lentamente quando le feature hanno scale molto diverse; a volte non convergono affatto o oscillano.

3. **Regolarizzazione (C, l1, l2) penalizza in modo sbagliato**  
   I termini di penalità λ|w| trattano tutti i pesi allo stesso modo; se una feature ha valori grandi, il suo peso sarà piccolo in valore assoluto e **non verrà penalizzato** come dovuto.

4. **Stima della distanza nei k-NN, SVM, reti neurali, PCA…**  
   Anche se tu stessi usando un altro modello, la distanza euclidea cambierebbe completamente significato senza scaling.

In sintesi: standardizzare garantisce che **ogni feature contribuisca in base alla sua vera capacità predittiva**, non al suo range numerico, migliorando velocità, stabilità e interpretabilità del modello.
 """
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

""" print("X Train dopo la trasformazione")
print(X_train)
print("X Test dopo la trasformazione")
print(X_test) """

# ============================================================
# 3. ADDESTRAMENTO DEL MODELLO
# ============================================================

model = LogisticRegression(max_iter=1000)

#Addestra il modello
model.fit(X_train, y_train)

# ============================================================
# 4. VALUTAZIONE
# ============================================================

#Effettua una previsione
y_pred = model.predict(X_test)


print("\nAccuratezza:", accuracy_score(y_test, y_pred) , "/1")

# ============================================================
# 5. ESEMPIO DI PREDIZIONE
# ============================================================

# Paziente di esempio
paziente = {
    'age': 60, 'sex': 1, 'cp': 0, 'trestbps': 150, 'chol': 250,
    'fbs': 1, 'restecg': 1, 'thalach': 130, 'exang': 1,
    'oldpeak': 2.3, 'slope': 1, 'ca': 1, 'thal': 3
}

paziente_df = pd.DataFrame([paziente])
paziente_scaled = scaler.transform(paziente_df)

predizione = model.predict(paziente_scaled)[0]

print(predizione)

prob = model.predict_proba(paziente_scaled)[0][1]

print("\nESEMPIO PAZIENTE")
print("Predizione:", "A RISCHIO" if predizione == 1 else "SANO")
print(f"Probabilità di rischio: {prob:.1%}")
