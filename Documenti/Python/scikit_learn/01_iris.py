from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carica il dataset Iris
iris = load_iris()

""" X - Le Features (Caratteristiche)
X contiene le caratteristiche misurabili di ogni fiore. È una matrice con 150 righe (fiori) e 4 colonne:

Lunghezza del sepalo (sepal length in cm)
Larghezza del sepalo (sepal width in cm)
Lunghezza del petalo (petal length in cm)
Larghezza del petalo (petal width in cm)


"" Lunghezza e larghezza del sepalo
"" Lunghezza e larghezza del petalo

Esempio di una riga: [5.1, 3.5, 1.4, 0.2] """

""" y - Il Target (Etichetta)
y contiene la specie di ogni fiore. È un array con 150 valori che possono essere:

0 = Iris Setosa
1 = Iris Versicolor
2 = Iris Virginica """

NomiFiori = iris.target_names

print("Nomi Fiori")
print(NomiFiori)

X = iris.data           #Dati originali
y = iris.target         #Possibili specie per ogni fiore

print("X")
print(X)
print("Y")
print(y)

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" Questo codice serve a suddividere un dataset in due parti: una per l’addestramento e una per il test di un modello di machine learning.

X contiene le feature (le variabili indipendenti)
y contiene le etichette o valori target

La funzione train_test_split della libreria scikit-learn:
assegna l’80 percento dei dati al training set (X_train, y_train)
assegna il 20 percento al test set (X_test, y_test), grazie a test_size=0.2
mescola i dati in modo casuale prima della divisione
rende la suddivisione riproducibile grazie a random_state=42 (ogni esecuzione produce la stessa divisione)
In sintesi:
il modello  viene addestrato su X_train e y_train, e poi valutato su dati mai visti prima (X_test, y_test), evitando così l’illusione ottica dell’overfitting.
"""

print("X Train")
print(X_train)


print("X Test")
print(X_test)

print("Y Train")
print(y_train)

print("Y Test")
print(y_test)


exit
# Crea e addestra il modello
""" Il random_state è un numero "seme" (seed) che controlla la casualità in scikit-learn. Serve a rendere i risultati riproducibili.
Perché serve?
Quando dividi i dati con train_test_split, scikit-learn mescola casualmente i dati prima di dividerli. Senza random_state, ogni volta che esegui il programma ottieni una divisione diversa:
 """
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


""" "Cosa fa il metodo fit()
"Apprendimento dai dati: Prende in input i dati di addestramento (features X e target y) e calcola i parametri interni del modello (ad es., i coefficienti in una regressione lineare).
"Costruzione del modello: Memorizza la conoscenza acquisita dai dati, creando un modello pronto per le predizioni.
"Standardizzazione: È un metodo comune a tutte le classi di scikit-learn (es. LinearRegression, Perceptron, RandomForestClassifier), garantendo un'API uniforme. 


"Il metodo predict è il meccanismo con cui un modello di machine learning già addestrato produce una previsione a partire da nuovi dati.

"Predict fa una predizione basata sui dati di test 
 """

# Fai predizioni sul test set
y_pred = model.predict(X_test)

print("Y Predizione")
print(y_pred)

""" "Confronta il livello di accuratezza tra i valori in """

# Calcola e stampa l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2%}")

# Testa con un nuovo esempio
nuovo_fiore = [[5.1, 3.5, 1.4, 0.2]]
predizione = model.predict(nuovo_fiore)
print(f"Specie prevista: {iris.target_names[predizione[0]]}")

# Informazioni aggiuntive
print(f"\nNumero di esempi nel training set: {len(X_train)}")
print(f"Numero di esempi nel test set: {len(X_test)}")
print(f"Specie disponibili: {iris.target_names}")