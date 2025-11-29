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

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X Train")
print(X_train)

print("X Train")
print(X_test)

print("Y Train")
print(y_train)

print("Y Test")
print(y_test)


# Crea e addestra il modello
""" Il random_state è un numero "seme" (seed) che controlla la casualità in scikit-learn. Serve a rendere i risultati riproducibili.
Perché serve?
Quando dividi i dati con train_test_split, scikit-learn mescola casualmente i dati prima di dividerli. Senza random_state, ogni volta che esegui il programma ottieni una divisione diversa:
 """
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)



# Fai predizioni sul test set
y_pred = model.predict(X_test)

print("Y Predizione")
print(y_pred)


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