import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Variabili globali
modello = None
colonne_modello = None  # Servirà per allineare i dati in fase di previsione


def carica_e_prepara_dati(percorso_csv):
    """
    Carica il dataset e trasforma le variabili categoriali
    in variabili numeriche tramite One-Hot Encoding.
    """

    df = pd.read_csv(percorso_csv)

    # Separiamo la variabile target prima della codifica
    y = df["Priorità"]
    X = df.drop("Priorità", axis=1)


    print("X prima del coding:", X)
    # One-Hot Encoding: ogni categoria diventa una colonna binaria (0/1)
    X = pd.get_dummies(X)

    print("X prima dopo il coding:", X)
    return X, y


def addestra_modello():
    """
    Addestra un Decision Tree per prevedere la priorità
    delle richieste di assistenza.
    """
    global modello, colonne_modello

    print("\nAddestramento del modello in corso...")

    X, y = carica_e_prepara_dati("richieste_assistenza_esteso_plus.csv")

    # Salviamo le colonne generate dal One-Hot Encoding
    colonne_modello = X.columns

    # Suddivisione training/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Creazione modello
    modello = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    )

    # Addestramento
    modello.fit(X_train, y_train)

    # Valutazione
    y_pred = modello.predict(X_test)
    accuratezza = accuracy_score(y_test, y_pred)

    print("Modello addestrato correttamente.")
    print(f"Accuratezza sul test set: {accuratezza:.2f}")


def effettua_previsione():
    """
    Effettua una previsione usando il modello già addestrato.
    """
    global modello, colonne_modello

    if modello is None or colonne_modello is None:
        print("\nErrore: devi prima addestrare il modello.")
        return

    print("\nInserisci i dati della richiesta:")

    dati = {
        "Tipo_problema": input("Tipo problema (software / hardware / rete): "),
        "Numero_utenti_coinvolti": input("Numero utenti coinvolti (1 / 2-5 / >5): "),
        "Impatto_servizio": input("Impatto sul servizio (basso / medio / alto): "),
        "Urgenza_dichiarata": input("Urgenza dichiarata (bassa / media / alta): ")
    }

    df_input = pd.DataFrame([dati])

    # Applichiamo la stessa codifica One-Hot
    df_input = pd.get_dummies(df_input)

    # Riallineiamo le colonne: eventuali colonne mancanti vengono riempite con 0
    df_input = df_input.reindex(columns=colonne_modello, fill_value=0)

    previsione = modello.predict(df_input)

    print(f"\nPriorità prevista: {previsione[0]}")


def menu():
    """
    Menu testuale per usare il sistema.
    """
    while True:
        print("\n--- MENÙ PRINCIPALE ---")
        print("1) Addestra il modello")
        print("2) Effettua una previsione")
        print("0) Esci")

        scelta = input("Seleziona un'opzione: ")

        if scelta == "1":
            addestra_modello()
        elif scelta == "2":
            effettua_previsione()
        elif scelta == "0":
            print("Uscita dal programma.")
            break
        else:
            print("Scelta non valida.")


if __name__ == "__main__":
    menu()
