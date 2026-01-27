import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Variabili globali
modello = None
colonne_modello = None  # Colonne generate dal One-Hot Encoding


def carica_e_prepara_dati(percorso_csv):
    """
    Carica il dataset e applica One-Hot Encoding
    alle variabili categoriali.
    """
    df = pd.read_csv(percorso_csv)

    # Separiamo la variabile target
    y = df["Esito_richiesta"]
    X = df.drop("Esito_richiesta", axis=1)

    # Trasformazione delle variabili categoriali in colonne binarie
    X = pd.get_dummies(X)

    return X, y


def addestra_modello():
    """
    Addestra un Decision Tree per prevedere
    l'esito di una richiesta di microprestito.
    """
    global modello, colonne_modello

    print("\nAddestramento del modello in corso...")

    X, y = carica_e_prepara_dati("richieste_microprestiti.csv")

    # Salviamo la struttura delle colonne del training
    colonne_modello = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modello = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    modello.fit(X_train, y_train)

    y_pred = modello.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Modello addestrato correttamente.")
    print(f"Accuratezza sul test set: {acc:.2f}")


def effettua_previsione():
    """
    Effettua una previsione utilizzando il modello addestrato.
    """
    global modello, colonne_modello

    if modello is None or colonne_modello is None:
        print("\nDevi prima addestrare il modello.")
        return

    print("\nInserisci i dati del richiedente:")

    dati = {
        "Fascia_età": input("Fascia età (18-25 / 26-40 / 41-60 / >60): "),
        "Situazione_lavorativa": input("Situazione lavorativa: "),
        "Reddito_mensile": input("Reddito mensile (nessuno / basso / medio / alto): "),
        "Storico_creditizio": input("Storico creditizio (assente / regolare / ritardi_passati / insolvenze): "),
        "Importo_richiesto": input("Importo richiesto (molto_basso / basso / medio / alto): "),
        "Finalità_prestito": input("Finalità prestito (formazione / avvio_attività / acquisto_strumenti / altro): ")
    }

    df_input = pd.DataFrame([dati])

    # Applichiamo la stessa codifica One-Hot usata nel training
    df_input = pd.get_dummies(df_input)

    # Riallineamento delle colonne per combaciare con il modello
    df_input = df_input.reindex(columns=colonne_modello, fill_value=0)

    previsione = modello.predict(df_input)

    print(f"\nEsito previsto della richiesta: {previsione[0]}")


def menu():
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
