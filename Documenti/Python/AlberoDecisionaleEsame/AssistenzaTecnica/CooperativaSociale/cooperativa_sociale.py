import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Variabili globali
modello = None
encoder_dict = None


def carica_e_prepara_dati(percorso_csv):
    df = pd.read_csv(percorso_csv)

    encoder_locali = {}

    # Codifica di tutte le colonne categoriali
    for colonna in df.columns:
        le = LabelEncoder()
        df[colonna] = le.fit_transform(df[colonna])
        encoder_locali[colonna] = le

    X = df.drop("Esito_richiesta", axis=1)
    y = df["Esito_richiesta"]

    return X, y, encoder_locali


def addestra_modello():
    global modello, encoder_dict

    print("\nAddestramento del modello in corso...")

    X, y, encoder_dict = carica_e_prepara_dati("richieste_microprestiti.csv")

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
    global modello, encoder_dict

    if modello is None:
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

    for colonna in df_input.columns:
        le = encoder_dict[colonna]
        df_input[colonna] = le.transform(df_input[colonna])

    previsione = modello.predict(df_input)

    decoder = encoder_dict["Esito_richiesta"]
    risultato = decoder.inverse_transform(previsione)

    print(f"\nEsito previsto della richiesta: {risultato[0]}")


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
