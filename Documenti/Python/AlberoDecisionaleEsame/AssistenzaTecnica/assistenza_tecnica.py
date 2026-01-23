import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_FILE = "modello_albero.pkl"


def carica_e_prepara_dati(percorso_csv):
    df = pd.read_csv(percorso_csv)

    encoder_dict = {}
    for colonna in df.columns:
        le = LabelEncoder()
        df[colonna] = le.fit_transform(df[colonna])
        encoder_dict[colonna] = le

    X = df.drop("Priorità", axis=1)
    y = df["Priorità"]

    return X, y, encoder_dict


def addestra_modello():
    print("\nAddestramento del modello in corso...")

    X, y, encoder_dict = carica_e_prepara_dati(
        "richieste_assistenza_esteso_plus.csv"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modello = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    )

    modello.fit(X_train, y_train)

    y_pred = modello.predict(X_test)
    accuratezza = accuracy_score(y_test, y_pred)

    joblib.dump((modello, encoder_dict), MODEL_FILE)

    print("Modello addestrato correttamente.")
    print(f"Accuratezza sul test set: {accuratezza:.2f}")


def effettua_previsione():
    try:
        modello, encoder_dict = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("\nErrore: il modello non è stato ancora addestrato.")
        return

    print("\nInserisci i dati della richiesta:")

    dati = {
        "Tipo_problema": input("Tipo problema (software / hardware / rete): "),
        "Numero_utenti_coinvolti": input("Numero utenti coinvolti (1 / 2-5 / >5): "),
        "Impatto_servizio": input("Impatto sul servizio (basso / medio / alto): "),
        "Urgenza_dichiarata": input("Urgenza dichiarata (bassa / media / alta): ")
    }

    df_input = pd.DataFrame([dati])

    for colonna in df_input.columns:
        df_input[colonna] = encoder_dict[colonna].transform(df_input[colonna])

    previsione = modello.predict(df_input)

    priorita_decoder = encoder_dict["Priorità"]
    risultato = priorita_decoder.inverse_transform(previsione)

    print(f"\nPriorità prevista: {risultato[0]}")


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
