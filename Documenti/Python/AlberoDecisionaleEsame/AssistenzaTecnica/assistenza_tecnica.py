import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Nome del file su cui verrà salvato il modello addestrato
# insieme agli encoder utilizzati per le variabili categoriali
MODEL_FILE = "modello_albero.pkl"


def carica_e_prepara_dati(percorso_csv):
    """
    Carica il dataset da file CSV e prepara i dati
    per l'addestramento del modello di Machine Learning.

    Parametri:
    - percorso_csv: stringa con il percorso del file CSV

    Ritorna:
    - X: DataFrame con le feature
    - y: Series con la variabile target
    - encoder_dict: dizionario {nome_colonna: LabelEncoder}
    """

    # Lettura del file CSV tramite Pandas
    df = pd.read_csv(percorso_csv)

    # Dizionario che conterrà un encoder per ogni colonna
    encoder_dict = {}

    # Codifica di tutte le colonne categoriali
    # Ogni valore testuale viene trasformato in un valore numerico
    for colonna in df.columns:
        le = LabelEncoder()
        df[colonna] = le.fit_transform(df[colonna])
        encoder_dict[colonna] = le

    # Separazione delle feature dalla variabile target
    X = df.drop("Priorità", axis=1)  # Variabili indipendenti
    y = df["Priorità"]               # Variabile dipendente

    return X, y, encoder_dict


def addestra_modello():
    """
    Addestra un modello di Decision Tree per la previsione
    della priorità di una richiesta di assistenza.
    """

    print("\nAddestramento del modello in corso...")

    # Caricamento e preparazione dei dati
    X, y, encoder_dict = carica_e_prepara_dati(
        "richieste_assistenza_esteso_plus.csv"
    )

    # Suddivisione del dataset in training set e test set
    # 70% per l'addestramento, 30% per la valutazione
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Creazione del modello ad albero decisionale
    # - criterion="gini": indice di Gini come metrica di impurità
    # - max_depth=4: profondità massima dell'albero
    # - random_state: rende il risultato riproducibile
    modello = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    )

    # Addestramento del modello sui dati di training
    modello.fit(X_train, y_train)

    # Predizione sul test set
    y_pred = modello.predict(X_test)

    # Calcolo dell'accuratezza del modello
    accuratezza = accuracy_score(y_test, y_pred)

    # Salvataggio su file del modello e degli encoder
    joblib.dump((modello, encoder_dict), MODEL_FILE)

    print("Modello addestrato correttamente.")
    print(f"Accuratezza sul test set: {accuratezza:.2f}")


def effettua_previsione():
    """
    Carica il modello addestrato e permette all'utente
    di inserire manualmente i dati per ottenere
    una previsione della priorità.
    """

    # Tentativo di caricamento del modello salvato
    try:
        modello, encoder_dict = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("\nErrore: il modello non è stato ancora addestrato.")
        return

    print("\nInserisci i dati della richiesta:")

    # Raccolta dei dati tramite input utente
    dati = {
        "Tipo_problema": input("Tipo problema (software / hardware / rete): "),
        "Numero_utenti_coinvolti": input("Numero utenti coinvolti (1 / 2-5 / >5): "),
        "Impatto_servizio": input("Impatto sul servizio (basso / medio / alto): "),
        "Urgenza_dichiarata": input("Urgenza dichiarata (bassa / media / alta): ")
    }

    # Creazione di un DataFrame a partire dai dati inseriti
    df_input = pd.DataFrame([dati])

    # Applicazione degli stessi encoder usati in fase di addestramento
    for colonna in df_input.columns:
        df_input[colonna] = encoder_dict[colonna].transform(df_input[colonna])

    # Predizione della priorità tramite il modello
    previsione = modello.predict(df_input)

    # Decodifica del valore numerico in etichetta testuale
    priorita_decoder = encoder_dict["Priorità"]
    risultato = priorita_decoder.inverse_transform(previsione)

    print(f"\nPriorità prevista: {risultato[0]}")


def menu():
    """
    Gestisce il menu principale dell'applicazione
    e l'interazione con l'utente.
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


# Punto di ingresso del programma
# Garantisce che il menu venga eseguito solo
# se il file è avviato direttamente
if __name__ == "__main__":
    menu()
