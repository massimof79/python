from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_FILE = "modello_albero.pkl"

# Caricamento modello ed encoder
modello, encoder_dict = joblib.load(MODEL_FILE)

@app.route("/", methods=["GET", "POST"])
def index():
    risultato = None

    if request.method == "POST":
        dati = {
            "Tipo_problema": request.form["tipo_problema"],
            "Numero_utenti_coinvolti": request.form["utenti"],
            "Impatto_servizio": request.form["impatto"],
            "Urgenza_dichiarata": request.form["urgenza"]
        }

        df_input = pd.DataFrame([dati])
        df_input.columns = df_input.columns.str.strip().str.replace(" ", "_")

        for colonna in df_input.columns:
            df_input[colonna] = encoder_dict[colonna].transform(df_input[colonna])

        previsione = modello.predict(df_input)
        risultato = encoder_dict["Priorità"].inverse_transform(previsione)[0]

    return render_template("index.html", risultato=risultato)

if __name__ == "__main__":
    app.run(debug=True)
