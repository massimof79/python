"""
API Flask per la predizione del rischio cardiovascolare
Modello di Machine Learning: Regressione Logistica
Il modello viene addestrato automaticamente all'avvio del server.
"""

# ===================== LIBRERIE =====================

from flask import Flask, request, jsonify          # Framework web e gestione richieste/risposte JSON
from flask_cors import CORS                        # Abilita richieste da domini diversi (utile con frontend separato)
import pandas as pd                                # Gestione dati tabellari
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   # Standardizzazione delle feature numeriche
from sklearn.linear_model import LogisticRegression

# ===================== INIZIALIZZAZIONE APP =====================

app = Flask(__name__)   # Crea l'applicazione Flask
CORS(app)               # Abilita CORS per tutte le rotte (da modificare in produzione)

# ============================================================
# CARICAMENTO DATI E ADDESTRAMENTO MODELLO ALL'AVVIO
# ============================================================

print("Caricamento del modello in corso...")

# Dataset pubblico UCI Heart Disease (Cleveland)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Nomi delle colonne (il dataset originale ne è privo)
colonne = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

# Caricamento dati: '?' viene interpretato come valore mancante
df = pd.read_csv(url, names=colonne, na_values='?')

# Rimozione righe con valori mancanti (scelta semplice ma riduce i dati disponibili)
df = df.dropna()

# Trasformazione del target in problema binario:
# 0 = nessuna malattia, 1 = presenza di malattia
df['target'] = (df['target'] > 0).astype(int)

# Separazione feature (X) e variabile target (y)
X = df.drop('target', axis=1)
y = df['target']

# Suddivisione in training e test set mantenendo la proporzione delle classi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardizzazione: media 0 e deviazione standard 1
# Il fit viene fatto SOLO sul training set per evitare data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creazione del modello di regressione logistica
# max_iter aumentato per garantire la convergenza dell'algoritmo
model = LogisticRegression(max_iter=1000)

# Addestramento del modello
model.fit(X_train, y_train)

print("Modello addestrato e pronto all'uso.")

# ============================================================
# ENDPOINT API
# ============================================================
#Permette di far richiamare dal client una funzione.


@app.route('/predict', methods=['POST'])
def predict():
    """
    Riceve in input un JSON con i parametri clinici del paziente
    e restituisce la predizione del rischio cardiovascolare.
    """
    try:
        # Lettura del contenuto JSON inviato dal client
        data = request.json

        # Creazione DataFrame con una sola riga (il paziente)
        # I nomi delle chiavi devono coincidere con le feature del modello
        paziente = pd.DataFrame([{
            'age': data['age'],
            'sex': data['sex'],
            'cp': data['cp'],
            'trestbps': data['trestbps'],
            'chol': data['chol'],
            'fbs': data['fbs'],
            'restecg': data['restecg'],
            'thalach': data['thalach'],
            'exang': data['exang'],
            'oldpeak': data['oldpeak'],
            'slope': data['slope'],
            'ca': data['ca'],
            'thal': data['thal']
        }])

        # Applicazione della stessa standardizzazione usata in fase di training
        paziente_scaled = scaler.transform(paziente)

        # Predizione classe (0 = basso rischio, 1 = alto rischio)
        predizione = model.predict(paziente_scaled)[0]

        # Probabilità associata alla classe positiva (malattia)
        probabilita = model.predict_proba(paziente_scaled)[0][1]

        # Risposta JSON restituita al client
        return jsonify({
            'prediction': int(predizione),
            'probability': float(probabilita),
            'risk': 'high' if predizione == 1 else 'low',
            'message': 'A RISCHIO' if predizione == 1 else 'SANO'
        })

    except Exception as e:
        # In caso di errore (input mancante o malformato)
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint di controllo per verificare che il servizio sia attivo.
    Utile per monitoring o sistemi di orchestrazione.
    """
    return jsonify({
        'status': 'ok',
        'model': 'Logistic Regression'
    })


# ============================================================
# AVVIO SERVER
# ============================================================

if __name__ == '__main__':
    # debug=True utile in sviluppo (auto-reload e messaggi di errore dettagliati)
    # Da disabilitare in produzione
    app.run(debug=True, port=5000)
