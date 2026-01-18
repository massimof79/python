"""
API Flask per Predizione Rischio Cardiovascolare
Basato su Regressione Logistica
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

# ============================================================
# CARICAMENTO E ADDESTRAMENTO DEL MODELLO ALL'AVVIO
# ============================================================

print("Caricamento del modello...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

colonne = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=colonne, na_values='?')
df = df.dropna()
df['target'] = (df['target'] > 0).astype(int)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Modello addestrato e pronto!")

# ============================================================
# ENDPOINT API
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Crea DataFrame del paziente
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
        
        # Standardizza
        paziente_scaled = scaler.transform(paziente)
        
        # Predizione
        predizione = model.predict(paziente_scaled)[0]
        probabilita = model.predict_proba(paziente_scaled)[0][1]
        
        return jsonify({
            'prediction': int(predizione),
            'probability': float(probabilita),
            'risk': 'high' if predizione == 1 else 'low',
            'message': 'A RISCHIO' if predizione == 1 else 'SANO'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Logistic Regression'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)