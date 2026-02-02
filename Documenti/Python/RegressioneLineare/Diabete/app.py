from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("modello_diabete.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    values = np.array([[
        float(data["pregnancies"]),
        float(data["glucose"]),
        float(data["bloodpressure"]),
        float(data["skinthickness"]),
        float(data["insulin"]),
        float(data["bmi"]),
        float(data["dpf"]),
        float(data["age"])
    ]])

    prob = model.predict_proba(values)[0][1]
    pred = int(prob >= 0.5)

    return jsonify({
        "probability": round(float(prob), 3),
        "prediction": pred
    })

if __name__ == "__main__":
    app.run(debug=True)
