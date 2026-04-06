from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("modello_case.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    values = np.array([[
        float(data["rm"]),
        float(data["lstat"]),
        float(data["ptratio"]),
        float(data["tax"]),
        float(data["age"])
    ]])

    prediction = model.predict(values)[0]

    return jsonify({
        "predicted_price": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
