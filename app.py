
from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

FEATURES = [
    "Time"
] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for f in FEATURES:
            values.append(float(request.form[f]))

        x = np.array(values).reshape(1, -1)
        pred = int(model.predict(x)[0])

        result = "⚠️ Fraud Detected!" if pred == 1 else "✔️ Normal Transaction"

        return render_template("index.html", prediction=result, features=FEATURES)

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e), features=FEATURES)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
