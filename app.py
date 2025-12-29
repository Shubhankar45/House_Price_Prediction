from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_model.joblib")

FEATURES = ['MedInc', 'HouseAge', 'Population', 'AveOccup', 'AveRooms', 'Longitude']

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = []

    for feature in FEATURES:
        val = request.form.get(feature)
        values.append(float(val))

    prediction = model.predict([values])[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
