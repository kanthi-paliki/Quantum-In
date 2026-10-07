from flask import Flask, request, jsonify, render_template
import numpy as np

from quantum_model import predict


app = Flask(__name__)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def prediction():

    data = request.get_json()

    features = np.array([
        [
            float(data["mean"]),
            float(data["std"]),
            float(data["edge_density"]),
            float(data["texture"])
        ]
    ])

    result = predict(features)

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)