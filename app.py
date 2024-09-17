from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the best model
model_path = "mlruns/685198553198724242/0c9c9f6e6f254e48b74283eefda14c4b/artifacts/best_model"
model = mlflow.sklearn.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        predictions = model.predict(df)
        return jsonify(predictions.tolist())
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
