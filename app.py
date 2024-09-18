from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the best model
model_path = "mlruns/243160397281920152/37758a0e99404b89b207ff86ad1058ac/artifacts/model"
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
