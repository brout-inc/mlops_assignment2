import lime
import lime.lime_tabular
import numpy as np
# import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, preprocess_data


# Load Model
model_path = "mlruns/243160397281920152/37758a0e99404b89b207ff86ad1058ac/artifacts/model"
model = mlflow.sklearn.load_model(model_path)


# Load Data
# wine quality dataset
filepath = "data/winequality-red.csv"
# Load dataset
df = load_data(filepath)
# Preprocess dataset
df = preprocess_data(df)
# Split into features and Target
X = df.drop('quality', axis=1)
y = df['quality']


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature names from the wine quality dataset
feature_names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']
# Initialize the LIME explainer for tabular data
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=[str(i) for i in sorted(np.unique(y_train))],  # Class names based on quality scores
    mode='classification'  # Set mode to 'classification'
)
# Select a data point from the test set for explanation (e.g., first instance)
data_point = X_test.iloc[0]
# print(data_point)
# Generate explanation for the selected data point
explanation = explainer.explain_instance(
    data_row=data_point,
    predict_fn=model.predict_proba
)
# Save the explanation as an HTML file if not in a notebook
explanation.save_to_file('xai_reports/lime_explanation_classification.html')
