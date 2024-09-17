# Import Libraries
import pandas as pd
from dataprep.eda import create_report

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    # Preprocessing steps
    return data


def generate_eda_report(data):
    #Generate the EDA report
    report = create_report(data)
    #Save the report to an HTML file
    report.save("eda_reports/wine_quality_eda_report.html")


if __name__ == '__main__':
    filepath="data/winequality-red.csv"
    data = load_data(filepath)
    generate_eda_report(data)