# Import Libraries
import pandas as pd
from dataprep.eda import create_report
from sklearn.preprocessing import StandardScaler


# Load Data to a Dataframe
def load_data(filepath):
    data = pd.read_csv(filepath)
    # Replace spaces with underscores in column names
    data.columns = data.columns.str.replace(' ', '_')
    return data


# Generate EDA Report
def generate_eda_report(data):
    #Generate the EDA report
    report = create_report(data)
    #Save the report to an HTML file
    report.save("eda_reports/wine_quality_eda_report.html")


# Function to detect and remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define lower and upper bound to identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# Preprocess Data
def preprocess_data(data):
    # Preprocessing steps as per the EDA Report
    # Remove Duplicates
    data = data.drop_duplicates()
    # Outlier Handling
    # List of columns to handle outliers for
    columns_with_outliers = ['residual_sugar', 'chlorides', 'sulphates']
    # Remove outliers
    data = remove_outliers_iqr(data, columns_with_outliers)
    # Numerical Scaling
    # Identify numerical columns (exclude 'quality')
    numerical_columns = data.drop('quality', axis=1).columns
    # Standardize the numerical columns (mean=0, variance=1)
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data


# Main
if __name__ == '__main__':
    filepath="data/winequality-red.csv"
    data = load_data(filepath)
    generate_eda_report(data)
