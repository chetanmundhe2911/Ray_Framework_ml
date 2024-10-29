import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

def generate_synthetic_data(filename='synthetic_data.csv'):
    # Generate synthetic dataset
    X, y = make_classification(n_samples=100000, n_features=20, 
                               n_informative=15, n_redundant=5, 
                               random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    data['Outcome'] = y
    data.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")

def load_data(filename):
    # Load the synthetic dataset
    return pd.read_csv(filename)

def preprocess_data(data):
    X = data.drop('Outcome', axis=1).values.copy()
    y = data['Outcome'].values.copy()
    y = y.astype(int)  # Ensure y is of type int
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
