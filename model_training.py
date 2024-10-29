#model_training.py

import ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
@ray.remote
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    # Convert to NumPy arrays
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    model.fit(X_train_np, y_train_np)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy
