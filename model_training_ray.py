import ray
from ray import tune
from ray.tune import Trainable
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize Ray
ray.init()

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class RandomForestTrainable(Trainable):
    def setup(self, config):
        self.model = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            random_state=42
        )

    def step(self):
        # Fit the model
        self.model.fit(X_train, y_train)
        # Make predictions
        predictions = self.model.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        return {"accuracy": accuracy}

# Define the search space for hyperparameters
config = {
    "n_estimators": tune.randint(10, 200),
    "max_depth": tune.randint(1, 20),
}

# Run the hyperparameter tuning
analysis = tune.run(
    RandomForestTrainable,
    config=config,
    num_samples=10,  # Number of different hyperparameter combinations to try
    resources_per_trial={"cpu": 1},
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print("Best hyperparameters found:", best_config)

# Shut down Ray
ray.shutdown()
