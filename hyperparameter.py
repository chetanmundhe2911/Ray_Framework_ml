import ray
from ray import tune
from ray.tune import Trainable
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna

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

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 1, 20)

    config = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    # Run Ray Tune with the suggested configuration
    analysis = tune.run(
        RandomForestTrainable,
        config=config,
        num_samples=1,  # Use one sample for each trial
        resources_per_trial={"cpu": 1},
        local_dir="./ray_results",  # Directory to store results
        stop={"training_iteration": 1}  # Stop after 1 training iteration for each trial
    )

    # Get the best accuracy from this trial
    return analysis.best_result["accuracy"]

# Create a study object and optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Number of Optuna trials

# Print the best hyperparameters found
print("Best hyperparameters:", study.best_params)

# Shut down Ray
ray.shutdown()
