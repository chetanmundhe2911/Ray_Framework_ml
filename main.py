import ray
from data_processing import generate_synthetic_data, load_data, preprocess_data
from model_training import train_model, evaluate_model
from model_testing import test_model
from prometheus_client import start_http_server, Summary, Counter

# Create Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
MODEL_TRAINING_COUNTER = Counter('model_training_total', 'Total number of model training calls')

@REQUEST_TIME.time()
def main_process():
    # Generate synthetic data
    generate_synthetic_data('synthetic_data.csv')

    # Load and preprocess data
    data = load_data('synthetic_data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train model
    MODEL_TRAINING_COUNTER.inc()  # Increment the model training counter
    model_future = train_model.remote(X_train, y_train)
    model = ray.get(model_future)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model Accuracy: {accuracy}')

    # Test model
    report = test_model(model, X_test, y_test)
    print(report)

if __name__ == "__main__":
    # Initialize Ray
    ray.init(include_dashboard=True)

    # Start Prometheus metrics server
    start_http_server(8000)  # Expose metrics on port 8000

    # Main processing
    main_process()

    # Shutdown Ray
    ray.shutdown()
