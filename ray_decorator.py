import ray

# Define a simple logging decorator
def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}.")
        return result
    return wrapper

@ray.remote
@log_decorator
def model_training(data):
    # Simulate training process
    print("Training model on data...")
    # Your training logic here
    return "Model trained!"

# Example usage
if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    # Call the remote function
    result = model_training.remote("training_data")
    
    # Get the result
    print(ray.get(result))
