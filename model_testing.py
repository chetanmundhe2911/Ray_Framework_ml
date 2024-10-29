# model_testing.py

def test_model(model, X_test, y_test):
    from sklearn.metrics import classification_report
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report
