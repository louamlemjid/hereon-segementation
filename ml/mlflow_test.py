import mlflow

mlflow.set_experiment("debug_test")

with mlflow.start_run():
    print("RUN STARTED")
    mlflow.log_metric("loss", 0.99)
    print("LOGGED")