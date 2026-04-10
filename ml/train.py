import torch
import mlflow
import mlflow.pytorch

def train(model, dataloader, optimizer, loss_fn, epochs=10, device="cuda"):

    model.to(device)

    # Start MLflow experiment run
    mlflow.start_run()

    # (optional but recommended)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("loss_fn", loss_fn.__class__.__name__)

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # log metrics to MLflow
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # =========================
    # SAVE MODEL TO MLFLOW
    # =========================
    mlflow.pytorch.log_model(model, "model")

    mlflow.end_run()