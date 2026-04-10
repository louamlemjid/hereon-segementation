import torch
import mlflow
import mlflow.pytorch

from models.model_architecture import CustomUNet
from data.base_loader import SegmentationDataset
from torch.utils.data import DataLoader


def train():

    # -------------------------
    # DATA
    # -------------------------
    dataset = SegmentationDataset("data/train")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # -------------------------
    # MODEL
    # -------------------------
    model = CustomUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # -------------------------
    # MLFLOW START
    # -------------------------
    mlflow.set_experiment("unet_segmentation")

    with mlflow.start_run():

        # log hyperparams
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("optimizer", "Adam")

        epochs = 5

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for x, y in loader:
                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            print(f"Epoch {epoch} loss: {avg_loss:.4f}")

            # log metric
            mlflow.log_metric("loss", avg_loss, step=epoch)

        # save model
        mlflow.pytorch.log_model(model, "model")