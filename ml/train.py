import torch
import mlflow
import mlflow.pytorch
import os

from models.model_architecture import CustomUNet
from data.base_loader import SegmentationDataset
from torch.utils.data import DataLoader
from mlflow.models.signature import infer_signature


def move_batch(x, y, device):
    return x.to(device), y.to(device)

# -------------------------
# DICE METRIC
# -------------------------
def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    return (2 * (pred * target).sum()) / ((pred + target).sum() + smooth)


def train():
    batch_size = 8 #!!!!!!!!!! change it to 32 to match the colab !!!!!!
    
    # -------------------------
    # DATA
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "dataset/train")

    dataset = SegmentationDataset(root_dir=DATA_DIR, img_size=(128, 128))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------
    # MODEL with gpu
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomUNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # -------------------------
    # SCHEDULER
    # -------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2
    )

    # -------------------------
    # EARLY STOPPING
    # -------------------------
    best_loss = float("inf")
    patience = 5
    counter = 0

    epochs = 50
    
    # -------------------------
    # MLFLOW SETUP
    # -------------------------
    mlflow.set_experiment("unet_segmentation")

    with mlflow.start_run():

        # -------------------------
        # LOG ALL HYPERPARAMS
        # -------------------------
        mlflow.log_params({
            "model": "CustomUNet",
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": batch_size,
            "epochs": epochs,
            "loss_fn": "BCEWithLogitsLoss",
            "image_size": "128x128",
            "dataset": "salt_dataset"
        })

        # -------------------------
        # TAGS (dashboard filters)
        # -------------------------
        mlflow.set_tag("project", "segmentation")
        mlflow.set_tag("model_type", "UNet")
        mlflow.set_tag("framework", "pytorch")

        

        for epoch in range(epochs):

            # -------------------------
            # TRAINING
            # -------------------------
            model.train()
            train_loss = 0

            for x, y in loader:
                x, y = move_batch(x, y, device)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(loader)

            # -------------------------
            # VALIDATION
            # -------------------------
            model.eval()
            val_loss = 0
            dice_total = 0

            with torch.no_grad():
                for x, y in loader:
                    x, y = move_batch(x, y, device)
                    pred = model(x)

                    loss = loss_fn(pred, y)
                    val_loss += loss.item()

                    dice_total += dice_score(pred, y).item()

            val_loss /= len(loader)
            dice_total /= len(loader)

            # -------------------------
            # LOG METRICS
            # -------------------------
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("dice", dice_total, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} dice={dice_total:.4f}")

            # -------------------------
            # LR SCHEDULER
            # -------------------------
            scheduler.step(val_loss)

            # -------------------------
            # EARLY STOPPING + BEST MODEL
            # -------------------------
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0

                model_cpu = model.cpu().eval()

                example_input = torch.randn(1, 3, 128, 128)

                with torch.no_grad():
                    output = model_cpu(example_input)

                signature = infer_signature(
                    example_input.numpy(),
                    output.numpy()
                )

                traced_model = torch.jit.trace(model_cpu, example_input)

                mlflow.pytorch.log_model(
                    traced_model,
                    "model",
                    signature=signature
                )

                # move model back to GPU for next training steps (IMPORTANT)
                model = model.to(device)

                print("✅ Best model saved")

            else:
                counter += 1
                if counter >= patience:
                    print("⛔ Early stopping triggered")
                    break


        print("Training finished")


def main():
    train()


if __name__ == "__main__":
    main()