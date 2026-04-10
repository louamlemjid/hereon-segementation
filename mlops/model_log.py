import mlflow
import mlflow.pytorch
import torch

from hereon_segementation.imageProcessing.model import build_custom_unet
# print(hereon_segementation.__file__)

# # Set tracking server (adjust if using remote MLflow)
mlflow.set_tracking_uri("http://localhost:5050")

# # Create experiment
mlflow.set_experiment("segmentation_unet")

# # Build model
model = build_custom_unet()

# # (Optional) load pretrained weights
# # model.load_state_dict(torch.load("model.pth"))

model.eval()

with mlflow.start_run(run_name="unet_v1"):
    # Log parameters (optional but important)
    mlflow.log_param("model_type", "CustomUNet")
    mlflow.log_param("input_size", "3xHxW")

    # Log the model as artifact
    mlflow.pytorch.log_model(model, "model")

    print("Model logged successfully!")
