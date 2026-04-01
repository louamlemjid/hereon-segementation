# ImageProcessing/service/image_service.py
import base64
import io
import numpy as np
import torch
from PIL import Image, ImageFile

from model import build_custom_unet

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Setup device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load model and move it to the device
model = build_custom_unet()

# NOTE: PyTorch cannot read Keras .h5 files directly. 
# You need to save your PyTorch weights as a .pth file first.
model.load_state_dict(torch.load("./model_weights.pth", map_location=device))
model.to(device)
model.eval()  # Set the model to inference mode


def base64_to_image(base64_str):
    try:
        base64_str = base64_str.strip()
        decoded = base64.b64decode(base64_str)

        if len(decoded) == 0:
            raise ValueError("Decoded image is empty!")

        # Open image and resize
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        img = img.resize((128, 128))

        # Convert to numpy and normalize
        img_array = (np.array(img) / 255.0).astype(np.float32)

        # Keras to PyTorch shape conversion: (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Convert to torch tensor and add batch dimension -> (1, C, H, W)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return img_tensor.to(device)

    except Exception as e:
        print("Failed to decode image:", e)
        raise e


def image_to_base64(mask_array):
    # Denormalize back to 0-255
    mask_array = (mask_array * 255).astype(np.uint8)

    # PyTorch output will be (1, 128, 128). Squeeze removes the channel dim.
    img = Image.fromarray(mask_array.squeeze(), mode="L")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_image(base64_str):
    img_tensor = base64_to_image(base64_str)

    # Run inference without tracking gradients (saves memory/speed)
    with torch.no_grad():
        mask_tensor = model(img_tensor)

    # Move back to CPU and convert back to a NumPy array for decoding
    mask_array = mask_tensor.cpu().numpy()

    # mask_array shape is (1, 1, 128, 128). 
    # We pass the first image in the batch to the base64 encoder.
    return image_to_base64(mask_array[0])