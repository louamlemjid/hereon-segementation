# ImageProcessing/service/image_service.py
import base64
import numpy as np
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
from model import build_custom_unet

# Load model and weights once
model = build_custom_unet()
model.load_weights("./model_weights_strade.h5")

def base64_to_image(base64_str):
    try:
        # Remove whitespace/newlines
        base64_str = base64_str.strip()

        # Decode base64
        decoded = base64.b64decode(base64_str)

        if len(decoded) == 0:
            raise ValueError("Decoded image is empty!")

        # Open image
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        img = img.resize((128, 128))
        img_array = (np.array(img) / 255.0).astype(np.float32)
        return np.expand_dims(img_array, axis=0)  # add batch dimension

    except Exception as e:
        print("Failed to decode image:", e)
        raise e
    
def image_to_base64(mask_array):
    mask_array = (mask_array * 255).astype(np.uint8)
    img = Image.fromarray(mask_array.squeeze(), mode="L")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_image(base64_str):
    img_array = base64_to_image(base64_str)
    mask = model.predict(img_array)
    return image_to_base64(mask[0])