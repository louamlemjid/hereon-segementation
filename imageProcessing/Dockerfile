# 1. Use an official PyTorch image with GPU/CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies (libgl1 is needed for OpenCV/PIL if you use it)
RUN apt-get update && apt-get install -y \
    build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: Removed --ignore-installed so it doesn't break the pre-installed torch in the base image!
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the specific PyTorch weights file we generated
COPY model_weights.pth .
COPY sam_vit_h_4b8939.pth .

# Copy the rest of your app code
COPY . .

EXPOSE 5000

CMD ["python", "main.py"]