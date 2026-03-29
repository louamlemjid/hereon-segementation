FROM python:3.10

WORKDIR /app

# System dependencies (important for tensorflow & pillow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Copy weights into the container
COPY model_weights_strade.h5 .

# Upgrade pip + install deps
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 5000

CMD ["python", "main.py"]