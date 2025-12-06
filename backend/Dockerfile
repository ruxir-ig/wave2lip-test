FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone first-order-model repository (to /app/first-order-model)
RUN git clone --depth 1 https://github.com/eyaler/first-order-model

# Download model weights into the cloned repo
RUN wget --no-check-certificate -nc \
    https://openavatarify.s3.amazonaws.com/weights/vox-adv-cpk.pth.tar \
    -O /app/first-order-model/vox-adv-cpk.pth.tar

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]