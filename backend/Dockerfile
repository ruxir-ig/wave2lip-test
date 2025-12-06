FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone first-order-model repository
RUN git clone --depth 1 https://github.com/eyaler/first-order-model /app/first-order-model

# Download model weights
RUN mkdir -p /app/first-order-model && \
    wget --no-check-certificate -nc \
    https://openavatarify.s3.amazonaws.com/weights/vox-adv-cpk.pth.tar \
    -O /app/first-order-model/vox-adv-cpk.pth.tar || true

# Copy application code
COPY . .

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--port", "8000"]
