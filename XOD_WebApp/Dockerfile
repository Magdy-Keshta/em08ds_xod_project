FROM python:3.11-slim

WORKDIR /app

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Thread limits (same as your app.py)
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

# System libraries required by OpenCV, Torch and Ultralytics
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    libxkbcommon0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create writable directories
RUN mkdir -p /app/models
RUN mkdir -p /app/data/ultralytics
RUN chmod -R 777 /app/models
RUN chmod -R 777 /app/data

EXPOSE 5000

CMD ["python", "app.py"]