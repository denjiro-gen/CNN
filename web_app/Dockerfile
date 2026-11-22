FROM python:3.11-slim

# Install system dependencies (OpenCV & image support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose web port
EXPOSE 5000

ENV KERAS_HOME=/app/.keras_cache
ENV TF_ENABLE_ONEDNN_OPTS=0

CMD ["python", "app.py", "5000"]
