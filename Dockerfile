FROM python:3.10-slim

# Install Linux graphics dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start the Flask app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]