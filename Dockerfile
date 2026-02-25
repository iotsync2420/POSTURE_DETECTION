FROM python:3.10-bullseye

# We use bullseye instead of slim because it comes with many 
# libraries pre-installed, reducing the chance of download failure.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . .

# Start the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
