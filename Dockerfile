FROM python:3.9-slim

WORKDIR /app

# Copy requirements and SSL certificate
COPY requirements.txt .
COPY emqxsl-ca.crt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY subscriber.py .

# Run the subscriber
CMD ["python", "subscriber.py"]
