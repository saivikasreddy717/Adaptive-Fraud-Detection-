# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose port (if running an API, for example)
EXPOSE 5000

# Default command (adjust as needed)
CMD ["python", "models/pipeline.py"]
